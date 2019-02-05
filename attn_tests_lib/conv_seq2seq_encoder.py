from overrides import overrides
import torch
import torch.nn as nn
from torch.autograd import Variable
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
import numpy as np
from copy import deepcopy


class ListProxy(object):
    def __init__(self, module, prefix, nums):
        self.module = module
        self.prefix = prefix
        self.nums = nums

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(self.nums[i]))


class ConvNetNoEmbeddings(nn.Module):
    def __init__(self, vector_dimension, output_dimension, min_convolution_size, max_convolution_size,
                 train_convolutions, dropout=.5):
        super(ConvNetNoEmbeddings, self).__init__()
        self.input_dim = vector_dimension
        self.conv_sizes = [i for i in range(min_convolution_size, max_convolution_size + 1, 2)]
        assert output_dimension % len(self.conv_sizes) == 0
        for i in self.conv_sizes:
            self.add_module("convs_" + str(i), nn.Conv1d(vector_dimension, output_dimension // len(self.conv_sizes),
                                                         i, stride=1, bias=True))
            temp_conv = getattr(self, "convs_" + str(i))
            if not train_convolutions:
                temp_conv.weight.requires_grad = False
        self.convs = ListProxy(self, 'convs_', self.conv_sizes)
        setattr(self, "convpadding_" + str(min_convolution_size), None)
        self.nonlinearity = nn.ReLU() # was nn.Tanh() before
        self.dropout = nn.Dropout(p=dropout)
        self.output_dimension = output_dimension

    def forward(self, x, mask):
        if mask is not None:
            # padding for the shorter sequences isn't zero on the right unless we do this
            starting_mask = mask.float()
            assert mask.dim() <= x.dim(), ("Passed in mask with dimension " + str(mask.dim()) +
                                           " but output-in-progress only has dimension " + str(x.dim()))
            for i in range(starting_mask.dim()):
                assert starting_mask.size(i) == x.size(i), ("Dimension mismatch: planned output has size " + str(x.size()) +
                                                   ", but provided mask has size " + str(mask.size()))
            while starting_mask.dim() < x.dim():
                starting_mask = starting_mask.unsqueeze(starting_mask.dim())
        x = x * starting_mask.expand(*(x.size()))
        num_instances = x.size(0)
        num_items_in_seq = x.size(1)
        if getattr(self, "convpadding_" + str(self.conv_sizes[0])) is None:
            for i in self.conv_sizes:
                if i >= 3:
                    padding_to_add = x.new_zeros(1, self.input_dim, (int((i - 1) / 2)))
                    setattr(self, "convpadding_" + str(i), padding_to_add)
        # by this point, x is a 3-dim tensor: sentence x word x embedding_dimension
        x = x.permute(0, 2, 1).contiguous().view(num_instances, self.input_dim, num_items_in_seq)
        x = self.dropout(x)
        pieces_of_x = []
        # each piece_of_x_with_padding will be of size batch_size x num_output_dims_div_by_num_convs x seq_len
        for conv_layer, filter_size in zip(self.convs, self.conv_sizes):
            if filter_size >= 3:
                padding = getattr(self, "convpadding_" + str(filter_size))
                padding = padding.expand(num_instances, padding.shape[1], padding.shape[2])
                padding = Variable(padding)
                piece_of_x_with_padding = torch.cat([padding, x, padding], 2)
            else:
                piece_of_x_with_padding = x
            piece_of_x_with_padding = conv_layer(piece_of_x_with_padding)

            piece_of_x_with_padding = self.nonlinearity(piece_of_x_with_padding)
            pieces_of_x.append(piece_of_x_with_padding)
        x = torch.cat(pieces_of_x, 1)
        x = x.permute(0, 2, 1).contiguous().view(num_instances, num_items_in_seq, self.output_dimension)
        if mask is not None:
            mask = mask.float()
            assert mask.dim() <= x.dim(), ("Passed in mask with dimension " + str(mask.dim()) +
                                           " but output-in-progress only has dimension " + str(x.dim()))
            for i in range(mask.dim()):
                assert mask.size(i) == x.size(i), ("Dimension mismatch: planned output has size " + str(x.size()) +
                                                   ", but provided mask has size " + str(mask.size()))
            while mask.dim() < x.dim():
                mask = mask.unsqueeze(mask.dim())
            x = x * mask.expand(*(x.size()))
        return x


@Seq2SeqEncoder.register("convolutional_rnn_substitute")
class ConvSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 min_conv_size: int = 3,
                 max_conv_size: int = 5,
                 dropout: float = 0.3) -> None:
        assert min_conv_size % 2 == 1 and max_conv_size % 2 == 1, "Both min_conv_size and max_conv_size must be odd."
        super(ConvSeq2SeqEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_encoder = ConvNetNoEmbeddings(input_size, hidden_size, min_conv_size, max_conv_size, True,
                                                dropout=dropout)

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,  # not packed
                mask: torch.Tensor,
                hidden_state: torch.Tensor = None) -> torch.Tensor:
        # assume batch is first
        return self.conv_encoder(inputs, mask)

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def get_input_dim(self) -> int:
        return self.input_size

    @overrides
    def get_output_dim(self) -> int:
        return self.hidden_size


def convert_list_of_instances_to_3d_torch_tensor(list_of_instances, third_dim):
    list_of_instances = deepcopy(list_of_instances)
    max_instance_len = 0
    for instance in list_of_instances:
        if len(instance) > max_instance_len:
            max_instance_len = len(instance)
    for instance in list_of_instances:
        while len(instance) < max_instance_len:
            instance.append(0.0)
    twod_arr = np.array(list_of_instances)
    arr_to_mult_by = np.random.uniform(1, 4, [len(list_of_instances), max_instance_len, third_dim])
    mask = torch.from_numpy(twod_arr)
    arr_to_mult_by = torch.from_numpy(arr_to_mult_by)
    all_ones = mask.new_ones(arr_to_mult_by.size())
    return (all_ones * arr_to_mult_by).float(), mask.long()


def test_for_padding_usage():
    input_dimension = 12
    output_dimension = 6
    list_of_instances = [[1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0]]
    conv = ConvSeq2SeqEncoder(input_dimension, output_dimension, dropout=0.0)

    conv.conv_encoder.convs_3.weight.data.copy_(torch.from_numpy(np.random.uniform(1, 4, [output_dimension // 2,
                                                                                          input_dimension, 3])))
    conv.conv_encoder.convs_3.bias.data.copy_(torch.from_numpy(np.random.uniform(1, 4, [output_dimension // 2])))
    conv.conv_encoder.convs_5.weight.data.copy_(torch.from_numpy(np.random.uniform(1, 4, [output_dimension // 2,
                                                                                          input_dimension, 5])))
    conv.conv_encoder.convs_5.bias.data.copy_(torch.from_numpy(np.random.uniform(1, 4, [output_dimension // 2])))

    conv = conv.eval()
    input_tens, mask = convert_list_of_instances_to_3d_torch_tensor(list_of_instances, input_dimension)
    full_to_print = str(conv(input_tens, mask)[0].data.cpu().numpy())
    new_inp_tensor = input_tens[0, :len(list_of_instances[0]), :].view(1, len(list_of_instances[0]), input_tens.size(2))
    new_mask = mask[0, :len(list_of_instances[0])].view(1, len(list_of_instances[0]))
    print(mask)
    print(new_mask)
    print("Dimensions of new input tensor: " + str(new_inp_tensor.size()))
    print(new_inp_tensor)
    print()
    print(full_to_print)
    print("Mask used: " + str(mask.cpu().numpy()))
    print(conv(new_inp_tensor, new_mask)[0].data.cpu().numpy())
    print("Those should both be the same.")


if __name__ == '__main__':
    test_for_padding_usage()
