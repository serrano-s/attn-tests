from overrides import overrides
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders.intra_sentence_attention import IntraSentenceAttentionEncoder
from allennlp.nn import util
import torch
import os
import numpy as np
from attn_tests_lib import SimpleHanAttention


def binary_search_for_num_non_padded_tokens_in_instance(full_array, row_ind):
    assert full_array.dim() == 2
    open_for_checking_start = 0
    open_for_checking_endplus1 = full_array.size(1)
    look_at = (open_for_checking_start + open_for_checking_endplus1) // 2
    first_zero_ind_identified = None
    while first_zero_ind_identified is None:
        if full_array[row_ind, look_at] != 0:
            open_for_checking_start = look_at + 1
        else:
            if full_array[row_ind, look_at - 1] != 0:
                first_zero_ind_identified = look_at
            else:
                open_for_checking_endplus1 = look_at
        if open_for_checking_start == open_for_checking_endplus1:
            assert open_for_checking_endplus1 == full_array.size(1)
            first_zero_ind_identified = full_array.size(1)
        else:
            look_at = (open_for_checking_start + open_for_checking_endplus1) // 2
    return first_zero_ind_identified


@Seq2SeqEncoder.register("talkative_simple_han_attention")
class TalkativeSimpleHanAttention(Seq2SeqEncoder):
    def __init__(self,
                 attn_params: SimpleHanAttention,
                 attn_weight_filename,
                 corr_vector_dir,
                 total_num_test_instances) -> None:
        super(TalkativeSimpleHanAttention, self).__init__()
        self._mlp = attn_params._mlp
        self._context_dot_product = attn_params._context_dot_product
        self.vec_dim = self._mlp.weight.size(1)
        self._total_num_test_instances = total_num_test_instances
        self._attn_weight_filename = attn_weight_filename
        self._input_vector_dir_name = corr_vector_dir
        if not self._input_vector_dir_name.endswith('/'):
            self._input_vector_dir_name += '/'
        self._next_available_counter_ind_file = self._input_vector_dir_name + "next_available_counter.txt"

    @overrides
    def get_input_dim(self) -> int:
        return self.vec_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.vec_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        assert mask is not None
        batch_size, sequence_length, embedding_dim = tokens.size()

        attn_weights = tokens.view(batch_size * sequence_length, embedding_dim)
        attn_weights = torch.tanh(self._mlp(attn_weights))
        attn_weights = self._context_dot_product(attn_weights)
        attn_weights = attn_weights.view(batch_size, -1)  # batch_size x seq_len

        self.report_unnormalized_log_attn_weights(mask * attn_weights, tokens, mask)

        attn_weights = util.masked_softmax(attn_weights, mask)
        attn_weights = attn_weights.unsqueeze(2).expand(batch_size, sequence_length, embedding_dim)

        return tokens * attn_weights

    def report_unnormalized_log_attn_weights(self, attn_weights, input_vects, mask):
        assert attn_weights.dim() == 2, \
            ("Size of attn weights (" + str(attn_weights.size()) + ") indicates multiheaded attention, but " +
             "TalkativeSimpleHanAttention currently assumes single-head attention.")
        if not os.path.isfile(self._attn_weight_filename):
            next_available_ind = 1
            if not os.path.isdir(self._input_vector_dir_name):
                os.makedirs(self._input_vector_dir_name)
            else:
                if len(os.listdir(self._input_vector_dir_name)) != 0:
                    print("ERROR: couldn't find file " + str(self._attn_weight_filename) + ", but " +
                          self._input_vector_dir_name + " exists and is nonempty.")
                    exit(1)
        else:
            assert os.path.isfile(self._next_available_counter_ind_file)
            with open(self._next_available_counter_ind_file, 'r') as f:
                next_available_ind = int(f.readline())
        assert next_available_ind <= self._total_num_test_instances, \
            "Looks like you're overwriting previously saved results."
        input_vects_filename = (self._input_vector_dir_name + str(next_available_ind) + '-' +
                                str(next_available_ind + attn_weights.size(0)))
        np.save(input_vects_filename, input_vects.data.cpu().numpy())
        with open(self._attn_weight_filename, 'a') as f:
            for i in range(attn_weights.size(0)):
                f.write(str(next_available_ind) + ": ")
                num_nonpadding_pieces_in_row = \
                    binary_search_for_num_non_padded_tokens_in_instance(mask, i)
                assert num_nonpadding_pieces_in_row > 0, str(mask)
                weights_to_write = [float(attn_weights[i, j])
                                    for j in range(num_nonpadding_pieces_in_row)]
                min_val = min(weights_to_write)
                max_val = max(weights_to_write)
                val_to_subtract = ((max_val - min_val) / 2) + min_val
                f.write(str(" ".join([str(w - val_to_subtract) for w in weights_to_write])))
                f.write('\n')
                next_available_ind += 1
        with open(self._next_available_counter_ind_file, 'w') as f:
            f.write(str(next_available_ind))
