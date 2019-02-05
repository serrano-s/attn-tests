from overrides import overrides
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders.intra_sentence_attention import IntraSentenceAttentionEncoder
from allennlp.nn import util
import torch
import os
import numpy as np


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


#@Seq2SeqEncoder.register("talkative_intra_sentence_attention")
class TalkativeIntraSentenceAttentionEncoder(Seq2SeqEncoder):
    def __init__(self,
                 encoder_wrapped: Seq2SeqEncoder,
                 attn_weight_filename: str,
                 input_vector_dir_name: str,
                 total_num_test_instances: int) -> None:
        super(TalkativeIntraSentenceAttentionEncoder, self).__init__()
        self._encoder_wrapped = encoder_wrapped
        self._matrix_attention = encoder_wrapped._matrix_attention
        self._num_attention_heads = encoder_wrapped._num_attention_heads
        self._projection = encoder_wrapped._projection
        self._output_projection = encoder_wrapped._output_projection
        self._combination = encoder_wrapped._combination
        self._attn_weight_filename = attn_weight_filename
        self._input_vector_dir_name = input_vector_dir_name
        if not self._input_vector_dir_name.endswith('/'):
            self._input_vector_dir_name += '/'
        self._next_available_counter_ind_file = self._input_vector_dir_name + "next_available_counter.txt"
        self._total_num_test_instances = total_num_test_instances

    @overrides
    def get_input_dim(self) -> int:
        return self._encoder_wrapped.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self._encoder_wrapped.get_output_dim()

    @overrides
    def is_bidirectional(self):
        return self._encoder_wrapped.is_bidirectional()

    @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        assert mask is not None, "TalkativeIntraSentenceAttentionEncoder requires a mask to be provided."
        batch_size, sequence_length, _ = tokens.size()
        # Shape: (batch_size, sequence_length, sequence_length)
        similarity_matrix = self._matrix_attention(tokens, tokens)

        if self._num_attention_heads > 1:
            # In this case, the similarity matrix actually has shape
            # (batch_size, sequence_length, sequence_length, num_heads).  To make the rest of the
            # logic below easier, we'll permute this to
            # (batch_size, sequence_length, num_heads, sequence_length).
            similarity_matrix = similarity_matrix.permute(0, 1, 3, 2)

        # Shape: (batch_size, sequence_length, [num_heads,] sequence_length)
        similarity_matrix = similarity_matrix.contiguous()
        temp_mask = mask.unsqueeze(1)
        # Shape: (batch_size, sequence_length, projection_dim)
        output_token_representation = self._projection(tokens)
        self.report_unnormalized_log_attn_weights(similarity_matrix * temp_mask, output_token_representation)

        intra_sentence_attention = util.masked_softmax(similarity_matrix.contiguous(), mask)

        if self._num_attention_heads > 1:
            # We need to split and permute the output representation to be
            # (batch_size, num_heads, sequence_length, projection_dim / num_heads), so that we can
            # do a proper weighted sum with `intra_sentence_attention`.
            shape = list(output_token_representation.size())
            new_shape = shape[:-1] + [self._num_attention_heads, -1]
            # Shape: (batch_size, sequence_length, num_heads, projection_dim / num_heads)
            output_token_representation = output_token_representation.view(*new_shape)
            # Shape: (batch_size, num_heads, sequence_length, projection_dim / num_heads)
            output_token_representation = output_token_representation.permute(0, 2, 1, 3)

        # Shape: (batch_size, sequence_length, [num_heads,] projection_dim [/ num_heads])
        attended_sentence = util.weighted_sum(output_token_representation,
                                              intra_sentence_attention)

        if self._num_attention_heads > 1:
            # Here we concatenate the weighted representation for each head.  We'll accomplish this
            # just with a resize.
            # Shape: (batch_size, sequence_length, projection_dim)
            attended_sentence = attended_sentence.view(batch_size, sequence_length, -1)

        # Shape: (batch_size, sequence_length, combination_dim)
        combined_tensors = util.combine_tensors(self._combination, [tokens, attended_sentence])
        return self._output_projection(combined_tensors)

    def report_unnormalized_log_attn_weights(self, attn_weights, input_vects):
        assert attn_weights.dim() == 3 and attn_weights.size(1) == attn_weights.size(2), \
            ("Size of attn weights (" + str(attn_weights.size()) + ") indicates multiheaded attention, but " +
             "TalkativeIntraSentenceAttentionEncoder currently assumes single-head attention.")
        vects_of_attn_weights = attn_weights[:, 0, :]
        should_be_same_as_attn_weights = vects_of_attn_weights.unsqueeze(1).expand(*(attn_weights.size()))
        assert (attn_weights - should_be_same_as_attn_weights).nonzero().size(0) == 0, \
            ("Printing attn weights: \n" + str(attn_weights.data.cpu().numpy()) + "\nPrinted attention weights, " +
             "which are not the same for every element of a single item in a sequence. " +
             "TalkativeIntraSentenceAttentionEncoder currently assumes the attention weight for each dimension " +
             "is the same.")
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
                                str(next_available_ind + vects_of_attn_weights.size(0)))
        np.save(input_vects_filename, input_vects.data.cpu().numpy())
        with open(self._attn_weight_filename, 'a') as f:
            for i in range(vects_of_attn_weights.size(0)):
                f.write(str(next_available_ind) + ": ")
                num_nonpadding_pieces_in_row = \
                    binary_search_for_num_non_padded_tokens_in_instance(vects_of_attn_weights, i)
                weights_to_write = [float(vects_of_attn_weights[i, j])
                                    for j in range(num_nonpadding_pieces_in_row)]
                min_val = min(weights_to_write)
                max_val = max(weights_to_write)
                val_to_subtract = ((max_val - min_val) / 2) + min_val
                f.write(str(" ".join([str(w - val_to_subtract) for w in weights_to_write])))
                f.write('\n')
                next_available_ind += 1
        with open(self._next_available_counter_ind_file, 'w') as f:
            f.write(str(next_available_ind))
