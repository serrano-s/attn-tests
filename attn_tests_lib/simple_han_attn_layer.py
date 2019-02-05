from overrides import overrides
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders.intra_sentence_attention import IntraSentenceAttentionEncoder
from allennlp.nn import util
import torch
import os
import numpy as np


@Seq2SeqEncoder.register("simple_han_attention")
class SimpleHanAttention(Seq2SeqEncoder):
    def __init__(self,
                 input_dim : int = None,
                 context_vector_dim: int = None) -> None:
        super(SimpleHanAttention, self).__init__()
        self._mlp = torch.nn.Linear(input_dim, context_vector_dim, bias=True)
        self._context_dot_product = torch.nn.Linear(context_vector_dim, 1, bias=False)
        self.vec_dim = self._mlp.weight.size(1)

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
        attn_weights = util.masked_softmax(attn_weights, mask)
        attn_weights = attn_weights.unsqueeze(2).expand(batch_size, sequence_length, embedding_dim)

        return tokens * attn_weights
