from overrides import overrides
import torch

from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction


#@SimilarityFunction.register("han_paper")
class HanPaperSimilarityFunction(SimilarityFunction):
    def __init__(self,
                 input_dim: int,
                 context_vect_dim: int) -> None:
        super(HanPaperSimilarityFunction, self).__init__()
        self._mlp = torch.nn.Linear(input_dim, context_vect_dim, bias=True)
        self._context_dot_product = torch.nn.Linear(context_vect_dim, 1, bias=False)

    @overrides
    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        # un-expand tensor_1, which is the only one we'll use
        tensor_1 = tensor_1[:, :, 0, :].view(tensor_1.size(0), tensor_1.size(1), tensor_1.size(3)).contiguous()

        # new shape: batch_size x seq_len x embedding_dim
        batch_size = tensor_1.size(0)
        tensor_1 = tensor_1.view(batch_size * tensor_1.size(1), tensor_1.size(2))
        tensor_1 = torch.tanh(self._mlp(tensor_1))
        tensor_1 = self._context_dot_product(tensor_1)
        tensor_1 = tensor_1.view(batch_size, -1)  # batch_size x seq_len
        tensor_1 = tensor_1.unsqueeze(1).expand(batch_size, tensor_1.size(1), tensor_1.size(1))
        return tensor_1
