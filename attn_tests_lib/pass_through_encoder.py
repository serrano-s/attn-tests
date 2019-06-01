from overrides import overrides
import torch
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

@Seq2SeqEncoder.register("pass_through_encoder")
class PassThroughSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(self,
                 input_size: int,
                 hidden_size: int) -> None:
        super(PassThroughSeq2SeqEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,  # not packed
                mask: torch.Tensor,
                hidden_state: torch.Tensor = None) -> torch.Tensor:
        # assume batch is first
        return inputs

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def get_input_dim(self) -> int:
        return self.input_size

    @overrides
    def get_output_dim(self) -> int:
        return self.hidden_size