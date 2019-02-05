from typing import Dict
import torch
from allennlp.nn import util
import pickle
from allennlp.modules import FeedForward
from overrides import overrides


class ClassifierFromAttnAndInputVects(torch.nn.Module):
    def __init__(self,
                 classification_module: FeedForward) -> None:
        super(ClassifierFromAttnAndInputVects, self).__init__()
        self._classification_module = classification_module

    def forward(self,
                input_vects: torch.Tensor,
                intra_sentence_attention: torch.Tensor)-> Dict[str, torch.Tensor]:
        # Shape: (batch_size, sequence_length, projection_dim)
        batch_size = input_vects.size(0)
        sequence_length = input_vects.size(1)
        output_token_representation = input_vects
        attn_weights = intra_sentence_attention

        attn_weights = attn_weights.unsqueeze(2).expand(batch_size, sequence_length,
                                                        output_token_representation.size(2))
        # Shape: (batch_size, sequence_length, [num_heads,] projection_dim [/ num_heads])
        correct = (((attn_weights[:, :, 0].sum(dim=1) > .98) & (attn_weights[:, :, 0].sum(dim=1) < 1.02)) |
                   (attn_weights[:, :, 0].sum(dim=1) == 0))
        assert torch.sum(correct.float()) == batch_size, \
            (str(attn_weights[(((attn_weights[:, :, 0].sum(dim=1) <= .98) |
                                (attn_weights[:, :, 0].sum(dim=1) >= 1.02)) &
                               (attn_weights[:, :, 0].sum(dim=1) != 0))]) + "\n" +
             str(torch.sum(attn_weights, dim=1)[(((attn_weights[:, :, 0].sum(dim=1) <= .98) |
                                                  (attn_weights[:, :, 0].sum(dim=1) >= 1.02)) &
                                                 (attn_weights[:, :, 0].sum(dim=1) != 0))]))
        combined_tensors = output_token_representation * attn_weights

        document_repr = torch.sum(combined_tensors, 1)

        label_logits = self._classification_module(document_repr.view(batch_size, -1))
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        return output_dict


def backward_gradient_reporting_template(grad_input, filename):
    """
    Wrap this in a lambda function providing the filename when registering it as a backwards hook
    :param input:
    :param filename:
    :return:
    """
    tensors_to_cat = [grad_input[j].view(1, -1) for j in range(len(grad_input))]
    with open(filename, 'wb') as f:
        pickle.dump(torch.cat(tensors_to_cat, dim=0).cpu(), f)


class GradientReportingClassifierFromAttnAndInputVects(torch.nn.Module):
    def __init__(self,
                 classification_module: torch.nn.Module,
                 temp_filename: str = None) -> None:
        super(GradientReportingClassifierFromAttnAndInputVects, self).__init__()
        self._classification_module = classification_module
        assert self._classification_module.__class__.__name__ == 'FeedForward', \
            ("GradientReportingClassifierFromAttnAndInputVects currently assumes a feedforward output classifier " +
             "for dropout-zeroing purposes, but given output classifier type was " +
             self._classification_module.__class__.__name__)
        dropout_module_list = self._classification_module._dropout
        # set p in all dropouts to 0
        for i in range(len(dropout_module_list)):
            modified_dropout = dropout_module_list.__getitem__(i)
            modified_dropout.p = 0.0
            dropout_module_list.__setitem__(i, modified_dropout)
        self._temp_filename = temp_filename

    def set_temp_filename(self, fname):
        self._temp_filename = fname

    def forward(self,
                input_vects: torch.Tensor,
                intra_sentence_attention: torch.Tensor)-> Dict[str, torch.Tensor]:
        # Shape: (batch_size, sequence_length, projection_dim)
        batch_size = input_vects.size(0)
        sequence_length = input_vects.size(1)
        output_token_representation = input_vects
        attn_weights = intra_sentence_attention

        attn_weights.register_hook(lambda grad: backward_gradient_reporting_template(grad, self._temp_filename))

        attn_weights = attn_weights.unsqueeze(2).expand(batch_size, sequence_length,
                                                        output_token_representation.size(2))
        # Shape: (batch_size, sequence_length, [num_heads,] projection_dim [/ num_heads])

        correct = (((attn_weights[:, :, 0].sum(dim=1) > .98) & (attn_weights[:, :, 0].sum(dim=1) < 1.02)) |
                   (attn_weights[:, :, 0].sum(dim=1) == 0))
        assert torch.sum(correct.float()) == batch_size, \
            (str(attn_weights[(((attn_weights[:, :, 0].sum(dim=1) <= .98) |
                                (attn_weights[:, :, 0].sum(dim=1) >= 1.02)) &
                               (attn_weights[:, :, 0].sum(dim=1) != 0))]) + "\n" +
             str(torch.sum(attn_weights, dim=1)[(((attn_weights[:, :, 0].sum(dim=1) <= .98) |
                                                  (attn_weights[:, :, 0].sum(dim=1) >= 1.02)) &
                                                 (attn_weights[:, :, 0].sum(dim=1) != 0))]))

        combined_tensors = output_token_representation * attn_weights

        document_repr = torch.sum(combined_tensors, 1)

        label_logits = self._classification_module(document_repr.view(batch_size, -1))
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        return output_dict
