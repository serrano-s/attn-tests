from typing import Dict, Optional, List, Any
import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules import MatrixAttention
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.nn.util import weighted_sum, get_text_field_mask, get_final_encoder_states, get_mask_from_sequence_lengths
from math import isclose


def binary_search_for_num_non_padded_tokens_in_instance(full_array, allowed_start_ind, allowed_end_ind_plus1):
    open_for_checking_start = allowed_start_ind
    open_for_checking_endplus1 = allowed_end_ind_plus1
    look_at = (open_for_checking_start + open_for_checking_endplus1) // 2
    first_zero_ind_identified = None
    while first_zero_ind_identified is None:
        if full_array[look_at] != 0:
            open_for_checking_start = look_at + 1
        else:
            if full_array[look_at - 1] != 0:
                first_zero_ind_identified = look_at
            else:
                open_for_checking_endplus1 = look_at
        if open_for_checking_start == open_for_checking_endplus1:
            assert open_for_checking_endplus1 == allowed_end_ind_plus1
            first_zero_ind_identified = allowed_end_ind_plus1
        else:
            look_at = (open_for_checking_start + open_for_checking_endplus1) // 2
    return first_zero_ind_identified - allowed_start_ind


def get_endplus1_inds_in_condensed_representation(first_token_ind_in_each_sentence, max_num_sentences_per_instance):
    # perform binary search on each max_num_sentences_per_instance chunk to find the index of its first 0
    # (i.e., the actual num sentences + 1)
    batch_size = first_token_ind_in_each_sentence.size(0) // max_num_sentences_per_instance
    list_of_lengths = [binary_search_for_num_non_padded_tokens_in_instance(first_token_ind_in_each_sentence,
                                                                           i * max_num_sentences_per_instance,
                                                                           (i + 1) * max_num_sentences_per_instance)
                       for i in range(batch_size)]
    tensor_to_fill = first_token_ind_in_each_sentence.new_ones(batch_size)
    for i in range(batch_size):
        tensor_to_fill[i] = list_of_lengths[i]
    return torch.cumsum(tensor_to_fill, 0), max_num_sentences_per_instance - min(list_of_lengths)


@Model.register("flan")
class FlatAttentionNetwork(Model):
    """
    This ``Model`` implements the Hierarchical Attention Network described in
    https://www.semanticscholar.org/paper/Hierarchical-Attention-Networks-for-Document-Yang-Yang/1967ad3ac8a598adc6929e9e6b9682734f789427
    by Yang et. al, 2016.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    sentence_encoder : ``Seq2SeqEncoder``
        Used to encode sentences.
    document_encoder : ``Seq2SeqEncoder``
        Used to encode documents.
    word_attention : ``Seq2SeqEncoder``
        Used to perform intra-sentence attention (between words)
    sentence_attention: ``Seq2SeqEncoder``
        Used to perform intra-document attention (between sentences)
    output_feedforward : ``FeedForward``
        Used to prepare the encoded text for prediction.
    output_logit : ``FeedForward``
        This feedforward network computes the output logits.
    dropout : ``float``, optional (default=0.5)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 document_encoder: Seq2SeqEncoder,
                 word_attention: Seq2SeqEncoder,
                 output_logit: FeedForward,
                 pre_document_encoder_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._word_attention = word_attention
        self._pre_document_encoder_dropout = torch.nn.Dropout(p=pre_document_encoder_dropout)
        self._document_encoder = document_encoder

        self._output_logit = output_logit

        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField``. These are tokens should be segmented into their respective sentences.
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata to persist

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # tokens['tokens'] is a 2-d tensor: all_docs x max_num_words_in_any_doc
        # these tokens should NOT be sentence-segmented.
        tokens_ = tokens['tokens']
        batch_size = tokens_.size(0)
        embedded_text = self._text_field_embedder(tokens)
        batch_size, max_num_words, _ = embedded_text.size()

        # we encode each sentence with a seq2seq encoder + intra-sentence attention
        output_list = []
        mask = get_text_field_mask({"tokens": tokens_}).float()
        embedded_text = self._pre_document_encoder_dropout(embedded_text)
        encoded_words = self._document_encoder(embedded_text, mask)
        document_repr = self._word_attention(encoded_words, mask)
        document_repr = torch.sum(document_repr, 1)

        # we encode each document with a seq2seq encoder + intra-document attention

        label_logits = self._output_logit(document_repr.view(batch_size, -1))
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
