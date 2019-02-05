from typing import Dict, Optional, List, Any
import torch
import overrides
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn.util import get_text_field_mask, masked_softmax
from math import isclose


@Model.register("han")
class HierarchicalAttentionNetwork(Model):
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
    word_encoder : ``Seq2SeqEncoder``
        Used to encode words.
    sentence_encoder : ``Seq2SeqEncoder``
        Used to encode sentences.
    word_attention : ``Seq2VecEncoder``
        Seq2Vec layer that (in original implementation) uses attention to calculate a fixed-length vector
        representation of each sentence from that sentence's sequence of word vectors
    sentence_attention : ``Seq2VecEncoder``
        Seq2Vec layer that (in original implementation) uses attention to calculate a fixed-length vector
        representation of each document from that document's sequence of sentence vectors
    classification_layer : ``FeedForward``
        This feedforward network computes the output logits.
    pre_word_encoder_dropout : ``float``, optional (default=0.0)
        Dropout percentage to use before word_attention encoder.
    pre_sentence_encoder_dropout : ``float``, optional (default=0.0)
        Dropout percentage to use before sentence_attention encoder.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sentence_encoder: Seq2SeqEncoder,
                 document_encoder: Seq2SeqEncoder,
                 word_attention: Seq2SeqEncoder,
                 sentence_attention: Seq2SeqEncoder,
                 output_logit: FeedForward,
                 pre_sentence_encoder_dropout: float = 0.0,
                 pre_document_encoder_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 calculate_f1: bool = False) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._word_attention = word_attention
        self._sentence_attention = sentence_attention
        self._pre_sentence_encoder_dropout = torch.nn.Dropout(p=pre_sentence_encoder_dropout)
        self._sentence_encoder = sentence_encoder
        self._pre_document_encoder_dropout = torch.nn.Dropout(p=pre_document_encoder_dropout)
        self._document_encoder = document_encoder

        self._output_logit = output_logit

        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._accuracy = CategoricalAccuracy()
        self.calculate_f1 = calculate_f1
        if self.calculate_f1:
            self._f1 = F1Measure(1)
        self._loss = torch.nn.CrossEntropyLoss()  # torch.cuda.FloatTensor([1, 3.65]))

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
        # tokens['tokens'] is a 3-d tensor: all_docs x max_num_sents_in_any_doc x max_num_tokens_in_any_doc_sent
        # these tokens should be sentence-segmented.
        tokens_ = tokens['tokens']
        batch_size = tokens_.size(0)
        max_num_sents = tokens_.size(1)
        first_token_ind_in_each_sentence = tokens_[:, :, 0].view(batch_size * max_num_sents)
        sentence_level_mask = tokens_.new_zeros(batch_size * max_num_sents).float()
        inds_of_nonzero_rows = torch.nonzero(first_token_ind_in_each_sentence)
        inds_of_nonzero_rows = inds_of_nonzero_rows.view(inds_of_nonzero_rows.shape[0])
        sentence_level_mask[inds_of_nonzero_rows] = 1
        sentence_level_mask = sentence_level_mask.view(batch_size, max_num_sents)

        embedded_words = self._text_field_embedder(tokens)
        batch_size, max_sents, _, _ = embedded_words.size()
        embedded_words = embedded_words.view(batch_size * max_num_sents, embedded_words.size(2), -1)
        tokens_ = tokens_.view(batch_size * max_num_sents, -1)

        # we encode each sentence with a seq2seq encoder on its words, then seq2vec encoder incorporating attention
        mask = get_text_field_mask({"tokens": tokens_}).float()
        embedded_words = self._pre_sentence_encoder_dropout(embedded_words)
        encoded_words = self._sentence_encoder(embedded_words, mask)
        sentence_repr = self._word_attention(encoded_words, mask)
        sentence_repr = torch.sum(sentence_repr, 1)
        sentence_repr = sentence_repr.view(batch_size, max_num_sents, -1)

        # we encode each document with a seq2seq encoder on its sentences, then seq2vec encoder incorporating attention
        sentence_repr = self._pre_document_encoder_dropout(sentence_repr)
        encoded_sents = self._document_encoder(sentence_repr, sentence_level_mask)
        document_repr = self._sentence_attention(encoded_sents, sentence_level_mask)
        document_repr = torch.sum(document_repr, 1)

        label_logits = self._output_logit(document_repr.view(batch_size, -1))
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            if self.calculate_f1:
                self._f1(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def reconstruct_attn_weights_for_testing(self, encoded_sentences, sentence_repr):
        assert encoded_sentences.size(2) == sentence_repr.size(2), \
            "reconstruct_attn_weights_for_testing doesn't work if there's a projection"
        weights = encoded_sentences.new_zeros((encoded_sentences.size(0), encoded_sentences.size(1))).cpu()
        for sent_ind in range(encoded_sentences.size(0)):
            for word_ind in range(encoded_sentences.size(1)):
                maybe_multiplier = None
                for other_dim in range(encoded_sentences.size(2)):
                    if maybe_multiplier is not None:
                        assert isclose(float((sentence_repr[sent_ind, word_ind, other_dim] /
                                              encoded_sentences[sent_ind, word_ind, other_dim]).data), maybe_multiplier,
                                       abs_tol=.0001), \
                        ("Hypothesized multiplier is " + str(maybe_multiplier) + " but just found a multiplier of " +
                         str(float((sentence_repr[sent_ind, word_ind, other_dim] /
                                    encoded_sentences[sent_ind, word_ind, other_dim]).data)))
                    elif encoded_sentences[sent_ind, word_ind, other_dim].data != 0:
                        maybe_multiplier = float((sentence_repr[sent_ind, word_ind, other_dim] /
                                                  encoded_sentences[sent_ind, word_ind, other_dim]).data)
                if maybe_multiplier is None:
                    # assume it's zero
                    maybe_multiplier = 0
                weights[sent_ind, word_ind] = maybe_multiplier

            assert .99 < torch.sum(weights[sent_ind, :]) < 1.01, str(torch.sum(weights[sent_ind, :]))
        print(weights[0])
        exit(1)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.calculate_f1:
            f1_pieces = self._f1.get_metric(reset)
            return {'accuracy': self._accuracy.get_metric(reset), 'f1': f1_pieces[2],
                    'precision': f1_pieces[0], 'recall': f1_pieces[1]}
        else:
            return {'accuracy': self._accuracy.get_metric(reset)}


@Seq2VecEncoder.register("han_attention")
class HanAttention(Seq2VecEncoder):
    def __init__(self,
                 input_dim: int = None,
                 context_vector_dim: int = None) -> None:
        super(HanAttention, self).__init__()
        self._mlp = torch.nn.Linear(input_dim, context_vector_dim, bias=True)
        self._context_dot_product = torch.nn.Linear(context_vector_dim, 1, bias=False)
        self.vec_dim = self._mlp.weight.size(1)

    def get_input_dim(self) -> int:
        return self.vec_dim

    def get_output_dim(self) -> int:
        return self.vec_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        assert mask is not None
        batch_size, sequence_length, embedding_dim = tokens.size()

        attn_weights = tokens.view(batch_size * sequence_length, embedding_dim)
        attn_weights = torch.tanh(self._mlp(attn_weights))
        attn_weights = self._context_dot_product(attn_weights)
        attn_weights = attn_weights.view(batch_size, -1)  # batch_size x seq_len
        attn_weights = masked_softmax(attn_weights, mask)
        attn_weights = attn_weights.unsqueeze(2).expand(batch_size, sequence_length, embedding_dim)

        return torch.sum(tokens * attn_weights, 1)
