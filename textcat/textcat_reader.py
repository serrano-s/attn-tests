from typing import Dict, List
import logging
import numpy as np
import re
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.common.checks import ConfigurationError
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from .sentence_tokenizer import SentenceTokenizer
from allennlp.data.tokenizers.word_filter import StopwordFilter, PassThroughWordFilter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("textcat")
class TextCatReader(DatasetReader):
    """
    Reads tokens and their topic labels.

    Assumes that data in file_path provided to _read is tab-separated, containing (at least) the two
    fields 'tokens' and 'category', in no particular order, with each document/label on one line.
    (So this means that documents must not contain either newlines or tabs.)

    Example:

    category    tokens
    sample_label_1  This is a document. It contains a couple of sentences.
    sample_label_1  This is another document. It also contains two sentences.
    sample_label_2  This document has a different label.

    and so on.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        tokens: ``TextField`` and
        label: ``LabelField``

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 word_tokenizer: Tokenizer = None,
                 segment_sentences: bool = False,
                 lazy: bool = False,
                 column_titles_to_index: List[str] = ("tokens", )) -> None:
        super().__init__(lazy=lazy)
        self._word_tokenizer = word_tokenizer or WordTokenizer(word_filter=PassThroughWordFilter())
        self._segment_sentences = segment_sentences
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._column_titles_to_index = column_titles_to_index
        assert len(self._column_titles_to_index) > 0
        if self._segment_sentences:
            self._sentence_segmenter = SentenceTokenizer()        
        

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            columns = data_file.readline().strip('\n').split('\t')
            token_col_inds = [columns.index(self._column_titles_to_index[field_ind])
                              for field_ind in range(len(self._column_titles_to_index))]
            for line in data_file.readlines():
                if not line:
                    continue
                items = line.strip("\n").split("\t")
                tokens = ''
                for col_ind in token_col_inds:
                    tokens += items[col_ind] + ' '
                tokens = tokens[:-1]
                tokens = items[columns.index("tokens")]
                if len(tokens.strip()) == 0:
                    continue
                category = items[columns.index("category")]
                instance = self.text_to_instance(tokens=tokens,
                                                 category=category)
                if instance is not None:
                    yield instance
                

    @overrides
    def text_to_instance(self, tokens: List[str], category: str = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        category ``str``, optional, (default = None).
            The category for this sentence.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The category label of the sentence or phrase.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_fields = []
        if self._segment_sentences:
            sentence_tokens = self._sentence_segmenter.tokenize(tokens)
            original_len_sentence_tokens = len(sentence_tokens)
            corresponding_sentence_ind = 0
            for i in range(original_len_sentence_tokens):
                sentence = sentence_tokens[corresponding_sentence_ind]
                word_tokens = self._word_tokenizer.tokenize(sentence)
                if len(word_tokens) == 0:
                    del sentence_tokens[corresponding_sentence_ind]
                    continue
                text_fields.append(TextField(word_tokens, self._token_indexers))
                corresponding_sentence_ind += 1
            if len(text_fields) == 0:
                return None
            fields['tokens'] = ListField(text_fields)
        else:
            fields['tokens'] = TextField(self._word_tokenizer.tokenize(tokens),
                                            self._token_indexers)
        fields['label'] = LabelField(category)
        return Instance(fields)
