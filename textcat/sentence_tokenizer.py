from typing import List

from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from .sentence_splitter import SpacySentenceSplitter


@Tokenizer.register("sentence")
class SentenceTokenizer(Tokenizer):
    """
    A ``WordTokenizer`` handles the splitting of strings into words as well as any desired
    post-processing (e.g., stemming, filtering, etc.).  Note that we leave one particular piece of
    post-processing for later: the decision of whether or not to lowercase the token.  This is for
    two reasons: (1) if you want to make two different casing decisions for whatever reason, you
    won't have to run the tokenizer twice, and more importantly (2) if you want to lowercase words
    for your word embedding, but retain capitalization in a character-level representation, we need
    to retain the capitalization here.

    Parameters
    ----------
    word_splitter : ``WordSplitter``, optional
        The :class:`WordSplitter` to use for splitting text strings into word tokens.  The default
        is to use the ``SpacyWordSplitter`` with default parameters.
    word_filter : ``WordFilter``, optional
        The :class:`WordFilter` to use for, e.g., removing stopwords.  Default is to do no
        filtering.
    word_stemmer : ``WordStemmer``, optional
        The :class:`WordStemmer` to use.  Default is no stemming.
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.
    """
    def __init__(self) -> None:
        self._sentence_splitter = SpacySentenceSplitter()

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        Does whatever processing is required to convert a string of text into a sequence of tokens.

        At a minimum, this uses a ``WordSplitter`` to split words into text.  It may also do
        stemming or stopword removal, depending on the parameters given to the constructor.
        """
        sents = self._sentence_splitter.split_sentences(text)
        return sents

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        batched_sents = self._sentence_splitter.batch_split_sentences(texts)
        return batched_sents
