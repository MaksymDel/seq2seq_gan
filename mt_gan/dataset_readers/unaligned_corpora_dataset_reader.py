from typing import Dict
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("unaligned_corpora")
class UnilignedCorporaDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``SimpleSeq2Seq`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 tokenizer_A: Tokenizer = None,
                 tokenizer_B: Tokenizer = None,
                 token_indexers_A: Dict[str, TokenIndexer] = None,
                 token_indexers_B: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)

        self._tokenizer_A = tokenizer_A or WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._tokenizer_B = tokenizer_B or self._tokenizer_A

        self._source_token_indexers = token_indexers_A or {"ids": SingleIdTokenIndexer(namespace="vocab_A")}
        self._target_token_indexers = token_indexers_B or {"ids": SingleIdTokenIndexer(namespace="vocab_B")}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 2:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                source_sequence, target_sequence = line_parts
                yield self.text_to_instance(source_sequence, target_sequence)

    @overrides
    def text_to_instance(self, string_A: str = None, string_B: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        if string_A is None and string_B is None:
            raise ValueError("You should provide a batch at least in one domain")

        if string_A is not None and string_B is not None:
            field_A = self.string_to_field(string_A, self._source_token_indexers)
            field_B = self.string_to_field(string_B, self._target_token_indexers)
            return Instance({"batch_A": field_A, "batch_B": field_B})
        elif string_A is not None:
            field_A = self.string_to_field(string_A, self._source_token_indexers)
            return Instance({'batch_A': field_A})
        elif string_B is not None:
            field_B = self.string_to_field(string_B, self._target_token_indexers)
            return Instance({'batch_B': field_B})

    def string_to_field(self, string, token_indexers):
        tokenized_string = self._tokenizer_A.tokenize(string)
        tokenized_string.insert(0, Token(START_SYMBOL))
        tokenized_string.append(Token(END_SYMBOL))
        field = TextField(tokenized_string, token_indexers)
        return field
