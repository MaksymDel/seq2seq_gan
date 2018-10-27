from typing import Dict

import torch
from overrides import overrides

from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from torch.nn import Sigmoid
from torch.nn.modules import Linear
from allennlp.data import Vocabulary


class Generator(Model):
    """
    Abstract class for Seq2Seq Generator

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    """

    def __init__(self,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)


    @overrides
    def forward(self, tokens: Dict[str, torch.LongTensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.

        Returns
        -------
        Dict
        """
        raise NotImplementedError

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Source tokens is a dict containing 2 keys: ["ids"] and ["onehots"]
        Return dict is in the same format

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.
        Returns
        -------
        Dict[str, torch.Tensor]
        """
        raise NotImplementedError
