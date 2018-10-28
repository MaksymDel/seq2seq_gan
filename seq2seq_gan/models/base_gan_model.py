import logging
from typing import Dict

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model

from seq2seq_gan.losses import *

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class BaseGanModel(Model):
    """
    This ``SimpleSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent generators_discriminators for these tasks.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.

    """

    def __init__(self, vocab: Vocabulary) -> None:
        super(BaseGanModel, self).__init__(vocab)

    @overrides
    def forward(self,  # type: ignore
                real_A: Dict[str, torch.LongTensor] = None,
                real_B: Dict[str, torch.LongTensor] = None,
                answers_for_A: Dict[str, torch.LongTensor] = None,
                answers_for_B: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        We optionally use answers only to better track progress during experimentation (for metrics).
        They are just a metadata and do not help the model to learn in any way.

        Answers for B in in A domain and vice versa.

        Parameters
        ----------
        real_A : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        real_B : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        raise NotImplementedError

    @staticmethod
    def _ids_to_onehot(token_ids: torch.LongTensor, num_classes: int) -> torch.Tensor:
        batch_size = token_ids.size()[0]
        seq_len = token_ids.size()[1]

        onehots = torch.zeros(batch_size, seq_len, num_classes)
        onehots.scatter_(dim=2, index=token_ids.unsqueeze(2), value=1)

        # TODO: work out the correct device

        return onehots

    @staticmethod
    def _set_requires_grad(nets, requires_grad=False):
        """
        Prevents network parameters from accumulating gradients
        but keeps recording computational graph for the backward pass.
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def _detach_batch(batch):
        return {"ids": batch["ids"], "onehots": batch["onehots"].detach()}
