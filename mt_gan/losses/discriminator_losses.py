from torch.nn.modules.loss import _Loss
import torch
from overrides import overrides
from typing import Dict


class _DiscriminatorLoss(_Loss):

    @overrides
    def forward(self, probs_batch_being_real: Dict[str, torch.FloatTensor], batch_is_real: bool) -> float:
        """
        Takes a fake or real probabilities batch of examples being real and returns a single loss value to minimize/
        """
        raise NotImplementedError


class ClassicDiscriminatorLoss(_DiscriminatorLoss):

    @overrides
    def forward(self, probs_batch_being_real: Dict[str, torch.FloatTensor], batch_is_real: bool) -> float:
        """
        Takes a fake or real probabilities batch of examples being real and returns a single loss value to minimize.
        """
        if batch_is_real:  # we want to maximize probs of real batch being real, so need to invert probs
            examples_losses = 1 - probs_batch_being_real
        else:
            examples_losses = probs_batch_being_real  # we want to minimize probs of fake batch being real

        if self.reduction == "elementwise_mean":  # default
            loss = torch.mean(examples_losses)
        elif self.reduction == "sum":
            loss = torch.sum(examples_losses)
        else:
            raise NotImplementedError

        return loss





