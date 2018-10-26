from overrides import overrides
from typing import Dict

import torch
from torch.nn.modules.loss import _Loss


class _GeneratorLoss(_Loss):
    @overrides
    def forward(self, probs_fake_batch_being_real: Dict[str, torch.FloatTensor]) -> float:
        """
        Takes a fake batch of probabilities of examples being real and returns a single loss value to minimize
        """
        raise NotImplementedError


class InvertedProbabilityGeneratorLoss(_GeneratorLoss):
    @overrides
    def forward(self, probs_fake_batch_being_real: Dict[str, torch.FloatTensor]) -> float:
        # we always want to maximize probability of input fake batch being real, so we invert the probs
        examples_losses = 1 - probs_fake_batch_being_real["onehots"]

        if self.reduction == "elementwise_mean":  # default
            loss = torch.mean(examples_losses)
        elif self.reduction == "sum":
            loss = torch.sum(examples_losses)
        else:
            raise NotImplementedError

        return loss


class MinusProbabilityGeneratorLoss(_GeneratorLoss):
    @overrides
    def forward(self, probs_fake_batch_being_real: Dict[str, torch.FloatTensor]) -> float:
        # we always want to maximize probability of input fake batch being real, so we take a probs with minus sign
        examples_losses = -probs_fake_batch_being_real["onehots"]

        if self.reduction == "elementwise_mean":  # default
            loss = torch.mean(examples_losses)
        elif self.reduction == "sum":
            loss = torch.sum(examples_losses)
        else:
            raise NotImplementedError

        return loss
