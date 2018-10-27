import warnings
from typing import Dict

import torch
from allennlp.nn.util import get_text_field_mask
from overrides import overrides
from torch.nn.modules.loss import _Loss


class _ReconstructionLoss(_Loss):
    @overrides
    def forward(self,
                original: Dict[str, torch.FloatTensor],
                reconstructed: Dict[str, torch.FloatTensor]) -> float:
        """
        Takes an original batch of examples and reconstructed batch, and returns a single loss value to minimize
        """

        raise NotImplementedError


class CrossEntropyReconstructionLoss(_ReconstructionLoss):

    def forward(self,
                reconstructed: Dict[str, torch.FloatTensor],
                original: Dict[str, torch.FloatTensor]) -> float:
        """
        Compute loss.

        Takes logits (unnormalized output from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

           The complete sequence would correspond to w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  w1  w2  w3  <E> <P> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_mask = get_text_field_mask(original)

        # shape: (batch_size, num_decoding_steps)
        relevant_targets = original["ids"]

        return self._negative_log_likelihood_with_probs(probs=reconstructed["onehots"],
                                                        targets=relevant_targets,
                                                        weights=relevant_mask)

    @staticmethod
    def _negative_log_likelihood_with_probs(probs: torch.FloatTensor,
                                            targets: torch.LongTensor,
                                            weights: torch.FloatTensor,
                                            batch_average: bool = None,
                                            average: str = "batch",
                                            label_smoothing: float = None) -> torch.FloatTensor:
        """
        Computes the cross entropy loss of a sequence, weighted with respect to
        some user provided weights. Note that the weighting here is not the same as
        in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
        classes; here we are weighting the loss contribution from particular elements
        in the sequence. This allows loss computations for models which use padding.

        Parameters
        ----------
        probs : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
            which contains the normalized probability for each class.
        targets : ``torch.LongTensor``, required.
            A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
            index of the true class for each corresponding step.
        weights : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of size (batch, sequence_length)
        batch_average : bool, optional, (default = None).
            A bool indicating whether the loss should be averaged across the batch,
            or returned as a vector of losses per batch element.

            .. deprecated:: 0.6.2
               ``batch_average`` was deprecated and replaced with
               the more general ``average`` in version 0.6.2. It will be removed
               in version 0.8.

        average: str, optional (default = "batch")
            If "batch", average the loss across the batches. If "token", average
            the loss across each item in the input. If ``None``, return a vector
            of losses per batch element.
        label_smoothing : ``float``, optional (default = None)
            Whether or not to apply label smoothing to the cross-entropy loss.
            For example, with a label smoothing value of 0.2, a 4 class classifcation
            target would look like ``[0.05, 0.05, 0.85, 0.05]`` if the 3rd class was
            the correct label.

        Returns
        -------
        A torch.FloatTensor representing the cross entropy loss.
        If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
        If ``average is None``, the returned loss is a vector of shape (batch_size,).

        """
        if batch_average is not None:
            # Maintain old behavior
            if batch_average:
                warnings.warn("batch_average=True was deprecated and replaced "
                              "with average='batch' in version 0.6.2. It will be "
                              "removed in version 0.8.", DeprecationWarning)
                average = "batch"
            else:
                warnings.warn("batch_average=False was deprecated and replaced "
                              "with average=None in version 0.6.2. It will be "
                              "removed in version 0.8.", DeprecationWarning)
                average = None
        if average not in {None, "token", "batch"}:
            raise ValueError("Got average f{average}, expected one of "
                             "None, 'token', or 'batch'")

        # shape : (batch * sequence_length, num_classes)
        probs_flat = probs.view(-1, probs.size(-1))
        # shape : (batch * sequence_length, num_classes)
        log_probs_flat = torch.log(probs_flat)
        # shape : (batch * max_len, 1)
        targets_flat = targets.view(-1, 1).long()

        if label_smoothing is not None and label_smoothing > 0.0:
            num_classes = probs.size(-1)
            smoothing_value = label_smoothing / num_classes
            # Fill all the correct indices with 1 - smoothing value.
            one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
            smoothed_targets = one_hot_targets + smoothing_value
            negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
            negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
        else:
            # Contribution to the negative log likelihood only comes from the exact indices
            # of the targets, as the target distributions are one-hot. Here we use torch.gather
            # to extract the indices of the num_classes dimension which contribute to the loss.
            # shape : (batch * sequence_length, 1)
            negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood * weights.float()

        if average == "batch":
            # shape : (batch_size,)
            per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
            num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
            return per_batch_loss.sum() / num_non_empty_sequences
        elif average == "token":
            return negative_log_likelihood.sum() / (weights.sum().float() + 1e-13)
        else:
            # shape : (batch_size,)
            per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
            return per_batch_loss
