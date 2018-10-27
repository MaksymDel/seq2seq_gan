from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from overrides import overrides
from torch.nn import Sigmoid
from torch.nn.modules import Linear

from seq2seq_gan.modules.discriminators.discriminator import Discriminator


# Model.register("basic")
class BasicDiscriminator(Discriminator):
    """
    Predicts the probability of batch being real

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    embedding : ``Embedding``, required
        Embedding for source side sequences
    encoder : ``Seq2VecEncoder``, required
        The encoder of the model
    """

    def __init__(self,
                 vocab: Vocabulary,
                 encoder: Seq2VecEncoder,
                 embedding: Embedding = None) -> None:
        super().__init__(vocab)
        # Dense embedding of source vocab tokens.
        self._embedding = embedding

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = encoder

        self._sigmoid = Sigmoid()

        self._projection_layer = Linear(encoder.get_output_dim(), 1)

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
        torch.Tensor
        """
        # shape: (batch_size, max_input_sequence_length, embedding_dim)
        embedded_input = torch.matmul(tokens["onehots"].float(), self._embedding.weight)

        batch_size, _, _ = embedded_input.size()

        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(tokens)

        # shape: (batch_size, encoder_output_dim)
        encoder_output = self._encoder(embedded_input, source_mask)

        # shape: (batch_size, 1)
        energies = self._projection_layer(encoder_output)

        return self._sigmoid(energies).squeeze(1)
