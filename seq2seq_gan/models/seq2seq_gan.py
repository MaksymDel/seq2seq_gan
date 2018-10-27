import logging
from typing import Dict, Union, Tuple, Optional

import numpy
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.training.metrics import Average

from seq2seq_gan.modules import BasicDiscriminator, RnnDecoderGenerator
from seq2seq_gan.metrics import PerElementAccuracy
from seq2seq_gan.losses import *

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("mt_gan")
class MtGan(Model):
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
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : ``int``
        Maximum length of decoded sequences.
    target_namespace : ``str``, optional (default = 'target_tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : ``int``, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention : ``Attention``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    attention_function: ``SimilarityFunction``, optional (default = None)
        This is if you want to use the legacy implementation of attention. This will be deprecated
        since it consumes more memory than the specialized attention modules.
    beam_size : ``int``, optional (default = None)
        Width of the beam for beam search. If not specified, greedy decoding is used.
    scheduled_sampling_ratio : ``float``, optional (default = 0.)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        `Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015 <https://arxiv.org/abs/1506.03099>`_.
    """

    # TODO: make base classes for G and D and specify them as types there
    def __init__(self,
                 vocab: Vocabulary,
                 generator_A_to_B: RnnDecoderGenerator,
                 generator_B_to_A: RnnDecoderGenerator,
                 discriminator_A: BasicDiscriminator,
                 discriminator_B: BasicDiscriminator,
                 vocab_namespace_A: str = "vocab_B",
                 vocab_namespace_B: str = "vocab_A",
                 cycle_loss_weight: float = 1.0,
                 g_metrics_to_print: str = "loss") -> None:
        super(MtGan, self).__init__(vocab)

        # define players of min-max game
        self._generator_A_to_B = generator_A_to_B
        self._generator_B_to_A = generator_B_to_A
        self._discriminator_A = discriminator_A
        self._discriminator_B = discriminator_B

        # define vocabulary
        self._vocab = vocab
        self._num_classes_A = vocab.get_vocab_size(namespace=vocab_namespace_A)
        self._num_classes_B = vocab.get_vocab_size(namespace=vocab_namespace_B)

        # define loss calculators
        self._cycle_loss_weight = cycle_loss_weight
        self._loss_calculator_cycle = CrossEntropyReconstructionLoss()
        self._loss_calculator_generator = InvertedProbabilityGeneratorLoss()
        self._loss_calculator_discriminator = ClassicDiscriminatorLoss()

        # define metrics to print
        self._g_metrics_to_print = g_metrics_to_print.split("-")

        if "accuracy" in self._g_metrics_to_print:
            self.metric_accuracy_cycle_ABA = PerElementAccuracy()
            self.metric_accuracy_cycle_BAB = PerElementAccuracy()
            self.metric_accuracy_g_A_to_B = PerElementAccuracy()
            self.metric_accuracy_g_B_to_A = PerElementAccuracy()

        if "loss" in self._g_metrics_to_print:
            self.metric_loss_cycle_ABA = Average()
            self.metric_loss_cycle_BAB = Average()
            self.metric_loss_g_A_to_B = Average()
            self.metric_loss_g_B_to_A = Average()

        self.metric_loss_d_A_real = Average()
        self.metric_loss_d_A_fake = Average()
        self.metric_loss_d_B_real = Average()
        self.metric_loss_d_B_fake = Average()

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
        if real_A is None or real_B is None:  # Test time!
            raise ValueError("Testing is not implemented yet")

        # get a couple of real batches of un-parallel data
        real_A["onehots"] = self._ids_to_onehot(real_A["ids"], self._num_classes_A)
        real_B["onehots"] = self._ids_to_onehot(real_B["ids"], self._num_classes_B)

        # LOSSES FOR GENERATORS: freeze discriminators first!
        self._set_requires_grad([self._discriminator_A, self._discriminator_B], False)

        # Loss generator A -> B
        loss_g_A_to_B, fake_B = self._forward_generator(real_A, self._generator_A_to_B, self._discriminator_B)
        # Loss generator B -> A
        loss_g_B_to_A, fake_A = self._forward_generator(real_B, self._generator_B_to_A, self._discriminator_A)
        # Cycle loss A -> B -> A
        loss_cycle_ABA, reconstructed_A = self._forward_cycle(real_A, fake_B, self._generator_B_to_A)
        # Cycle loss B -> A -> B
        loss_cycle_BAB, reconstructed_B = self._forward_cycle(real_B, fake_A, self._generator_A_to_B)

        # LOSSES FOR DISCRIMINATORS: unfreeze them now!
        self._set_requires_grad([self._discriminator_A, self._discriminator_B], True)

        # Fake and real losses for Discriminator A
        loss_d_A_real, loss_d_A_fake = self._forward_discriminator(real_A, fake_A, self._discriminator_A)
        # Fake and real losses for Discriminator B
        loss_d_B_real, loss_d_B_fake = self._forward_discriminator(real_B, fake_B, self._discriminator_B)

        # Compute total loss to return and minimize
        loss_cycle_ABA = self._cycle_loss_weight * loss_cycle_ABA
        loss_cycle_BAB = self._cycle_loss_weight * loss_cycle_BAB

        total_loss = loss_g_A_to_B + loss_g_B_to_A + loss_cycle_ABA + loss_cycle_BAB + \
                     loss_d_A_real + loss_d_A_fake + loss_d_B_real + loss_d_B_fake

        # -------------------------------------------------------------------------------------------------------------

        # Compute metrics
        if "accuracy" in self._g_metrics_to_print:
            self.metric_accuracy_g_A_to_B(fake_B['ids'], answers_for_A["ids"], get_text_field_mask(answers_for_A))
            self.metric_accuracy_g_B_to_A(fake_A['ids'], answers_for_B["ids"], get_text_field_mask(answers_for_B))
            self.metric_accuracy_cycle_ABA(reconstructed_A["ids"], real_A['ids'], get_text_field_mask(real_A))
            self.metric_accuracy_cycle_BAB(reconstructed_B["ids"], real_B['ids'], get_text_field_mask(real_B))

        if "loss" in self._g_metrics_to_print:
            self.metric_loss_cycle_ABA(loss_cycle_ABA.item())
            self.metric_loss_cycle_BAB(loss_cycle_BAB.item())
            self.metric_loss_g_A_to_B(loss_g_A_to_B)
            self.metric_loss_g_B_to_A(loss_g_B_to_A)

        self.metric_loss_d_A_real(loss_d_A_real.item())
        self.metric_loss_d_A_fake(loss_d_A_fake.item())
        self.metric_loss_d_B_real(loss_d_B_real.item())
        self.metric_loss_d_B_fake(loss_d_B_fake.item())

        return {"loss": total_loss}

    def _forward_cycle(self, real, fake, generator_fake_to_real):
        reconstructed = generator_fake_to_real(source_tokens=fake, target_tokens=real)
        loss_cycle = self._loss_calculator_cycle(reconstructed=reconstructed,
                                                 original=real)
        return loss_cycle, reconstructed

    def _forward_generator(self, real, generator_real_to_fake, discriminator_target):
        fake = generator_real_to_fake(real)
        probs_fake = discriminator_target(tokens=fake)
        loss_generator = self._loss_calculator_generator(probs_fake_being_real=probs_fake)

        return loss_generator, fake

    def _forward_discriminator(self, real, fake, discriminator):
        probs_real = discriminator(tokens=real)
        loss_real = self._loss_calculator_discriminator(probs_being_real=probs_real,
                                                        is_real=True)

        # detach from generator
        fake_detached = self._detach_batch(fake)  # TODO: implement experience reply
        probs_fake = discriminator(tokens=fake_detached)
        loss_fake = self._loss_calculator_discriminator(probs_being_real=probs_fake,
                                                        is_real=False)

        return loss_real, loss_fake

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_dict = {}

        if "loss" in self._g_metrics_to_print:
            metrics_dict.update({
                "l_cycle_ABA": self.metric_loss_cycle_ABA.get_metric(reset),
                "l_cycle_BAB": self.metric_loss_cycle_BAB.get_metric(reset),
                "l_g_A2B": self.metric_loss_g_A_to_B.get_metric(reset),
                "l_g_B2A": self.metric_loss_g_B_to_A.get_metric(reset)})

        metrics_dict.update({
            "l_d_A_real": self.metric_loss_d_A_real.get_metric(reset),
            "l_d_A_fake": self.metric_loss_d_A_fake.get_metric(reset),
            "l_d_B_real": self.metric_loss_d_B_real.get_metric(reset),
            "l_d_B_fake": self.metric_loss_d_B_fake.get_metric(reset)})

        if "accuracy" in self._g_metrics_to_print:
            metrics_dict.update({
                "acc_cycle_ABA": self.metric_accuracy_cycle_ABA.get_metric(reset),
                "acc_cycle_BAB": self.metric_accuracy_cycle_BAB.get_metric(reset),
                "acc_g_A2B": self.metric_accuracy_g_A_to_B.get_metric(reset),
                "acc_g_B2A": self.metric_accuracy_g_B_to_A.get_metric(reset)})
        return metrics_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

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

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'MtGan':  # type: ignore
        # pylint: disable=arguments-differ
        cycle_loss_weight = params.pop("cycle_loss_weight", 1.0)
        g_metrics_to_print = params.pop("g_metrics_to_print", "loss")

        vocab_namespace_A = params.pop("vocab_namespace_A", "vocab_A")
        vocab_namespace_B = params.pop("vocab_namespace_B", "vocab_B")

        num_classes_A = vocab.get_vocab_size(namespace=vocab_namespace_A)
        num_classes_B = vocab.get_vocab_size(namespace=vocab_namespace_B)

        params_generators = params.pop("generators")
        if params_generators.pop("type") == "rnn2rnn":
            generators_embedding_dim = params_generators.pop("embedding_dim")
            embedding_A_generator = Embedding(num_embeddings=num_classes_A, embedding_dim=generators_embedding_dim)
            embedding_B_generator = Embedding(num_embeddings=num_classes_B, embedding_dim=generators_embedding_dim)

            params_encoder_generators = params_generators.pop("encoder")
            generator_A_to_B_encoder = Seq2SeqEncoder.from_params(params_encoder_generators.duplicate())
            generator_B_to_A_encoder = Seq2SeqEncoder.from_params(params_encoder_generators.duplicate())

            generator_attention_params = params_generators.pop("attention")
            attention_generator_A_to_B = Attention.from_params(generator_attention_params.duplicate())
            attention_generator_B_to_A = Attention.from_params(generator_attention_params.duplicate())

            generators_max_decoding_steps = params_generators.pop("max_decoding_steps")

            generator_A_to_B = RnnDecoderGenerator(vocab=vocab,
                                                   source_embedding=embedding_A_generator,
                                                   target_embedding=embedding_B_generator,
                                                   encoder=generator_A_to_B_encoder,
                                                   max_decoding_steps=generators_max_decoding_steps,
                                                   source_namespace=vocab_namespace_A,
                                                   target_namespace=vocab_namespace_B,
                                                   attention=attention_generator_A_to_B)

            generator_B_to_A = RnnDecoderGenerator(vocab=vocab,
                                                   source_embedding=embedding_B_generator,
                                                   target_embedding=embedding_A_generator,
                                                   encoder=generator_B_to_A_encoder,
                                                   max_decoding_steps=generators_max_decoding_steps,
                                                   source_namespace=vocab_namespace_B,
                                                   target_namespace=vocab_namespace_A,
                                                   attention=attention_generator_B_to_A)
        else:
            raise ConfigurationError(message="This generators model type is not supported")

        discriminators_params = params.pop("discriminators")
        if discriminators_params.pop("type") == "seq2prob":
            params_encoder_discriminators = discriminators_params.pop("encoder")
            discriminator_A_encoder = Seq2VecEncoder.from_params(params_encoder_discriminators.duplicate())
            discriminator_B_encoder = Seq2VecEncoder.from_params(params_encoder_discriminators.duplicate())

            discriminators_embedding_dim = discriminators_params.pop("embedding_dim")
            embedding_A_discriminator = Embedding(num_classes_A, discriminators_embedding_dim)
            embedding_B_discriminator = Embedding(num_classes_B, discriminators_embedding_dim)

            discriminator_A = BasicDiscriminator(vocab=vocab, encoder=discriminator_A_encoder,
                                                 embedding=embedding_A_discriminator)
            discriminator_B = BasicDiscriminator(vocab=vocab, encoder=discriminator_B_encoder,
                                                 embedding=embedding_B_discriminator)
        else:
            raise ConfigurationError(message="This discriminators model type is not supported")

        return cls(vocab=vocab,
                   generator_A_to_B=generator_A_to_B,
                   generator_B_to_A=generator_B_to_A,
                   discriminator_A=discriminator_A,
                   discriminator_B=discriminator_B,
                   vocab_namespace_A=vocab_namespace_A,
                   vocab_namespace_B=vocab_namespace_B,
                   cycle_loss_weight=cycle_loss_weight,
                   g_metrics_to_print=g_metrics_to_print)



