from typing import Any

from allennlp.modules import Embedding
from allennlp.modules.attention.dot_product_attention import DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from utils_data import *
from models import *


def build_modules(vocab, opt) -> Dict[str, Any]:
    modules_dict = {}
    modules_dict = _build_embeddings(vocab, modules_dict, opt)
    modules_dict = _build_models(vocab, modules_dict, opt)
    return modules_dict


def _build_embeddings(vocab: Vocabulary, modules_dict: Dict[str, Any], opt) -> Dict[str, Any]:

    embedding_lang_a_generator = Embedding(num_embeddings=vocab.get_vocab_size('language_A'),
                                           embedding_dim=opt.embedding_dim)
    embedding_lang_b_generator = Embedding(num_embeddings=vocab.get_vocab_size('language_B'),
                                           embedding_dim=opt.embedding_dim)

    embedding_lang_a_discriminator = Embedding(num_embeddings=vocab.get_vocab_size('language_A'),
                                               embedding_dim=opt.embedding_dim)
    embedding_lang_b_discriminator = Embedding(num_embeddings=vocab.get_vocab_size('language_B'),
                                               embedding_dim=opt.embedding_dim)

    modules_dict["embedding_lang_a_generator"] = embedding_lang_a_generator
    modules_dict["embedding_lang_b_generator"] = embedding_lang_b_generator
    modules_dict["embedding_lang_a_discriminator"] = embedding_lang_a_discriminator
    modules_dict["embedding_lang_b_discriminator"] = embedding_lang_b_discriminator

    return modules_dict


def _build_models(vocab: Vocabulary, modules_dict: Dict[str, Any], opt) -> Dict[str, Any]:
    generator_a2b = Rnn2Rnn(vocab=vocab,
                            source_embedding=modules_dict["embedding_lang_a_generator"],
                            target_embedding=modules_dict["embedding_lang_b_generator"],
                            encoder=PytorchSeq2SeqWrapper(
                                torch.nn.LSTM(opt.embedding_dim,
                                              opt.hidden_dim,
                                              batch_first=True,
                                              bidirectional=opt.bidirectional)
                            ),
                            max_decoding_steps=opt.max_decoding_steps,
                            attention=DotProductAttention(),
                            target_namespace="language_B"
                            )

    generator_b2a = Rnn2Rnn(vocab=vocab,
                            source_embedding=modules_dict["embedding_lang_b_generator"],
                            target_embedding=modules_dict["embedding_lang_a_generator"],
                            encoder=PytorchSeq2SeqWrapper(
                                torch.nn.LSTM(opt.embedding_dim,
                                              opt.hidden_dim,
                                              batch_first=True,
                                              bidirectional=opt.bidirectional)
                            ),
                            max_decoding_steps=opt.max_decoding_steps,
                            attention=DotProductAttention(),
                            target_namespace="language_A"
                            )

    discriminator_a = Seq2Prob(vocab=vocab,
                               embedding=modules_dict["embedding_lang_a_discriminator"],
                               encoder=PytorchSeq2VecWrapper(
                                   torch.nn.LSTM(opt.embedding_dim,
                                                 opt.hidden_dim,
                                                 batch_first=True,
                                                 bidirectional=opt.bidirectional))
                               )

    discriminator_b = Seq2Prob(vocab=vocab,
                               embedding=modules_dict["embedding_lang_b_discriminator"],
                               encoder=PytorchSeq2VecWrapper(
                                   torch.nn.LSTM(opt.embedding_dim,
                                                 opt.hidden_dim,
                                                 batch_first=True,
                                                 bidirectional=opt.bidirectional))
                               )

    modules_dict["generator_a2b"] = generator_a2b
    modules_dict["generator_b2a"] = generator_b2a
    modules_dict["discriminator_a"] = discriminator_a
    modules_dict["discriminator_b"] = discriminator_b

    return modules_dict
