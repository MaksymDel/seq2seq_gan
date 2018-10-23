import typing
import torch
from typing import List, Dict, Generator
import numpy
from allennlp.data.vocabulary import Vocabulary


def sample_new_batch(generator: Generator, vocab: Vocabulary) -> Dict[str, Dict[str, torch.Tensor]]:
    vocab_size_src = vocab.get_vocab_size("language_A")
    vocab_size_tgt = vocab.get_vocab_size("language_B")

    model_input = generator.__next__()

    source_batch_onehots = ids2onehots(model_input['source_tokens']['ids'],
                                       vocab_size_src)
    target_batch_onehots = ids2onehots(model_input['target_tokens']['ids'],
                                       vocab_size_tgt)

    model_input['source_tokens']['onehots'] = source_batch_onehots
    model_input['target_tokens']['onehots'] = target_batch_onehots

    return model_input


def ids2onehots(token_ids: torch.LongTensor, num_classes: int) -> torch.Tensor:
    onehots = numpy.zeros((token_ids.size()[0], token_ids.size()[1], num_classes))
    n_rows, n_cols = token_ids.size()
    for i in range(n_rows):
        for j in range(n_cols):
            onehots[i, j, token_ids[i, j]] = 1
    onehots = torch.from_numpy(onehots).long()
    return onehots


def prepare_model_input(source_tokens_dict: Dict[str, torch.Tensor],
                        target_tokens_dict: Dict[str, torch.Tensor] = None) -> Dict[str, Dict[str, torch.Tensor]]:
    if 'loss' in source_tokens_dict.keys():
        source_tokens_dict.pop('loss')
    model_input = {'source_tokens': source_tokens_dict}
    if target_tokens_dict:
        if 'loss' in target_tokens_dict.keys():
            target_tokens_dict.pop('loss')
        model_input['target_tokens'] = target_tokens_dict

    # TODO: move to GPU
    return model_input
