import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor, LongTensor


logger = logging.getLogger(__name__)


PADDING_VALUE = 0


def pad_last_d(input: List[Tensor], padding_value=PADDING_VALUE) -> Tuple[Tensor, LongTensor]:
    """
    each of B inputs is of shape (..., S_i)

    result:
        stack:  (B, ..., max_i S_i)
        length: (B,) - initial lengths of each sequence
    """
    length = LongTensor([x.shape[-1] for x in input])
    max_len = length.max()

    shape = [len(input)] + list(input[0].shape)
    shape[-1] = max_len

    stack = torch.full(shape, padding_value, dtype=input[0].dtype)

    for i, x in enumerate(input):  # (..., S_i)
        stack[i, ..., :x.shape[-1]] = x

    return stack, length


def collate_fn(dataset_items: List[dict]) -> Dict[str, Any]:
    """
    Collate and pad fields in dataset items
    """
    all_items = defaultdict(list)  # {str: [val1, val2, ...], ...}
    for items in dataset_items:
        for key, val in items.items():
            all_items[key].append(val)

    result_batch = {}

    result_batch['id'] = all_items['id']

    # mel
    if 'mel' in all_items:
        result_batch['true_mel_spec'], result_batch['mel_length'] = pad_last_d(all_items['mel'])

    # text
    if 'text' in all_items:
        result_batch['text'] = all_items['text']

    # alignment
    if 'alignment' in all_items:
        result_batch['true_duration'], _ = pad_last_d(all_items['alignment'])

    # pitch
    if 'pitch' in all_items:
        result_batch['true_pitch'], _ = pad_last_d(all_items['pitch'])

    # energy
    if 'energy' in all_items:
        result_batch['true_energy'], _ = pad_last_d(all_items['energy'])

    # wave
    if 'wave' in all_items:
        result_batch['wave'], _ = pad_last_d(all_items['wave'])

    # phonemes tokens
    result_batch['phonemes_tokens'], result_batch['phonemes_sequence_length'] = pad_last_d(all_items['phonemes_tokens'])

    return result_batch
