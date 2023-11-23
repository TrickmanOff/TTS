from typing import Dict, Mapping

import torch
from torch import Tensor

from .base_loss import BaseLoss


class CombinedLoss(BaseLoss):
    def __init__(self, losses: Mapping[str, BaseLoss], weights: Mapping[str, float]):
        super().__init__()
        self._losses = losses
        self._weights = weights

    def forward(self, **batch) -> Dict[str, Tensor]:
        res = {}
        total_loss = torch.tensor(0.)
        for part_name, part_loss_fn in self._losses.items():
            part_loss = part_loss_fn(**batch)
            total_loss += self._weights[part_name] * part_loss
            res[part_name] = part_loss
        res['loss'] = total_loss
        return res
