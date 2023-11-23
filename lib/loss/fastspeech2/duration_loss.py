import torch
import torch.nn.functional as F
from torch import Tensor

from lib.loss.base_loss import BaseLoss


class DurationLoss(BaseLoss):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self._eps = eps

    def forward(self, true_duration: Tensor, pred_log_duration: Tensor, **batch) -> Tensor:
        """
        true_duration:     (B, S)
        pred_log_duration: (B, S)

        returns mean loss
        """
        return F.mse_loss(pred_log_duration, torch.log(true_duration + self._eps), reduction='mean')
