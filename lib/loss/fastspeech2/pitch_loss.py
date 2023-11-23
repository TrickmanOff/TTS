import torch.nn.functional as F
from torch import Tensor

from lib.loss.base_loss import BaseLoss


class PitchLoss(BaseLoss):
    def forward(self, true_pitch: Tensor, pred_pitch: Tensor, **batch) -> Tensor:
        """
        true_pitch: (B, S')
        pred_pitch: (B, S')

        returns mean loss
        """
        return F.mse_loss(pred_pitch, true_pitch, reduction='mean')
