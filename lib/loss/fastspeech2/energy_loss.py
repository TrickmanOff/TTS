import torch.nn.functional as F
from torch import Tensor

from lib.loss.base_loss import BaseLoss


class EnergyLoss(BaseLoss):
    def forward(self, true_energy: Tensor, pred_energy: Tensor, **batch) -> Tensor:
        """
        true_energy: (B, S')
        pred_energy: (B, S')

        returns mean loss
        """
        return F.mse_loss(pred_energy, true_energy, reduction='mean')
