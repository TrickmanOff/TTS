import torch.nn.functional as F
from torch import Tensor

from lib.loss.base_loss import BaseLoss


class MelLoss(BaseLoss):
    def forward(self, true_mel_spec: Tensor, pred_mel_spec: Tensor, **batch) -> Tensor:
        """
        true_mel_spec: (B, S', freqs)
        pred_mel_spec: (B, S', freqs)

        returns mean loss
        """
        return F.mse_loss(pred_mel_spec, true_mel_spec, reduction='mean')
