from abc import abstractmethod
from typing import Dict, List, Union

from torch import Tensor, nn


class BaseLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def get_loss_parts_names(self) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, **batch) -> Union[Tensor, Dict[str, Tensor]]:
        raise NotImplementedError()
