from abc import abstractmethod
from typing import Union

import numpy as np
from torch import Tensor, nn

from lib.dataset.base_dataset import BaseDataset


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, **batch) -> Union[Tensor, dict]:
        """
        :return: Model output
        """
        raise NotImplementedError()

    def calculate_statistics(self, train_dataset: BaseDataset):
        pass

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
