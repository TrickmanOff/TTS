from abc import abstractmethod
from typing import Dict


class BasePostprocessor:
    @abstractmethod
    def __call__(self, **batch) -> Dict:
        raise NotImplementedError()
