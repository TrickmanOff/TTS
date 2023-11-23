from abc import abstractmethod

from torch import LongTensor


class BaseTextEncoder:
    @abstractmethod
    def encode(self, text: str) -> LongTensor:
        raise NotImplementedError
