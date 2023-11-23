from abc import abstractmethod
from pathlib import Path
from typing import Union

from torch import Tensor


class BaseDecoder:
    @abstractmethod
    def decode_as_wav(self, mel_spec: Tensor, wav_filepath: Union[str, Path]):
        raise NotImplementedError

    @abstractmethod
    def decode_as_wave(self, mel_spec: Tensor) -> Tensor:
        """
        mel_spec: (B, freqs, T)
        """
        raise NotImplementedError()
