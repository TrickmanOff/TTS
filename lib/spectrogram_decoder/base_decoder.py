from abc import abstractmethod

from torch import Tensor


class BaseDecoder:
    def decode_as_wave(self, mel_spec: Tensor) -> Tensor:
        pass

    @abstractmethod
    def decode_as_wav(self, mel_spec: Tensor, wav_filepath):
        pass
