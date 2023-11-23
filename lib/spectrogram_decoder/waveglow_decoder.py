import shutil
import torch
from pathlib import Path
from typing import Union

from speechbrain.utils.data_utils import download_file
from torch import Tensor

from .base_decoder import BaseDecoder
from .waveglow import inference as waveglow_inference


class WaveglowDecoder(BaseDecoder):
    CHECKPOINT_URL = 'https://www.googleapis.com/drive/v3/files/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx?alt=media&key=AIzaSyBAigZjTwo8uh77umBBmOytKc_qjpTfRjI'

    def __init__(self, waveglow_path: Path = Path('waveglow') / 'pretrained_model' / 'waveglow_256channels.pt',
                 device=torch.device('cuda'), sampling_rate: int = 22050):
        super().__init__()
        if not waveglow_path.exists():
            self._load_checkpoint(waveglow_path)

        wave_glow = torch.load(waveglow_path, map_location=device)['model']
        wave_glow = wave_glow.remove_weightnorm(wave_glow)
        wave_glow.to(device).eval()
        for m in wave_glow.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')

        self.wave_glow = wave_glow
        self.sampling_rate = sampling_rate
        print('Waveglow weights loaded')

    def decode_as_wave(self, mel_spec: Tensor) -> Tensor:
        return waveglow_inference.get_wav(mel_spec, self.wave_glow,
                                          sampling_rate=self.sampling_rate)

    def decode_as_wav(self, mel_spec: Tensor, wav_filepath: Union[str, Path]):
        """
        mel_spec : (B, freqs, t)
        """
        waveglow_inference.inference(mel_spec, self.wave_glow, wav_filepath,
                                     sampling_rate=self.sampling_rate)

    def _load_checkpoint(self, target_filepath: Path):
        target_filepath.parent.mkdir(parents=True, exist_ok=True)
        loaded_checkpoint_filepath = target_filepath.parent / 'waveglow_256channels_ljs_v2.pt'
        if not loaded_checkpoint_filepath.exists():
            print(f'Downloading WaveGlow weights...')
            download_file(self.CHECKPOINT_URL, loaded_checkpoint_filepath)
        shutil.move(loaded_checkpoint_filepath, target_filepath)
