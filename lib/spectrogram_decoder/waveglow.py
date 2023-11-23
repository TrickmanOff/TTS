import shutil
import torch
from pathlib import Path

from speechbrain.utils.data_utils import download_file

from .base_decoder import BaseDecoder


class WaveglowDecoder(BaseDecoder):
    CHECKPOINT_URL = 'https://www.googleapis.com/drive/v3/files/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx?alt=media&key=AIzaSyBAigZjTwo8uh77umBBmOytKc_qjpTfRjI'

    def __init__(self, waveglow_path: Path = Path('waveglow') / 'pretrained_model' / 'waveglow_256channels.pt'):
        super().__init__()
        if not waveglow_path.exists():
            self._load_checkpoint(waveglow_path)

        wave_glow = torch.load(waveglow_path)['model']
        wave_glow = wave_glow.remove_weightnorm(wave_glow)
        wave_glow.cuda().eval()
        for m in wave_glow.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')

        self.wave_glow = wave_glow

    def _load_checkpoint(self, target_filepath: Path):
        loaded_checkpoint_filepath = Path('waveglow_256channels_ljs_v2.pt')
        if not loaded_checkpoint_filepath.exists():
            print(f'Downloading WaveGlow weights...')
            download_file(self.CHECKPOINT_URL, loaded_checkpoint_filepath)
        target_filepath.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(loaded_checkpoint_filepath, target_filepath)
