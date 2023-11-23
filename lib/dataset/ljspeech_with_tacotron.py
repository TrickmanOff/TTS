import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pyworld as pw
import torch
import torchaudio
from speechbrain.utils.data_utils import download_file
from torch import Tensor
from tqdm import tqdm

from .base_dataset import BaseDataset
from lib.text_encoder.base_encoder import BaseTextEncoder


def generate_pitch(pitch_filepath: Path, wave: Tensor, mel: Tensor, sample_rate: int, hop_len: int):
    wave = wave.squeeze(0).numpy().astype(np.double)
    pitch, t = pw.dio(wave,
                      fs=sample_rate,
                      frame_period=hop_len / sample_rate * 1000)
    # each pitch value corresponds to the beggining of an fft window
    pitch = pw.stonemask(wave, pitch, t, sample_rate)  # (T,)
    pitch = pitch[:mel.shape[1]]

    np.save(str(pitch_filepath), pitch)


class LJSpeechWithTacotronDataset(BaseDataset):
    """
    All links from the seminar notebook are conveniently hidden here.
    """
    URL_LINKS = {
        'audio': 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
        'text': 'https://www.googleapis.com/drive/v3/files/1yDCbe3GpRXVjgUm3Kd_RUIh3AAKCNf_p?alt=media&key=AIzaSyBAigZjTwo8uh77umBBmOytKc_qjpTfRjI',
        'mel': 'https://www.googleapis.com/drive/v3/files/1XLD0VKO9AEIiIYzUlQPweYUoPVdm2z28?alt=media&key=AIzaSyBAigZjTwo8uh77umBBmOytKc_qjpTfRjI',
        'alignment': 'https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip',
    }

    SAMPLE_RATE = 22050
    NUM_ENTRIES = 13100
    STFT_WIN_LEN = 1024
    STFT_HOP_LEN = 256

    def __init__(self, data_dir: Union[str, Path], return_wave: bool = False,
                 encoder: Optional[BaseTextEncoder] = None, **kwargs):
        """
        data_dir is expected to have the following structure:
        data_dir
        |-- id1
        |  |-- audio.wav
        |  |-- mel.npy
        |  |-- text.txt
        |  |-- alignment.npy
        |  |-- pitch.npy
        |  |-- energy.npy
        |-- id2
        ...

        encoder MUST BE PhonemeEncoder

        each element of an entry is optional, if it is not present for one entry, than
        such element will be redownloaded for all entries
        """
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._return_wave = return_wave
        self._check_entries()
        self._encoder = encoder

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        id = self.get_id_by_idx(idx)
        entry_dirpath = self.get_entry_dirpath(id)
        res = {
            'mel': self.get_mel(idx),  # (freq, T)
            'text': open(entry_dirpath / 'text.txt').readline(),
            'alignment': torch.LongTensor(np.load(str(entry_dirpath / 'alignment.npy'))),  # (P,)
            'pitch': Tensor(np.load(str(entry_dirpath / 'pitch.npy'))),  # (T,)
            'energy': Tensor(np.load(str(entry_dirpath / 'energy.npy'))),  # (T,)
        }
        if self._return_wave:
            res['wave'] = self.get_wave(idx)  # (1, wave_len)
        if self._encoder is not None:
            res['phonemes_tokens'] = torch.LongTensor(self._encoder.encode(res['text']))  # (P,)

        assert res['mel'].shape[1] == len(res['pitch']) == len(res['energy'])
        if 'phonemes_tokens' in res:
            assert len(res['alignment']) == len(res['phonemes_tokens'])

        return res

    def __len__(self) -> int:
        return self.NUM_ENTRIES

    def get_wave(self, idx: int) -> Tensor:
        entry_dirpath = self.get_entry_dirpath(self.get_id_by_idx(idx))
        wave, sr = torchaudio.load(entry_dirpath / 'audio.wav')
        assert sr == self.sample_rate, f'Sample rate of an audio is not equal to {self.sample_rate}'
        return wave

    def get_mel(self, idx: int) -> Tensor:
        entry_dirpath = self.get_entry_dirpath(self.get_id_by_idx(idx))
        return Tensor(np.load(str(entry_dirpath / 'mel.npy'), allow_pickle=True).T)  # (freq, T)

    def get_entry_dirpath(self, id: str) -> Path:
        dirpath = self._data_dir / 'entries' / id
        dirpath.mkdir(parents=True, exist_ok=True)
        return dirpath

    @staticmethod
    def get_id_by_idx(idx: int) -> str:
        return f'{idx+1:05d}'  # assuming zero-indexing

    def _check_entries(self):
        all_entry_files = [
            'audio.wav', 'mel.npy', 'text.txt', 'alignment.npy', 'pitch.npy', 'energy.npy'
        ]
        not_in_all_entry_files = set()

        if len(os.listdir(self._data_dir / 'entries')) == 0:
            not_in_all_entry_files = set(all_entry_files)
        else:
            for entry_dirname in os.listdir(self._data_dir / 'entries'):
                entry_dirpath = self._data_dir / 'entries' / entry_dirname
                entry_files = os.listdir(entry_dirpath)
                for entry_file in all_entry_files:
                    if entry_file not in entry_files:
                        not_in_all_entry_files.add(entry_file)

        if 'audio.wav' in not_in_all_entry_files:
            print('Audios not present for some entries')
            self._download_audios()
        if 'text.txt' in not_in_all_entry_files:
            print('Texts not present for some entries')
            self._download_texts()
        if 'mel.npy' in not_in_all_entry_files:
            print('Mels not present for some entries')
            self._download_mels()
        if 'alignment.npy' in not_in_all_entry_files:
            print('Alignments not present for some entries')
            self._download_alignments()
        if 'pitch.npy' in not_in_all_entry_files:
            print('Pitches not present for some entries')
            self._generate_pitches()
        if 'energy.npy' in not_in_all_entry_files:
            print('Energies not present for some entries')
            self._generate_energies()

    @staticmethod
    def _download_archive(link: str, arch_filepath: Path, extracted_dirpath: Path, desc: str):
        if not extracted_dirpath.exists():
            if not arch_filepath.exists():
                print(f'Downloading LJSpeech {desc}...')
                download_file(link, arch_filepath)
            print('Extracting LJSpeech {desc}...')
            shutil.unpack_archive(arch_filepath, extracted_dirpath.parent)

    def _download_audios(self):
        arch_filepath = self._data_dir / 'LJSpeech-1.1.tar.bz2'
        extracted_dataset_dirpath = self._data_dir / 'LJSpeech-1.1'
        self._download_archive(self.URL_LINKS['audio'], arch_filepath, extracted_dataset_dirpath,
                               desc='audios')

        audio_filenames = sorted(filename for filename in os.listdir(extracted_dataset_dirpath / 'wavs')
                                 if filename.endswith('.wav') and not filename.startswith('.'))
        assert len(audio_filenames) == self.NUM_ENTRIES, f'Only {len(audio_filenames)} audios in the LJSpeech directory'
        print('Moving LJ speech audios to each entry...')
        audio_pattern = re.compile(r'^LJ\d{3}-\d{4}\.wav$')
        for idx, audio_filename in enumerate(tqdm(audio_filenames)):
            if not audio_pattern.match(audio_filename):
                continue
            audio_filepath = extracted_dataset_dirpath / 'wavs' / audio_filename
            id = self.get_id_by_idx(idx)
            entry_dirpath = self.get_entry_dirpath(id)
            shutil.copy(audio_filepath, entry_dirpath / 'audio.wav')

    def _download_texts(self):
        all_texts_filepath = self._data_dir / 'train.txt'
        if not all_texts_filepath.exists():
            download_file(self.URL_LINKS['text'], all_texts_filepath)
        print('Adding text to each entry...')
        for idx, text in enumerate(tqdm(open(all_texts_filepath, 'r'), total=self.NUM_ENTRIES)):
            text = text.strip()
            id = self.get_id_by_idx(idx)
            entry_dirpath = self.get_entry_dirpath(id)
            entry_text_filepath = entry_dirpath / 'text.txt'
            open(entry_text_filepath, 'w').write(text)

    def _download_mels(self):
        arch_filepath = self._data_dir / 'mel.tar.gz'
        extracted_dirpath = self._data_dir / 'mels'
        self._download_archive(self.URL_LINKS['mel'], arch_filepath, extracted_dirpath,
                               desc='mels')

        print('Adding mel to each entry...')
        mel_pattern = re.compile(r'^ljspeech-mel-\d{5}\.npy$')
        for mel_filename in tqdm(os.listdir(extracted_dirpath)):
            if not mel_pattern.match(mel_filename):
                continue
            id = mel_filename[-9:-4]
            entry_dirpath = self.get_entry_dirpath(id)
            shutil.copy(extracted_dirpath / mel_filename, entry_dirpath / 'mel.npy')

    def _download_alignments(self):
        """
        Relevant only for the used encoder
        """
        arch_filepath = self._data_dir / 'alignments.zip'
        extracted_dirpath = self._data_dir / 'alignments'
        self._download_archive(self.URL_LINKS['alignment'], arch_filepath, extracted_dirpath,
                               desc='alignments')

        print('Adding alignment to each entry...')
        alignment_pattern = re.compile(r'^\d+\.npy$')
        for alignment_filename in tqdm(os.listdir(extracted_dirpath)):
            if not alignment_pattern.match(alignment_filename):
                continue
            idx = int(alignment_filename.split('.')[0])
            id = self.get_id_by_idx(idx)
            entry_dirpath = self.get_entry_dirpath(id)
            shutil.copy(extracted_dirpath / alignment_filename, entry_dirpath / 'alignment.npy')

    def _generate_pitches(self, num_workers: int = 10):
        print('Generating pitches...')

        chunk_size = 1000
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for chunk in tqdm(range((len(self) + chunk_size - 1) // chunk_size)):
                futures = []

                for idx in range(chunk*chunk_size, min((chunk+1)*chunk_size, len(self))):
                    id = self.get_id_by_idx(idx)
                    entry_dirpath = self.get_entry_dirpath(id)
                    pitch_filepath = entry_dirpath / 'pitch.npy'
                    if pitch_filepath.exists():
                        continue

                    wave = self.get_wave(idx)
                    mel = self.get_mel(idx)

                    futures.append(pool.submit(generate_pitch, pitch_filepath, wave, mel,
                                               self.sample_rate, self.STFT_HOP_LEN))

                for i, future in enumerate(futures):
                    future.result()

                chunk += 1

    def _generate_energies(self):
        print('Generating energies...')

        for idx in tqdm(range(len(self))):
            id = self.get_id_by_idx(idx)
            entry_dirpath = self.get_entry_dirpath(id)
            energy_filepath = entry_dirpath / 'energy.npy'

            # if energy_filepath.exists():
            #     continue

            mel = self.get_mel(idx)
            energy = torch.linalg.norm(mel, dim=0)  # (freq,)
            np.save(str(energy_filepath), energy)
