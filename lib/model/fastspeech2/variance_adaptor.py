from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, LongTensor, nn


class Discretizer(nn.Module):
    def __init__(self, min_value: float = 1., max_value: float = 100.,
                 mode: str = 'linear', vals_cnt: int = 256):
        """
        :param mode: 'linear' or 'log'
        """
        super().__init__()
        self.register_buffer('min_value', torch.tensor(min_value))
        self.register_buffer('max_value', torch.tensor(max_value))
        self.register_buffer('boundaries', None, persistent=False)
        self._mode = mode
        self._vals_cnt = vals_cnt
        self.update_boundaries(min_value, max_value)

    def update_boundaries(self, min_value: float, max_value: float):
        if self._mode == 'linear':
            self.boundaries = torch.linspace(min_value, max_value, self._vals_cnt-1)
        elif self._mode == 'log':
            self.boundaries = torch.logspace(np.log(min_value), np.log(max_value), self._vals_cnt-1, base=np.e)
        else:
            raise NotImplementedError()

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        super().load_state_dict(state_dict, *args, **kwargs)
        self.update_boundaries(self.min_value, self.max_value)

    def forward(self, input: Tensor) -> LongTensor:
        return torch.bucketize(input, self.boundaries).long()


class PredictorConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, dropout_p: float = 0.5, kernel_size: int = 3):
        super().__init__()

        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding='same')
        self.act = nn.ReLU()
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, S, in_dim)

        output : (B, S, out_dim)
        """
        output = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, S, out_dim)
        output = self.act(output)
        output = self.ln(output)
        output = self.dropout(output)
        return output


class Predictor(nn.Module):
    def __init__(self, in_dim: int,
                 dropout_p: float = 0.5, kernels_sizes: Sequence[int] = (3, 3),
                 hidden_dim: int = 256, ff_dim: int = 256):
        super().__init__()

        self.conv_block_1 = PredictorConvBlock(in_dim, hidden_dim, dropout_p, kernels_sizes[0])
        self.conv_block_2 = PredictorConvBlock(hidden_dim, ff_dim, dropout_p, kernels_sizes[1])
        self.head = nn.Linear(ff_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, S, in_dim)

        output : (B, S)
        """
        output = self.conv_block_1(x)       # (B, S, hidden_dim)
        output = self.conv_block_2(output)  # (B, S, ff_dim)
        output = self.head(output)          # (B, S, 1)
        return output.squeeze(-1)


class LengthRegulator(nn.Module):
    @staticmethod
    def get_alignments_matrix(duration: Tensor, alpha: float = 1.):
        """
        duration: (B, S)

        output: (B, S', S)
        """
        batch_dim, max_len = duration.shape
        duration = (alpha * duration).round().long()

        new_max_len = duration.sum(axis=1).max()
        alignments_matrix = torch.zeros(batch_dim, new_max_len, max_len)  # (B, S', S)

        for k in range(batch_dim):
            already_picked_cnt = 0
            for phonem_idx, phonem_cnt in enumerate(duration[k]):
                if phonem_cnt != 0:
                    alignments_matrix[k, already_picked_cnt:already_picked_cnt+phonem_cnt, phonem_idx] = 1
                    already_picked_cnt += phonem_cnt

        return alignments_matrix

    def forward(self, phonems_embeds: Tensor, duration: Tensor, alpha: float = 1.):
        """
        phonems_embeds: (B, S, d)
        duration: (B, S)

        output: (B, S', d)
        """
        alignments_matrix = self.get_alignments_matrix(duration, alpha).to(phonems_embeds.device)  # (B, S', S)
        result = alignments_matrix @ phonems_embeds  # (B, S', d)
        return result


class VarianceAdaptor(nn.Module):
    def __init__(self,
                 in_dim: int,
                 pitch_discretizer_scale: str = 'log',
                 pitch_discretizer_values: int = 256,
                 energy_discretizer_scale: str = 'linear',
                 energy_discretizer_values: int = 256,
                 predictor_config=None):
        super().__init__()
        predictor_config = {} if predictor_config is None else predictor_config
        self.duration_predictor = Predictor(in_dim, **predictor_config)  # predicts logs
        self.length_regulator = LengthRegulator()

        self.pitch_predictor = Predictor(in_dim, **predictor_config)  # predicts values directly
        self.pitch_discretizer = Discretizer(mode=pitch_discretizer_scale, vals_cnt=pitch_discretizer_values)
        self.pitch_embeddings = nn.Embedding(pitch_discretizer_values, embedding_dim=in_dim)

        self.energy_predictor = Predictor(in_dim, **predictor_config)  # predicts values directly
        self.energy_discretizer = Discretizer(mode=energy_discretizer_scale, vals_cnt=energy_discretizer_values)
        self.energy_embeddings = nn.Embedding(energy_discretizer_values, embedding_dim=in_dim)

    def forward(self, phonems_embeds: Tensor, true_duration: Optional[LongTensor] = None) -> Dict[str, Tensor]:
        """
        phonems_embeds : (B, S, in_dim)
        true_duration : (B, S) - integer durations >= 1

        returns :
        {
            pred_log_duration: (B, S)
            pred_pitch: (B, S')
            pred_energy: (B, S')
            aligned_phonemes_embeds: (B, S', in_dim)

            where S' is the maximal estimated length of the mel spectrogram
        }
        """
        res = {}
        pred_log_duration = self.duration_predictor(phonems_embeds)
        res['pred_log_duration'] = pred_log_duration  # (B, S)
        if true_duration is None:
            assert not self.training, 'You should pass true duration to the variance adaptor during training'
            duration = torch.exp(pred_log_duration)  # (B, S)
        else:
            assert self.training, 'Do not pass true duration during inference'
            duration = true_duration
        aligned_phonems_embeds = self.length_regulator(phonems_embeds, duration)  # (B, S', in_dim)

        pred_pitch = self.pitch_predictor(aligned_phonems_embeds)        # (B, S')
        res['pred_pitch'] = pred_pitch
        pitch_discrete_values = self.pitch_discretizer(pred_pitch)       # (B, S')
        pitch_embeddings = self.pitch_embeddings(pitch_discrete_values)  # (B, S', in_dim)

        pred_energy = self.energy_predictor(aligned_phonems_embeds)      # (B, S')
        res['pred_energy'] = pred_energy
        energy_discrete_values = self.energy_discretizer(pred_energy)
        energy_embeddings = self.energy_embeddings(energy_discrete_values)  # (B, S', in_dim)

        res['aligned_phonemes_embeds'] = aligned_phonems_embeds + pitch_embeddings + energy_embeddings
        return res
