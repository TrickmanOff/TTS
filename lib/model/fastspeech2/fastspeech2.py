from typing import Dict, Optional

from torch import LongTensor, Tensor, nn

from lib.model.base_model import BaseModel
from .feed_forward_transformer import PositionalEncoding, FFTBlock
from .variance_adaptor import VarianceAdaptor


class FastSpeech2(BaseModel):
    def __init__(self, phonems_cnt: int, phonems_embed_dim: int = 256, max_sequence_length: int = 3000,
                 encoder_fft_layers: int = 4, decoder_fft_layers: int = 4, out_mel_freqs: int = 80,
                 fft_config: Optional[Dict] = None, variance_adaptor_config: Optional[Dict] = None):
        super().__init__()
        fft_config = {} if fft_config is None else fft_config
        variance_adaptor_config = {} if variance_adaptor_config is None else variance_adaptor_config

        self.phonems_embeds = nn.Embedding(phonems_cnt, embedding_dim=phonems_embed_dim)
        self.pos_encoding_1 = PositionalEncoding(phonems_embed_dim, max_sequence_length)

        self.encoder = nn.Sequential(*[
            FFTBlock(phonems_embed_dim, **fft_config)
            for _ in range(encoder_fft_layers)
        ])
        print(variance_adaptor_config)
        self.variance_adaptor = VarianceAdaptor(phonems_embed_dim, **variance_adaptor_config)
        self.pos_encoding_2 = PositionalEncoding(phonems_embed_dim, max_sequence_length)

        self.decoder = nn.Sequential(*[
            FFTBlock(phonems_embed_dim, **fft_config)
            for _ in range(decoder_fft_layers)
        ])

        self.head = nn.Linear(phonems_embed_dim, out_mel_freqs)

    def forward(self, phonemes_tokens: LongTensor, true_duration: Optional[LongTensor] = None,
                **batch) -> Dict[str, Tensor]:
        """
        phonemes_tokens: (B, S)

        output:
            pred_log_duration: (B, S)
            pred_pitch:        (B, S')
            pred_energy:       (B, S')
            pred_mel_spec:     (B, freqs, S')
        """
        phonemes_embeds = self.phonems_embeds(phonemes_tokens)  # (B, S, d=phonems_embed_dim)
        phonemes_embeds = self.pos_encoding_1(phonemes_embeds)  # (B, S, d)

        encoded_phonems = self.encoder(phonemes_embeds)  # (B, S, d)

        output = self.variance_adaptor(encoded_phonems, true_duration)
        aligned_encoded_phonems = output.pop('aligned_phonemes_embeds')  # (B, S', d)
        aligned_encoded_phonems = self.pos_encoding_2(aligned_encoded_phonems)

        aligned_decoded_phonems = self.decoder(aligned_encoded_phonems)  # (B, S', d)
        pred_mel_specs = self.head(aligned_decoded_phonems)  # (B, S', mel_freqs)
        output['pred_mel_spec'] = pred_mel_specs.transpose(1, 2)

        return output
