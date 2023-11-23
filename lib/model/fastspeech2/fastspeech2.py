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
        fft_config = {} if fft_config is None else None
        variance_adaptor_config = {} if variance_adaptor_config is None else None

        self.phonems_embeds = nn.Embedding(phonems_cnt, embedding_dim=phonems_embed_dim)
        self.pos_encoding_1 = PositionalEncoding(phonems_embed_dim, max_sequence_length)

        self.encoder = nn.Sequential(*[
            FFTBlock(phonems_embed_dim, **fft_config)
            for _ in range(encoder_fft_layers)
        ])

        self.variance_adaptor = VarianceAdaptor(phonems_embed_dim, **variance_adaptor_config)
        self.pos_encoding_2 = PositionalEncoding(phonems_embed_dim, max_sequence_length)

        self.decoder = nn.Sequential(*[
            FFTBlock(phonems_embed_dim, **fft_config)
            for _ in range(decoder_fft_layers)
        ])

        self.head = nn.Linear(phonems_embed_dim, out_mel_freqs)

    def forward(self, phonems_tokens: LongTensor, true_duration: Optional[LongTensor] = None,
                **batch) -> Dict[str, Tensor]:
        """
        phonems_tokens: (B, S)
        """
        phonems_embeds = self.phonems_embeds(phonems_tokens)  # (B, S, d=phonems_embed_dim)
        phonems_embeds = self.pos_encoding_1(phonems_embeds)  # (B, S, d)

        encoded_phonems = self.encoder(phonems_embeds)  # (B, S, d)

        output = self.variance_adaptor(encoded_phonems, true_duration)
        aligned_encoded_phonems = output['aligned_phonems_embeds']  # (B, S', d)
        aligned_encoded_phonems = self.pos_encoding_2(aligned_encoded_phonems)

        aligned_decoded_phonems = self.decoder(aligned_encoded_phonems)  # (B, S', d)
        pred_mel_specs = self.head(aligned_decoded_phonems)  # (B, S', mel_freqs)
        output['pred_mel_specs'] = pred_mel_specs

        return output
