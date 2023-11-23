from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        pos = torch.arange(max_len)
        i = torch.arange(0, embed_dim, 2)
        pos, i = torch.meshgrid(pos, i, indexing='ij')
        arg = pos / (10_000**(i / embed_dim))

        pe = torch.zeros(max_len, embed_dim)

        pe[:, ::2] = torch.sin(arg)
        pe[:, 1::2] = torch.cos(arg)[:, :arg.shape[1] - (embed_dim % 2)]

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int = 2,
                 d_v: Optional[int] = None, d_k: Optional[int] = None):
        super().__init__()
        if d_v is None:
            assert d_model % nhead == 0
            d_v = d_model // nhead
        if d_k is None:
            assert d_model % nhead == 0
            d_k = d_model // nhead

        self.nhead = nhead
        self.d_k = d_k
        self.d_v = d_v

        self.wq = nn.Linear(d_model, d_k * nhead)
        self.wk = nn.Linear(d_model, d_k * nhead)
        self.wv = nn.Linear(d_model, d_v * nhead)

        self.wo = nn.Linear(d_v * nhead, d_model)

    def reset_parameters(self):
        nn.init.normal_(self.wq.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.wk.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.wv.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_v)))

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        """
        q: (B, S1, d_model)
        k: (B, S2, d_model)
        v: (B, S2, d_model)

        output: (B, S1, d_model)
        """
        batch_dim, s1 = q.shape[:2]
        s2 = k.shape[1]
        nhead, d_k, d_v = self.nhead, self.d_k, self.d_v

        q = self.wq(q).view(batch_dim, s1, nhead, d_k)  # (B, S1, d_k * nhead) -> (B, S1, nhead, d_k)
        k = self.wk(k).view(batch_dim, s2, nhead, d_k)  # (B, S2, d_k * nhead) -> (B, S2, nhead, d_k)
        v = self.wv(v).view(batch_dim, s2, nhead, d_v)  # (B, S2, d_v * nhead) -> (B, S2, nhead, d_v)

        q = q.permute(2, 0, 1, 3).contiguous()  # (nhead, B, S1, d_k)
        k = k.permute(2, 0, 1, 3).contiguous()  # (nhead, B, S2, d_k)
        v = v.permute(2, 0, 1, 3).contiguous()  # (nhead, B, S2, d_v)

        q = q.view(-1, s1, d_k)  # (nhead*B, S1, d_k)
        k = k.view(-1, s2, d_k)  # (nhead*B, S2, d_k)
        v = v.view(-1, s2, d_v)  # (nhead*B, S2, d_v)

        output = F.scaled_dot_product_attention(q, k, v)  # (nhead*B, S1, d_v)
        output = output.view(nhead, batch_dim, s1, d_v)   # (nhead, B, S1, d_v)
        output = output.permute(1, 2, 0, 3).contiguous()  # (B, S1, nhead, d_v)
        output = output.view(batch_dim, s1, -1)  # (B, S1, nhead * d_v)
        output = self.wo(output)  # (B, S1, d_model)

        return output


class FFTBlock(nn.Module):
    def __init__(self, d_model: int, dropout_p: float = 0.1,
                 kernels_sizes: Sequence[int] = (9, 1),
                 d_hidden: Optional[int] = None, **kwargs):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_model

        self.attn = MultiHeadAttention(d_model, **kwargs)
        self.attn_dropout = nn.Dropout(dropout_p)
        self.attn_ln = nn.LayerNorm(d_model)

        self.conv1 = nn.Conv1d(d_model, d_hidden, kernel_size=kernels_sizes[0], padding='same')
        self.conv_act = nn.ReLU()
        self.conv2 = nn.Conv1d(d_hidden, d_model, kernel_size=kernels_sizes[1], padding='same')
        self.conv_dropout = nn.Dropout(dropout_p)
        self.conv_ln = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, S, d_model)

        output : (B, S, d_model)
        """
        attn_output = self.attn_dropout(self.attn(x, x, x))  # (B, S, d_model)
        attn_output += x                                     # (B, S, d_model)
        attn_output = self.attn_ln(attn_output)              # (B, S, d_model)

        x = attn_output
        conv_output = self.conv2(self.conv_act(self.conv1(x.transpose(1, 2)))).transpose(1, 2)  # (B, S, d_model)
        conv_output = self.conv_dropout(conv_output)
        conv_output += x
        conv_output = self.conv_ln(conv_output)

        return conv_output
