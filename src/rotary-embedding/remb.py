import math

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        self.head_dim = head_dim
        self.base = base
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim)
        )
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)

        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)

        self.cos_cached = cos
        self.sin_cached = sin

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(-2)
        if (
            self.cos_cached is None
            or self.sin_cached is None
            or self.cos_cached.size(0) < seq_len
            or self.cos_cached.device != x.device
            or self.cos_cached.dtype != x.dtype
        ):
            self._build_cache(seq_len, x.device, x.dtype)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class RopeSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        attn_mask: broadcastable to (batch, n_heads, seq_len, seq_len), optional
        """
        bsz, seq_len, _ = x.shape

        qkv = self.qkv(x)  # (b, s, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # (b, s, d_model) -> (b, n_heads, s, head_dim)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        y = torch.matmul(attn, v)  # (b, n_heads, s, head_dim)

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out(y)
    

if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    d_model = 8
    n_heads = 2

    x = torch.randn(batch_size, seq_len, d_model)
    attn_mask = torch.ones(batch_size, n_heads, seq_len, seq_len)

    model = RopeSelfAttention(d_model, n_heads)
    output = model(x, attn_mask)
    print(output.shape)  # should be (batch_size, seq_len, d_model)