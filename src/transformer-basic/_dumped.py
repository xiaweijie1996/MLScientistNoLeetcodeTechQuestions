from typing import Optional

import torch
import torch.nn as nn


# -----------------------------
# Manual Normalization
# -----------------------------
class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.lamda = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        return ((input - mean) / torch.sqrt(var + self.eps)) * self.lamda + self.beta


class RMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.lamda = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        rms = input.pow(2).mean(dim=-1, keepdim=True)
        return (input / torch.sqrt(rms + self.eps)) * self.lamda


# -----------------------------
# Manual Softmax
# -----------------------------
class SoftMax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        max_val = input.max(dim=self.dim, keepdim=True).values
        input = input - max_val
        exp_x = torch.exp(input)
        return exp_x / exp_x.sum(dim=self.dim, keepdim=True)


# -----------------------------
# Mask
# -----------------------------
def causal_mask(seq_len: int) -> torch.Tensor:
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))


# -----------------------------
# Manual Positional Encoding
# -----------------------------
def positional_encoding(seq_len: int, emb_dim: int) -> torch.Tensor:
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (S, 1)
    div_term = torch.arange(0, emb_dim, 2, dtype=torch.float32)
    div_term = 1.0 / (10000 ** (div_term / emb_dim))

    pe = torch.zeros(seq_len, emb_dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)

    # handle odd emb_dim safely
    if emb_dim % 2 == 0:
        pe[:, 1::2] = torch.cos(position * div_term)
    else:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])

    return pe


# -----------------------------
# Manual Multi-Head Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        head_num: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        if emb_dim % head_num != 0:
            raise ValueError("Embedding dimension must be divisible by head number")

        self.emb_dim = emb_dim
        self.head_num = head_num
        self.head_dim = emb_dim // head_num

        self.qkv_proj = nn.Linear(emb_dim, emb_dim * 3)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        self.softmax = SoftMax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input: torch.Tensor,               # (B, S, E)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, s, _ = input.shape

        qkv = self.qkv_proj(input)         # (B, S, 3E)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(b, s, self.head_num, self.head_dim).transpose(1, 2)  # (B, H, S, D)
        k = k.view(b, s, self.head_num, self.head_dim).transpose(1, 2)  # (B, H, S, D)
        v = v.view(b, s, self.head_num, self.head_dim).transpose(1, 2)  # (B, H, S, D)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B,H,S,S)

        if mask is not None:
            # allow mask shape (S,S), (B,S,S), or (B,1,S,S)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)   # (1,1,S,S)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)                # (B,1,S,S)

            scores = scores.masked_fill(~mask, float("-inf"))

        attn = self.softmax(scores)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)                  # (B,H,S,D)
        output = output.transpose(1, 2).contiguous().view(b, s, self.emb_dim)  # (B,S,E)

        return self.out_proj(output)


# -----------------------------
# Positionwise FeedForward
# -----------------------------
class FeedForward(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Single Encoder Layer
# -----------------------------
class EncoderLayer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        head_num: int,
        ff_hidden_dim: int,
        dropout: float = 0.1,
        norm_type: str = "layernorm",   # "layernorm" or "rmsnorm"
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            emb_dim=emb_dim,
            head_num=head_num,
            dropout=dropout,
        )
        self.ffn = FeedForward(
            emb_dim=emb_dim,
            hidden_dim=ff_hidden_dim,
            dropout=dropout,
        )

        if norm_type.lower() == "layernorm":
            self.norm1 = LayerNorm(emb_dim)
            self.norm2 = LayerNorm(emb_dim)
        elif norm_type.lower() == "rmsnorm":
            self.norm1 = RMSNorm(emb_dim)
            self.norm2 = RMSNorm(emb_dim)
        else:
            raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'")

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                      # (B, S, E)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention + residual + norm
        attn_out = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # FFN + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


# -----------------------------
# Transformer Encoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        head_num: int,
        ff_hidden_dim: int,
        num_layers: int,
        max_seq_len: int,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)

        pe = positional_encoding(max_seq_len, emb_dim)  # (S, E)
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, S, E)

        self.layers = nn.ModuleList([
            EncoderLayer(
                emb_dim=emb_dim,
                head_num=head_num,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout,
                norm_type=norm_type,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,                 # (B, S)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, s = input_ids.shape

        x = self.token_embedding(input_ids)      # (B, S, E)
        x = x + self.pe[:, :s, :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        return x


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # Hyperparameters
    batch_size = 2
    seq_len = 5
    vocab_size = 100
    emb_dim = 8
    head_num = 2
    ff_hidden_dim = 32
    num_layers = 3
    max_seq_len = 20

    # Dummy token ids
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print("input_ids shape:", input_ids.shape)

    # Encoder without causal mask (standard encoder)
    encoder = Encoder(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        head_num=head_num,
        ff_hidden_dim=ff_hidden_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        dropout=0.1,
        norm_type="layernorm",
    )

    output = encoder(input_ids)
    print("encoder output shape:", output.shape)   # (B, S, E)
    print(output)

    # Optional: test with causal mask
    mask = causal_mask(seq_len)   # (S, S)
    output_masked = encoder(input_ids, mask=mask)
    print("masked encoder output shape:", output_masked.shape)