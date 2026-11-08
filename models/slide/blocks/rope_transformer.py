"""RoPE Transformer encoder for the transformer_initial_state ablation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x of shape [B, H, T, D]."""
    # cos/sin: [T, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [T, dim/2]
        emb = torch.stack((freqs, freqs), dim=-1).flatten(-2)  # [T, dim]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE; unmasked (full) attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rope(T, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)  # [B, H, T, Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    """Post-norm transformer block: LN(x + Attn(x)), LN(x + FFN(x))."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = RoPEAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class RoPETransformerEncoder(nn.Module):
    """Stack of RoPE transformer blocks over a token sequence."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : Tensor [B, T, D]

        Returns
        -------
        Tensor [B, T, D]
        """
        x = tokens
        for layer in self.layers:
            x = layer(x)
        return x
