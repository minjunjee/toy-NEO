"""Post-norm ViT layers with 2D RoPE on grid tokens (BabyTheorist _transformer.py, V-bias dropped)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from components.positional import apply_rope, rope_2d


class RoPEAttention(nn.Module):
    """Self-attention where the first grid_dim**2 * tokens_per_cell tokens (cell-major) get 2D RoPE and trailing tokens do not."""

    def __init__(self, d_model: int, num_heads: int, grid_dim: int, dropout: float = 0.0, tokens_per_cell: int = 1) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.grid_seq_len = grid_dim * grid_dim * tokens_per_cell
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        cos, sin = rope_2d(grid_dim, self.head_dim)
        self.register_buffer("rope_cos", cos.repeat_interleave(tokens_per_cell, dim=0), persistent=False)
        self.register_buffer("rope_sin", sin.repeat_interleave(tokens_per_cell, dim=0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, d_model = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        n = self.grid_seq_len
        q = torch.cat([apply_rope(q[:, :, :n], self.rope_cos, self.rope_sin), q[:, :, n:]], dim=2)
        k = torch.cat([apply_rope(k[:, :, :n], self.rope_cos, self.rope_sin), k[:, :, n:]], dim=2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(batch, seq, d_model))


class RoPEEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, grid_dim: int, tokens_per_cell: int = 1) -> None:
        super().__init__()
        self.attn = RoPEAttention(d_model, num_heads, grid_dim, dropout, tokens_per_cell)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self.dropout1(self.attn(x)))
        return self.norm2(x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x))))))


class RoPEEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float, grid_dim: int, tokens_per_cell: int = 1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            RoPEEncoderLayer(d_model, num_heads, d_ff, dropout, grid_dim, tokens_per_cell) for _ in range(num_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
