"""AdaLN-Zero Transformer block (BabyTheorist transition.py, DiT-style)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from components.blocks.rope import RoPEAttention


class AdaLNZeroBlock(nn.Module):
    """c -> (gamma1, beta1, alpha1, gamma2, beta2, alpha2); zero-init so the block starts as identity.
    grid_dim given: tokens are grid cells and attention uses 2D RoPE (no additive positions needed)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0, grid_dim: int | None = None) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = RoPEAttention(d_model, num_heads, grid_dim, dropout) if grid_dim else nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN_modulation(c).unsqueeze(1).chunk(6, dim=-1)
        h = (1 + gamma1) * self.norm1(x) + beta1
        h = self.attn(h) if isinstance(self.attn, RoPEAttention) else self.attn(h, h, h, need_weights=False)[0]
        x = x + alpha1 * h
        h = (1 + gamma2) * self.norm2(x) + beta2
        return x + alpha2 * self.ffn(h)


class AdaLNZeroStack(nn.Module):
    """AdaLN-Zero blocks over tokens in ``d_model`` with a projected conditioning vector."""

    def __init__(self, token_dim: int, cond_dim: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float, grid_dim: int | None = None) -> None:
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.in_proj = nn.Linear(token_dim, d_model) if d_model != token_dim else nn.Identity()
        self.out_proj = nn.Linear(d_model, token_dim) if d_model != token_dim else nn.Identity()
        self.blocks = nn.ModuleList(AdaLNZeroBlock(d_model, num_heads, d_ff, dropout, grid_dim) for _ in range(num_layers))

    def forward(self, tokens: Tensor, cond: Tensor) -> Tensor:
        c = self.cond_proj(cond.reshape(cond.shape[0], -1))
        x = self.in_proj(tokens)
        for block in self.blocks:
            x = block(x, c)
        return self.out_proj(x)
