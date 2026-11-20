"""Pre-norm, bias-free Transformer blocks (WTM modules/transformer.py)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def prefix_lm_mask(prefix_length: int, total_length: int) -> Tensor:
    positions = torch.arange(total_length)
    prefix_queries = positions < prefix_length
    return (prefix_queries[:, None] & (positions[None, :] < prefix_length)) | (
        ~prefix_queries[:, None] & (positions[None, :] <= positions[:, None])
    )


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, length, d_model = x.shape
        qkv = self.qkv(x).view(batch_size, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if mask is not None:
            mask = mask[None, None]
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.output(attended.transpose(1, 2).reshape(batch_size, length, d_model))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, layer_norm_eps: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.attention = SelfAttention(d_model, num_heads, dropout)
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False), nn.SiLU(), nn.Linear(d_ff, d_model, bias=False)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.dropout(self.attention(self.attention_norm(x), mask))
        return x + self.dropout(self.feed_forward(self.feed_forward_norm(x)))


class TransformerStack(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float, layer_norm_eps: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, dropout, layer_norm_eps) for _ in range(num_layers)
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


def init_linear_scaled(module: nn.Module) -> None:
    """WTM init: N(0, in_features**-0.5) on every Linear weight."""

    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=module.in_features**-0.5)


def init_embedding_scaled(embedding: nn.Embedding) -> None:
    nn.init.normal_(embedding.weight, std=embedding.embedding_dim**-0.5)
