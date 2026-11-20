"""Encoder slot: grid -> state tokens. Registered under ENCODERS."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from components.blocks.rope import RoPEEncoder
from components.blocks.transformer import init_embedding_scaled
from core.registry import ENCODERS


@dataclass(frozen=True)
class GaussianPosterior:
    mean: Tensor
    log_variance: Tensor

    def sample(self) -> Tensor:
        std = torch.exp(0.5 * self.log_variance)
        return self.mean + torch.randn_like(std) * std

    def kl_to_standard_normal(self) -> Tensor:
        return -0.5 * torch.sum(1 + self.log_variance - self.mean.square() - self.log_variance.exp()) / self.mean.shape[0]


def embed_grid(grid: Tensor, embedding: nn.Embedding) -> Tensor:
    """(B,H,W) int or (B,H,W,C) soft -> (B,H,W,D)."""

    if grid.ndim == 4:
        return torch.einsum("bhwc,cd->bhwd", grid, embedding.weight)
    return embedding(grid)


class CNNVAEEncoder(nn.Module):
    """Official GridWorld VAE encoder (learning-to-theorize vae.py). State (B, num_state_tokens, state_dim)."""

    def __init__(
        self,
        grid_size: int = 10,
        num_colors: int = 9,
        state_dim: int = 32,
        num_state_tokens: int = 1,
        dropout: float = 0.0,
        variational: bool = True,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        if grid_size != 10:
            raise ValueError("the paper GridWorld CNN supports exactly 10x10 grids")
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.state_dim = state_dim
        self.num_state_tokens = num_state_tokens
        self.variational = variational
        self.deterministic = deterministic
        embedding_dim = 16
        self.embedding = nn.Embedding(num_colors, embedding_dim)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(dropout),
            nn.Conv2d(128, 256, 3, stride=2, padding=0), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout2d(dropout),
        )
        flattened_dim = 256 * 2 * 2
        output_dim = num_state_tokens * state_dim
        self.to_state = nn.Linear(flattened_dim, output_dim)
        if variational:
            self.to_mean = nn.Linear(flattened_dim, output_dim)
            self.to_logvar = nn.Linear(flattened_dim, output_dim)

    def encode(self, grid: Tensor) -> tuple[Tensor, GaussianPosterior | None]:
        batch_size = grid.shape[0]
        features = self.conv_layers(embed_grid(grid, self.embedding).permute(0, 3, 1, 2)).reshape(batch_size, -1)
        shape = (batch_size, self.num_state_tokens, self.state_dim)
        if not self.variational:
            return F.normalize(self.to_state(features).view(shape), dim=-1), None
        posterior = GaussianPosterior(self.to_mean(features).view(shape), self.to_logvar(features).view(shape))
        state = posterior.mean if self.deterministic else posterior.sample()
        return state, posterior

    def forward(self, grid: Tensor) -> Tensor:
        return self.encode(grid)[0]


class TokenEmbedEncoder(nn.Module):
    """WTM NLI state: one embedding per cell, no positions. (B,H,W) -> (B,H*W,d)."""

    def __init__(self, vocab: int, d_model: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab, d_model)
        init_embedding_scaled(self.embedding)

    def forward(self, grid: Tensor) -> Tensor:
        if grid.is_floating_point():  # (B,H,W,vocab) color distribution: soft re-embedding
            return self.embed_soft(grid.reshape(grid.shape[0], -1, grid.shape[-1]))
        return self.embedding(grid.reshape(grid.shape[0], -1))

    def encode(self, grid: Tensor) -> tuple[Tensor, None]:
        """VAE-model interface for codec pretraining: no posterior."""

        return self(grid), None

    def embed_soft(self, probs: Tensor) -> Tensor:
        """(B,L,vocab) distribution -> (B,L,d), the NLI re-embedding step."""

        return probs @ self.embedding.weight


class ViTLatentEncoder(nn.Module):
    """BabyTheorist SingleGridEncoderRoPE: grid tokens ++ latent tokens -> RoPE ViT -> latent tokens."""

    def __init__(
        self,
        grid_size: int,
        num_colors: int,
        state_dim: int,
        num_state_tokens: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        variational: bool = False,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.state_dim = state_dim
        self.num_state_tokens = num_state_tokens
        self.d_model = d_model
        self.variational = variational
        self.deterministic = deterministic
        self.embedding = nn.Embedding(num_colors, d_model)
        scale = d_model**-0.5
        self.latent_tokens = nn.Parameter(scale * torch.randn(num_state_tokens, d_model))
        self.latent_token_pos = nn.Parameter(scale * torch.randn(num_state_tokens, d_model))
        self.encoder = RoPEEncoder(d_model, num_layers, num_heads, d_ff, dropout, grid_size)
        self.to_state = nn.Linear(d_model, state_dim)
        if variational:
            self.to_mean = nn.Linear(d_model, state_dim)
            self.to_logvar = nn.Linear(d_model, state_dim)

    def encode(self, grid: Tensor) -> tuple[Tensor, GaussianPosterior | None]:
        batch_size = grid.shape[0]
        embedded = embed_grid(grid, self.embedding).view(batch_size, -1, self.d_model)
        latent = (self.latent_tokens + self.latent_token_pos).unsqueeze(0).expand(batch_size, -1, -1)
        pooled = self.encoder(torch.cat([embedded, latent.to(embedded.dtype)], dim=1))[:, -self.num_state_tokens:]
        if not self.variational:
            return self.to_state(pooled), None
        posterior = GaussianPosterior(self.to_mean(pooled), self.to_logvar(pooled))
        return (posterior.mean if self.deterministic else posterior.sample()), posterior

    def forward(self, grid: Tensor) -> Tensor:
        return self.encode(grid)[0]


ENCODERS["cnn_vae"] = CNNVAEEncoder
ENCODERS["token_embed"] = TokenEmbedEncoder
ENCODERS["vit_latent"] = ViTLatentEncoder
