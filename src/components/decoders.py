"""Decoder slot: state tokens -> per-cell logits. Registered under DECODERS."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from components.blocks.rope import RoPEEncoder
from components.blocks.transformer import init_linear_scaled
from core.registry import DECODERS


class CNNVAEDecoder(nn.Module):
    """Official GridWorld VAE decoder. (B, num_state_tokens, state_dim) -> (B, 10, 10, num_colors)."""

    def __init__(self, grid_size: int = 10, num_colors: int = 9, state_dim: int = 32, num_state_tokens: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        if grid_size != 10:
            raise ValueError("the paper GridWorld CNN supports exactly 10x10 grids")
        self.num_state_tokens = num_state_tokens
        self.state_dim = state_dim
        flattened_dim = 256 * 2 * 2
        self.from_state = nn.Linear(num_state_tokens * state_dim, flattened_dim)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0, output_padding=0), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(dropout),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(dropout),
            nn.Conv2d(64, num_colors, 3, padding=1),
        )

    def forward(self, state: Tensor) -> Tensor:
        batch_size = state.shape[0]
        features = self.from_state(state.reshape(batch_size, -1)).view(batch_size, 256, 2, 2)
        return self.deconv_layers(features).permute(0, 2, 3, 1)


class LinearHead(nn.Module):
    """WTM execution head: LayerNorm (no affine) + bias-free projection. (B,L,d) -> (B,L,vocab); spatial given -> (B,H,W,vocab)."""

    def __init__(self, d_model: int, vocab: int, layer_norm_eps: float = 1e-6, spatial: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.spatial = tuple(spatial) if spatial else None
        self.output_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.logits_projection = nn.Linear(d_model, vocab, bias=False)
        init_linear_scaled(self.logits_projection)

    def forward(self, tokens: Tensor) -> Tensor:
        logits = self.logits_projection(self.output_norm(tokens)).float()
        return logits.view(tokens.shape[0], *self.spatial, -1) if self.spatial else logits


class ViTLatentDecoder(nn.Module):
    """BabyTheorist SingleGridDecoderRoPE: output tokens ++ state tokens -> RoPE ViT -> logits (B,H,W,C)."""

    def __init__(self, grid_size: int, num_colors: int, state_dim: int, num_state_tokens: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.seq_len = grid_size * grid_size
        self.from_state = nn.Linear(state_dim, d_model)
        scale = d_model**-0.5
        self.output_tokens = nn.Parameter(scale * torch.randn(self.seq_len, d_model))
        self.latent_input_pe = nn.Parameter(scale * torch.randn(num_state_tokens, d_model))
        self.decoder = RoPEEncoder(d_model, num_layers, num_heads, d_ff, dropout, grid_size)
        self.to_logits = nn.Linear(d_model, num_colors)

    def forward(self, state: Tensor) -> Tensor:
        batch_size = state.shape[0]
        state_tokens = self.from_state(state) + self.latent_input_pe.to(state.dtype)
        output_tokens = self.output_tokens.unsqueeze(0).expand(batch_size, -1, -1).to(state_tokens.dtype)
        x = self.decoder(torch.cat([output_tokens, state_tokens], dim=1))
        return self.to_logits(x[:, : self.seq_len]).view(batch_size, self.grid_size, self.grid_size, self.num_colors)


DECODERS["cnn_vae"] = CNNVAEDecoder
DECODERS["linear_head"] = LinearHead
DECODERS["vit_latent"] = ViTLatentDecoder
