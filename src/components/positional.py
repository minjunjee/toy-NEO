"""Fixed position tables: sinusoidal (WTM) and 2D RoPE (BabyTheorist)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sinusoidal_positions(length: int, d_model: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
    )
    encoding = torch.zeros(length, d_model)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: d_model // 2])
    return encoding


def sinusoidal_grid_positions(height: int, width: int, d_model: int) -> Tensor:
    """Row/column sinusoidal encodings in row-major order, [H*W, d_model]."""

    if d_model % 4:
        raise ValueError("2D sinusoidal positions require d_model divisible by 4")
    axis_dim = d_model // 2
    rows = sinusoidal_positions(height, axis_dim)[:, None, :].expand(height, width, axis_dim)
    columns = sinusoidal_positions(width, axis_dim)[None, :, :].expand(height, width, axis_dim)
    return torch.cat((rows, columns), dim=-1).reshape(height * width, d_model)


def rope_2d(grid_dim: int, head_dim: int, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    """cos/sin tables for 2D RoPE, each [grid_dim**2, head_dim]. head_dim % 4 == 0."""

    if head_dim % 4:
        raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}")
    half_dim = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))
    angles = torch.outer(torch.arange(grid_dim).float(), freqs)
    cos_pos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
    sin_pos = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
    cos = torch.zeros(grid_dim, grid_dim, head_dim)
    sin = torch.zeros(grid_dim, grid_dim, head_dim)
    cos[:, :, :half_dim] = cos_pos[:, None, :]
    sin[:, :, :half_dim] = sin_pos[:, None, :]
    cos[:, :, half_dim:] = cos_pos[None, :, :]
    sin[:, :, half_dim:] = sin_pos[None, :, :]
    return cos.view(grid_dim * grid_dim, head_dim), sin.view(grid_dim * grid_dim, head_dim)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: [..., seq, head_dim]; cos/sin: [seq, head_dim]."""

    rotated_half = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
    return x * cos + rotated_half * sin
