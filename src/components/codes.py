"""Code slot: turn programmer output into the value the executor consumes. Registered under CODES."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.registry import CODES


@dataclass(frozen=True)
class CodeOutput:
    values: Tensor            # what the executor receives
    loss_per_sample: Tensor   # (B,) code regulariser (VQ commitment+codebook, Gaussian KL, ...)
    indices: Tensor | None    # (B*num_tokens,) discrete ids when they exist
    logits: Tensor | None
    temperature: float


class VQ(nn.Module):
    """Official NEO ActionQuantizer: cosine-distance VQ, stochastic softmax sampling with a cosine tau schedule."""

    def __init__(self, size: int, dim: int, num_tokens: int = 1, commitment: float = 0.25, stochastic: bool = True, tau_start: float = 0.3, tau_end: float = 0.1, tau_steps: int = 1, commitment_grad: bool = False) -> None:
        super().__init__()
        self.size = size
        self.dim = dim
        self.num_tokens = num_tokens
        self.commitment_cost = commitment
        self.stochastic = stochastic
        self.commitment_grad = commitment_grad
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_steps = tau_steps
        self.current_step = 0
        self.embedding = nn.Embedding(size, dim)
        self.embedding.weight.data.uniform_(-1.0 / size, 1.0 / size)

    def get_temperature(self) -> float:
        if self.current_step >= self.tau_steps:
            return self.tau_end
        cosine = math.cos(math.pi * self.current_step / self.tau_steps)
        return self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (1.0 + cosine)

    def step(self) -> None:
        self.current_step += 1

    def codebook(self, indices: Tensor) -> Tensor:
        return F.normalize(self.embedding(indices), dim=-1)

    def forward(self, actions: Tensor, training: bool = True) -> CodeOutput:
        with torch.autocast(device_type=actions.device.type, enabled=False):
            flattened = F.normalize(actions.flatten(end_dim=-2).float(), dim=-1)
            embedding = F.normalize(self.embedding.weight, dim=-1)
            distances = flattened.square().sum(dim=1, keepdim=True) + embedding.square().sum(dim=1) - 2 * flattened @ embedding.T
            temperature = self.get_temperature()
            if self.stochastic:
                if training:
                    logits = -distances / temperature
                    indices = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
                else:
                    logits = -distances
                    indices = logits.argmax(dim=-1)
            else:
                temperature = 0.0
                logits = distances.reciprocal()
                indices = distances.argmin(dim=-1)
            quantized = self.codebook(indices).view(actions.shape)
            normalized = F.normalize(actions.float(), dim=-1)
            # Official loss_per_sample detaches the action in both terms (quantizer.py); commitment_grad=True is the VQ-VAE form.
            committed = (quantized.detach() - normalized) if self.commitment_grad else (quantized - normalized.detach())
            commitment = self.commitment_cost * committed.square().mean(dim=(1, 2))
            codebook = (quantized - normalized.detach()).square().mean(dim=(1, 2))
            straight_through = normalized + (quantized - normalized).detach()
        return CodeOutput(straight_through, commitment + codebook, indices, logits, temperature)


class Gaussian(nn.Module):
    """Cont-Mono / LPN code: diagonal Gaussian over the programmer output, KL to N(0, I) per sample."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mu_projection = nn.Linear(dim, dim)
        self.logvar_projection = nn.Linear(dim, dim)

    def forward(self, actions: Tensor, training: bool = True) -> CodeOutput:
        mean = self.mu_projection(actions)
        if not training:
            return CodeOutput(mean, torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype), None, None, 0.0)
        log_variance = self.logvar_projection(actions)
        values = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        kl = -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp()).mean(dim=(1, 2))
        return CodeOutput(values, kl, None, None, 0.0)


CODES["vq"] = VQ
CODES["gaussian"] = Gaussian
