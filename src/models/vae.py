"""Codec pretraining model: encoder + decoder only."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from core.registry import DECODERS, ENCODERS, MODELS, build


@dataclass
class VAEOutput:
    logits: Tensor   # (N,H,W,C)
    targets: Tensor  # (N,H,W)
    state: Tensor
    posterior: object | None


class VAE(nn.Module):
    def __init__(self, encoder: dict, decoder: dict) -> None:
        super().__init__()
        self.encoder = build(ENCODERS, encoder)
        self.decoder = build(DECODERS, decoder)

    def forward(self, batch: dict, step: int = 0) -> VAEOutput:
        observations = torch.cat([batch["x"], batch["y"]], dim=0)
        state, posterior = self.encoder.encode(observations)
        return VAEOutput(self.decoder(state), observations, state, posterior)


MODELS["vae"] = VAE
