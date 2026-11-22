"""A codebook whose codes are indexed by a kind and an integer amount. Registered as CODES["vq_orbit"].

The lab VQ is a flat table: code 13 and code 42 have nothing to do with each other, so "move up by 2" and "move up by 3"
are two unrelated entries and an amount that was never trained has no entry at all. Here every code has two indices, a
kind g and an integer amount t, and the code vectors of one kind are tied together by how they are computed:

  rotation   c(g, t) = exp(t * A_g) b_g     A_g = M_g - M_g^T is skew-symmetric, so exp(t * A_g) is a rotation; the codes
                                            of one kind are the same rotation applied t times to one start vector
  line       c(g, t) = normalize(b_g + t * d_g)     the same indices on a straight line, to tell "rotation" from "ordered"
  free       c(g, t) = E[g, t]              a free vector per code, as in the lab VQ; only the indices carry the structure

M_g, b_g, d_g and E are trained by gradient through the lab VQ's codebook loss. Which code a pair gets is still the lab's
rule: the code nearest to the programmer's output (cosine distance), over all kinds and amounts. `shared_base` gives all
kinds one start vector b; with amount 0 in the list that vector is one "do nothing" code common to all kinds.

The indices let amounts be added: `sums[i, j]` is the index of (g, a + b) when code i is (g, a) and code j is (g, b) and
a + b is in the list of amounts, else -1. methods/neo_compose.py uses it to ask the executor that running (g, a) and then
(g, b) gives what (g, a + b) gives in one step.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from components.codes import VQ, CodeOutput
from core.registry import CODES

FAMILIES = ("rotation", "line", "free")


class OrbitVQ(VQ):
    def __init__(self, kinds: int, amounts: list[int], dim: int, family: str = "rotation", shared_base: bool = False, angle: float = 0.4,
                 size: int | None = None, **kwargs) -> None:
        amounts = [int(a) for a in amounts]
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if len(set(amounts)) != len(amounts):
            raise ValueError("amounts must be distinct integers")
        if size is not None and size != kinds * len(amounts):
            raise ValueError(f"size must be kinds * len(amounts) = {kinds * len(amounts)}")
        super().__init__(kinds * len(amounts), dim, **kwargs)
        self.kinds, self.family, self.shared_base = kinds, family, shared_base
        self.register_buffer("amounts", torch.tensor(amounts, dtype=torch.long))
        position = {a: i for i, a in enumerate(amounts)}
        sums = torch.full((self.size, self.size), -1, dtype=torch.long)
        for g in range(kinds):
            for a in amounts:
                for b in amounts:
                    if a + b in position:
                        sums[g * len(amounts) + position[a], g * len(amounts) + position[b]] = g * len(amounts) + position[a + b]
        self.register_buffer("sums", sums)
        # running share of the training pairs that picked each code (decay 0.99), for telling codes in use from idle ones
        self.register_buffer("usage", torch.zeros(self.size), persistent=False)  # not in checkpoints: it is rebuilt within a few hundred steps
        if family != "free":
            del self.embedding
            self.base = nn.Parameter(F.normalize(torch.randn(1 if shared_base else kinds, dim), dim=-1))
            if family == "rotation":
                # a unit vector is turned by about `angle` radians per unit of amount at the start
                self.generator = nn.Parameter(torch.randn(kinds, dim, dim) * angle / math.sqrt(2.0 * max(1, dim - 1)))
            else:
                self.direction = nn.Parameter(F.normalize(torch.randn(kinds, dim), dim=-1) * math.tan(angle))

    def kind_and_amount(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        return indices // len(self.amounts), self.amounts[indices % len(self.amounts)]

    def table(self) -> Tensor:
        """All code vectors, unit length, (kinds * amounts, dim); row g * len(amounts) + i is kind g with the i-th amount."""

        with torch.autocast(device_type=self.amounts.device.type, enabled=False):
            if self.family == "free":
                return F.normalize(self.embedding.weight.float(), dim=-1)
            base = self.base.float().expand(self.kinds, -1)
            amounts = self.amounts.to(base.dtype)
            if self.family == "rotation":
                skew = self.generator.float() - self.generator.float().transpose(1, 2)
                rotations = torch.linalg.matrix_exp(amounts.view(1, -1, 1, 1) * skew[:, None])    # (kinds, amounts, dim, dim)
                vectors = torch.einsum("gtij,gj->gti", rotations, F.normalize(base, dim=-1))
            else:
                vectors = base[:, None] + amounts.view(1, -1, 1) * self.direction.float()[:, None]
            return F.normalize(vectors.reshape(self.size, self.dim), dim=-1)

    def codebook(self, indices: Tensor) -> Tensor:
        return self.table()[indices]

    def index_of(self, kinds: Tensor, amounts: Tensor) -> Tensor:
        """Index of the code with that kind and that amount (the amount must be in the list)."""

        position = (self.amounts[None, :] == amounts[:, None]).float().argmax(dim=1)
        return kinds * len(self.amounts) + position

    def loss_for(self, actions: Tensor, indices: Tensor) -> Tensor:
        """The lab VQ's two loss terms per sample, toward the given codes instead of the nearest ones."""

        with torch.autocast(device_type=actions.device.type, enabled=False):
            quantized = self.table()[indices].view(actions.shape)
            normalized = F.normalize(actions.float(), dim=-1)
            committed = (quantized.detach() - normalized) if self.commitment_grad else (quantized - normalized.detach())
            commitment = self.commitment_cost * committed.square().mean(dim=(1, 2))
            return commitment + (quantized - normalized.detach()).square().mean(dim=(1, 2))

    def forward(self, actions: Tensor, training: bool = True) -> CodeOutput:
        """The lab VQ's forward with the table computed from the structure instead of read from an embedding."""

        with torch.autocast(device_type=actions.device.type, enabled=False):
            flattened = F.normalize(actions.flatten(end_dim=-2).float(), dim=-1)
            embedding = self.table()
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
            quantized = embedding[indices].view(actions.shape)
            normalized = F.normalize(actions.float(), dim=-1)
            committed = (quantized.detach() - normalized) if self.commitment_grad else (quantized - normalized.detach())
            commitment = self.commitment_cost * committed.square().mean(dim=(1, 2))
            codebook = (quantized - normalized.detach()).square().mean(dim=(1, 2))
            straight_through = normalized + (quantized - normalized).detach()
            if training:
                with torch.no_grad():
                    counts = torch.bincount(indices.flatten(), minlength=self.size).float()
                    self.usage.mul_(0.99).add_(counts / counts.sum().clamp(min=1.0), alpha=0.01)
        return CodeOutput(straight_through, commitment + codebook, indices, logits, temperature)


CODES["vq_orbit"] = OrbitVQ
