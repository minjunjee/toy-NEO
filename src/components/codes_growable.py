"""A codebook that can grow. Registered as CODES["vq_growable"].

The lab's VQ with one addition: a row can be inactive. An inactive row is never chosen, so the model starts with
`initial` codes out of `size` rows and an offline phase can switch a row on (and give it a vector) or off. With every
row active this is the lab's VQ, output for output.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from components.codes import VQ, CodeOutput
from core.registry import CODES


class GrowableVQ(VQ):
    def __init__(self, size: int, dim: int, initial: int | None = None, **kwargs) -> None:
        super().__init__(size, dim, **kwargs)
        initial = size if initial is None else initial
        if not 1 <= initial <= size:
            raise ValueError("initial must be between 1 and size")
        self.register_buffer("active", torch.arange(size) < initial)

    def forward(self, actions: Tensor, training: bool = True) -> CodeOutput:
        with torch.autocast(device_type=actions.device.type, enabled=False):
            flattened = F.normalize(actions.flatten(end_dim=-2).float(), dim=-1)
            embedding = F.normalize(self.embedding.weight, dim=-1)
            distances = flattened.square().sum(dim=1, keepdim=True) + embedding.square().sum(dim=1) - 2 * flattened @ embedding.T
            distances = distances.masked_fill(~self.active[None], float("inf"))  # the only change from the lab's VQ
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
            committed = (quantized.detach() - normalized) if self.commitment_grad else (quantized - normalized.detach())
            commitment = self.commitment_cost * committed.square().mean(dim=(1, 2))
            codebook = (quantized - normalized.detach()).square().mean(dim=(1, 2))
            straight_through = normalized + (quantized - normalized).detach()
        return CodeOutput(straight_through, commitment + codebook, indices, logits, temperature)


CODES["vq_growable"] = GrowableVQ
