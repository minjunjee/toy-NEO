"""The lab VQ with restarts of unused codes. Registered as CODES["vq_restart"].

With nearest-code selection a code that is never chosen never moves, so a codebook that starts with every programmer
output next to one code stays there for a long time (observed: one code used for the first 1,800 steps, then a few more one
at a time). The usual remedy in vector-quantised models is to re-initialise codes that are not being used from current
encoder outputs. This part does that and nothing else:

  - it keeps a running share of how often each code is chosen (decay `usage_decay`);
  - every `restart_every` training steps, a code whose running share is below `restart_below` times the uniform share
    1/size is moved onto the programmer output of a random pair of the current batch.

The forward pass, the loss and the selection are the lab VQ's. With `restart_every: 0` it is the lab VQ.
`restart_until` stops the restarts after that many training steps (0 = never stop), so late training is undisturbed.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from components.codes import VQ, CodeOutput
from core.registry import CODES


class RestartVQ(VQ):
    def __init__(self, size: int, dim: int, restart_every: int = 100, restart_below: float = 0.1, usage_decay: float = 0.99,
                 restart_until: int = 0, **kwargs) -> None:
        super().__init__(size, dim, **kwargs)
        self.restart_every, self.restart_below, self.usage_decay, self.restart_until = restart_every, restart_below, usage_decay, restart_until
        self.register_buffer("usage", torch.full((size,), 1.0 / size))
        self.register_buffer("restarts", torch.zeros((), dtype=torch.long))
        self.last_restart_step = -1

    def forward(self, actions: Tensor, training: bool = True) -> CodeOutput:
        out = super().forward(actions, training)
        if training and self.restart_every and out.indices is not None:
            with torch.no_grad():
                counts = torch.bincount(out.indices.flatten(), minlength=self.size).float()
                self.usage.mul_(self.usage_decay).add_(counts / counts.sum().clamp(min=1.0), alpha=1.0 - self.usage_decay)
                due = self.current_step > 0 and self.current_step % self.restart_every == 0 and self.current_step != self.last_restart_step
                if due and (not self.restart_until or self.current_step <= self.restart_until):
                    self.last_restart_step = self.current_step
                    unused = torch.nonzero(self.usage < self.restart_below / self.size).flatten()
                    if len(unused):
                        outputs = F.normalize(actions.detach().flatten(end_dim=-2).float(), dim=-1)
                        chosen = outputs[torch.randint(len(outputs), (len(unused),), device=outputs.device)]
                        self.embedding.weight.data[unused] = chosen.to(self.embedding.weight.dtype)
                        self.usage[unused] = 1.0 / self.size
                        self.restarts += len(unused)
        return out


CODES["vq_restart"] = RestartVQ
