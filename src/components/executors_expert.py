"""An executor with one small network per code. Registered as EXECUTORS["per_code"].

The lab executor is one transformer shared by every code: the code vector only conditions it. Training one code then
changes the weights every other code runs on, and with many similar primitives one network has to hold all of them
(docs/reports/2026-11-22-kind-and-amount.md and the scaling page: the seven orientation primitives fall from 91-99% to
12-24% transfer once they are all in one environment, and a new set of primitives pushes out the old ones).

Here every code has its own AdaLN-Zero transformer (the lab executor's blocks, smaller), and a code only ever runs its
own. A training pair moves the weights of the code it was explained with and of nothing else, so codes cannot interfere
through the executor, and a code that is not used is untouched (its weights are memory that is not active). With
`shared_layers` > 0, a transformer shared by all codes runs first (perception common to all codes) and the code's own
network after it.

Which network runs is the index of the code. The executor receives the code vector, as the lab executor does, and finds
the index as the codebook row it equals (the model hands it the codebook; models/neo_expert.py). The code vector still
conditions the network, as in the lab executor.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from components.blocks.adaln import AdaLNZeroStack
from core.registry import EXECUTORS


class PerCodeExecutor(nn.Module):
    def __init__(self, state_dim: int, cond_dim: int, size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float = 0.0,
                 grid_dim: int | None = None, shared_layers: int = 0, shared_d_model: int = 128, shared_d_ff: int = 512) -> None:
        super().__init__()
        self.size = size
        self.experts = nn.ModuleList(AdaLNZeroStack(state_dim, cond_dim, d_model, num_layers, num_heads, d_ff, dropout, grid_dim) for _ in range(size))
        self.shared = AdaLNZeroStack(state_dim, cond_dim, shared_d_model, shared_layers, num_heads, shared_d_ff, dropout, grid_dim) if shared_layers else None
        self.table = None  # a function returning the (size, cond_dim) code vectors, set by the model; not a submodule

    def indices(self, action: Tensor) -> Tensor:
        if self.table is None:
            raise RuntimeError("per_code executor needs the codebook: build it inside models/neo_expert.py")
        with torch.autocast(device_type=action.device.type, enabled=False):
            table = self.table().detach().float()
            return (action.detach().float().flatten(1) @ table.T).argmax(dim=1)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        if self.shared is not None:
            state = self.shared(state, action)
        index = self.indices(action)
        outputs, positions = [], []
        for code in index.unique().tolist():
            rows = torch.nonzero(index == code).flatten()
            outputs.append(self.experts[code](state[rows], action[rows]))
            positions.append(rows)
        order = torch.cat(positions).argsort()
        return torch.cat(outputs)[order]


EXECUTORS["per_code"] = PerCodeExecutor
