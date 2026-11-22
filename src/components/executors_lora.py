"""The lab executor plus a low-rank change per code whose up-projection is shared. Registered as EXECUTORS["lora_code"].

Between the lab executor (one network, the code only modulates it) and one network per code
(components/executors_expert.py) lies this: every linear map W of the lab executor's attention and feed-forward layers
becomes, for code k,

    W_k = W + scale * B A_k        B: (d_out, rank), shared by all codes;  A_k: (rank, d_in), the code's own

so each code changes the shared network inside a subspace that all codes share (B), by its own amounts (A_k). W, B and
every A_k are trained by gradient. A training pair moves W, B and the A of its own code; the A of every other code stays.
W and B still carry interference between codes; `shared_speed` < 1 makes B, and with `slow_base` also W, move that much
slower under Adam (they are stored divided by `shared_speed` and multiplied back in the forward pass), which is the rule
that shared structure changes slowly.

The code's index is found as in the per-code executor: the codebook row the code vector equals (models/neo_expert.py
hands the executor the codebook).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from components.blocks.adaln import AdaLNZeroStack
from core.registry import EXECUTORS


class CodeLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, size: int, rank: int, holder: dict, shared_speed: float, slow_base: bool) -> None:
        super().__init__()
        self.base, self.holder, self.speed, self.slow_base = base, holder, shared_speed, slow_base
        if slow_base and shared_speed != 1.0:
            with torch.no_grad():
                base.weight.div_(shared_speed)
                if base.bias is not None:
                    base.bias.div_(shared_speed)
        self.up = nn.Parameter(torch.zeros(base.out_features, rank))                                     # B, shared, starts at zero
        self.down = nn.Parameter(torch.randn(size, rank, base.in_features) / math.sqrt(base.in_features))  # A_k, one per code
        self.scale = 1.0 / rank

    def forward(self, x: Tensor) -> Tensor:
        if self.slow_base and self.speed != 1.0:
            y = nn.functional.linear(x, self.base.weight * self.speed, None if self.base.bias is None else self.base.bias * self.speed)
        else:
            y = self.base(x)
        index = self.holder["index"]
        down = self.down[index].to(x.dtype)                                    # (batch, rank, d_in)
        low = torch.einsum("bld,brd->blr", x, down)
        return y + self.scale * low @ (self.up * self.speed).to(x.dtype).T


class LoRACodeExecutor(nn.Module):
    def __init__(self, state_dim: int, cond_dim: int, size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float = 0.0,
                 grid_dim: int | None = None, rank: int = 8, shared_speed: float = 1.0, slow_base: bool = False) -> None:
        super().__init__()
        self.size = size
        self.stack = AdaLNZeroStack(state_dim, cond_dim, d_model, num_layers, num_heads, d_ff, dropout, grid_dim)
        self.holder = {"index": None}
        for block in self.stack.blocks:
            block.attn.qkv_proj = CodeLoRALinear(block.attn.qkv_proj, size, rank, self.holder, shared_speed, slow_base)
            block.attn.out_proj = CodeLoRALinear(block.attn.out_proj, size, rank, self.holder, shared_speed, slow_base)
            block.ffn[0] = CodeLoRALinear(block.ffn[0], size, rank, self.holder, shared_speed, slow_base)
            block.ffn[3] = CodeLoRALinear(block.ffn[3], size, rank, self.holder, shared_speed, slow_base)
        self.table = None

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        if self.table is None:
            raise RuntimeError("lora_code executor needs the codebook: build it inside models/neo_expert.py")
        with torch.autocast(device_type=action.device.type, enabled=False):
            self.holder["index"] = (action.detach().float().flatten(1) @ self.table().detach().float().T).argmax(dim=1)
        return self.stack(state, action)


EXECUTORS["lora_code"] = LoRACodeExecutor
