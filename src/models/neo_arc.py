"""NEO for ARC tasks: a code read from a task's pairs, executed on another input. Registered as MODELS["neo_arc"].

An ARC task is a few input and output grids that share one rule, and a test input. Here the code is the rule: one vector.
The programmer reads one pair (input state, output state) and writes a code; the codes of a task's pairs are averaged. The
executor turns an input state into an output state under that code, and the state decoder reads the grid off it, pad cells
included, so the size of the output is predicted with its content.

Training (an episode is `pairs` grids of one task variant). Every pair is predicted from the mean of the codes of the
other pairs, never from its own code, so the code has to say what the pairs share and cannot carry one pair's answer.

A code given by the task's name (`tasks` > 0). Each task variant of the training set also owns a code vector in a table.
With probability `table_share` an episode is executed with its table code instead of the programmer's, so the executor
learns the transformations from a code that does not depend on how well the programmer reads yet; and a tie term pulls the
programmer's codes and the table code of the same variant together, so the table lives where the programmer writes. The
table is never used on a task outside the training set: there the code comes from the pairs alone.

Refinement (`loops` > 1). The executor is applied `loops` times with the same weights and the same code; from the second
time on its input is the input state plus its previous output, detached, and every output is trained to decode to the
answer. One pass is the plain model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from core.registry import DECODERS, ENCODERS, EXECUTORS, MODELS, PROGRAMMERS, build


@dataclass
class NEOArcOutput:
    logits_by_loop: list[Tensor]     # loops x (B*N,S,S,V)
    targets: Tensor                  # (B*N,S,S)
    from_table: Tensor               # (B*N,) bool: executed with the table code
    tie: Tensor                      # scalar, 0 without a table
    code_norm: Tensor                # scalar, mean length of the programmer's leave-one-out codes


class NEOArc(nn.Module):
    def __init__(self, encoder: dict, decoder: dict, programmer: dict, executor: dict, code_dim: int, tasks: int = 0, variants: int = 1,
                 table_share: float = 0.5, loops: int = 1, table_std: float = 0.5) -> None:
        super().__init__()
        self.encoder = build(ENCODERS, encoder)
        self.decoder = build(DECODERS, decoder)
        self.programmer = build(PROGRAMMERS, programmer)
        self.executor = build(EXECUTORS, executor)
        self.code_dim, self.tasks, self.variants, self.table_share, self.loops = code_dim, tasks, variants, table_share, loops
        self.table = nn.Embedding(tasks * variants, code_dim) if tasks else None
        if self.table is not None:
            nn.init.normal_(self.table.weight, std=table_std * code_dim**-0.5)

    def pair_codes(self, inputs: Tensor, outputs: Tensor) -> Tensor:
        """(M,S,S), (M,S,S) -> (M,D): the code the programmer writes for each pair."""

        return self.programmer(self.encoder(inputs), self.encoder(outputs))[:, 0].float()

    def run(self, state: Tensor, code: Tensor, loops: int | None = None) -> list[Tensor]:
        """state (M,L,d), code (M,D) -> logits after each pass of the executor."""

        logits, carry = [], None
        for _ in range(loops or self.loops):
            carry = self.executor(state if carry is None else state + carry.detach(), code[:, None, :])
            logits.append(self.decoder(carry))
        return logits

    def execute(self, inputs: Tensor, code: Tensor, loops: int | None = None) -> Tensor:
        """(M,S,S), (M,D) -> logits (M,S,S,V) after the last pass."""

        return self.run(self.encoder(inputs), code, loops)[-1]

    def forward(self, batch: dict, step: int = 0) -> NEOArcOutput:
        x, y = batch["inputs"], batch["outputs"]                      # (B,N,S,S)
        b, n = x.shape[:2]
        state_x, state_y = self.encoder(x.flatten(0, 1)), self.encoder(y.flatten(0, 1))
        own = self.programmer(state_x, state_y)[:, 0].float().view(b, n, -1)
        code = (own.sum(dim=1, keepdim=True) - own) / (n - 1)         # the mean of the other pairs' codes
        from_table = torch.zeros(b, dtype=torch.bool, device=x.device)
        tie = torch.zeros((), device=x.device)
        code_norm = code.detach().norm(dim=-1).mean()
        if self.table is not None:
            given = self.table(batch["task"] * self.variants + batch["variant"]).float()       # (B,D)
            tie = (code - given[:, None].detach()).pow(2).sum(-1).mean() + (given - own.mean(dim=1).detach()).pow(2).sum(-1).mean()
            if self.training:
                from_table = torch.rand(b, device=x.device) < self.table_share
                code = torch.where(from_table[:, None, None], given[:, None].expand_as(code), code)
        logits = self.run(state_x, code.flatten(0, 1))
        return NEOArcOutput(logits, y.flatten(0, 1), from_table.repeat_interleave(n), tie, code_norm)


MODELS["neo_arc"] = NEOArc
