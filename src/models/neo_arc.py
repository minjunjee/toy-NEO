"""NEO for ARC tasks: a code read from a task's pairs, executed on another input. Registered as MODELS["neo_arc"].

An ARC task is a few input and output grids that share one rule, and a test input. Here the code is the rule: one vector.
The programmer reads one pair (input state, output state) and writes a code; the codes of a task's pairs are averaged. The
executor turns an input state into an output state under that code, and the state decoder reads the grid off it, pad cells
included, so the size of the output is predicted with its content.

Training (an episode is `pairs` grids of one task variant). Every pair is predicted from the mean of the codes of the
other pairs, never from its own code, so the code has to say what the pairs share and cannot carry one pair's answer.
An episode may be shorter (`mask`): the missing pairs are left out of the means and of the loss.

A code given by the task's name (`tasks` > 0). Each task variant of the training set also owns a code vector in a table.
With probability `table_share` an episode is executed with its table code instead of the programmer's, so the executor
learns the transformations from a code that does not depend on how well the programmer reads yet; and a tie term pulls the
programmer's codes and the table code of the same variant together, so the table lives where the programmer writes. The
tie has two sides: the table code is pulled to the mean of the programmer's codes, and the programmer's codes are pulled
to the table code with weight `tie_programmer` (0: the table only follows, and holds the average code of its variant).
The table is never used on a task outside the training set (task index -1): there the code comes from the pairs alone.

A code of several vectors (`code_tokens` > 1). The programmer writes `code_tokens` vectors and the executor reads them
as separate tokens (parts in components/code_tokens.py). Everywhere else the code is still one flat vector of `code_dim`
numbers: the mean over pairs, the table and the fitting at test time work on it unchanged.

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
    valid: Tensor                    # (B*N,) bool: the pair is there (episodes of tasks with few examples are shorter)


class NEOArc(nn.Module):
    def __init__(self, encoder: dict, decoder: dict, programmer: dict, executor: dict, code_dim: int, tasks: int = 0, variants: int = 1,
                 table_share: float = 0.5, loops: int = 1, table_std: float = 0.5, tie_programmer: float = 1.0, code_tokens: int = 1) -> None:
        super().__init__()
        self.encoder = build(ENCODERS, encoder)
        self.decoder = build(DECODERS, decoder)
        self.programmer = build(PROGRAMMERS, programmer)
        self.executor = build(EXECUTORS, executor)
        self.code_dim, self.tasks, self.variants, self.table_share, self.loops = code_dim, tasks, variants, table_share, loops
        self.tie_programmer, self.code_tokens = tie_programmer, code_tokens
        if code_dim % code_tokens:
            raise ValueError("code_dim must be a multiple of code_tokens")
        self.table = nn.Embedding(tasks * variants, code_dim) if tasks else None
        if self.table is not None:
            nn.init.normal_(self.table.weight, std=table_std * code_dim**-0.5)

    def pair_codes(self, inputs: Tensor, outputs: Tensor) -> Tensor:
        """(M,S,S), (M,S,S) -> (M,D): the code the programmer writes for each pair."""

        return self.programmer(self.encoder(inputs), self.encoder(outputs)).flatten(1).float()

    def run(self, state: Tensor, code: Tensor, loops: int | None = None) -> list[Tensor]:
        """state (M,L,d), code (M,D) -> logits after each pass of the executor."""

        logits, carry = [], None
        for _ in range(loops or self.loops):
            carry = self.executor(state if carry is None else state + carry.detach(), code.view(len(code), self.code_tokens, -1))
            logits.append(self.decoder(carry))
        return logits

    def execute(self, inputs: Tensor, code: Tensor, loops: int | None = None) -> Tensor:
        """(M,S,S), (M,D) -> logits (M,S,S,V) after the last pass."""

        return self.run(self.encoder(inputs), code, loops)[-1]

    def forward(self, batch: dict, step: int = 0) -> NEOArcOutput:
        x, y = batch["inputs"], batch["outputs"]                      # (B,N,S,S)
        b, n = x.shape[:2]
        mask = batch.get("mask")
        there = torch.ones(b, n, 1, device=x.device) if mask is None else mask.float()[..., None]      # pairs the episode really has
        state_x, state_y = self.encoder(x.flatten(0, 1)), self.encoder(y.flatten(0, 1))
        own = self.programmer(state_x, state_y).flatten(1).float().view(b, n, -1) * there
        count = there.sum(dim=1, keepdim=True)
        code = (own.sum(dim=1, keepdim=True) - own) / (count - 1).clamp(min=1)         # the mean of the other pairs' codes
        from_table = torch.zeros(b, dtype=torch.bool, device=x.device)
        tie = torch.zeros((), device=x.device)
        code_norm = code.detach().norm(dim=-1).mean()
        if self.table is not None:
            known = batch["task"] >= 0                                 # tasks that own a code in the table
            given = self.table(batch["task"].clamp(min=0) * self.variants + batch["variant"]).float()       # (B,D)
            if known.any():
                apart = ((code - given[:, None].detach()).pow(2).sum(-1) * there[..., 0]).sum(dim=1) / count[:, 0, 0]
                tie = (self.tie_programmer * apart + (given - (own.sum(dim=1) / count[:, 0]).detach()).pow(2).sum(-1))[known].mean()
            if self.training:
                from_table = (torch.rand(b, device=x.device) < self.table_share) & known
                code = torch.where(from_table[:, None, None], given[:, None].expand_as(code), code)
        logits = self.run(state_x, code.flatten(0, 1))
        return NEOArcOutput(logits, y.flatten(0, 1), from_table.repeat_interleave(n), tie, code_norm, there.flatten().bool())


MODELS["neo_arc"] = NEOArc
