"""The programmer learns to plan toward far targets from programs it dreams with its own executor. Registered as
METHODS["neo_dream"].

Why. A model trained on one program step composes at test time by naming one code, executing it, and naming the next
from the new grid. Executing the hidden program written in each primitive's code reaches 92% to 97% of eight-step
queries with the same executor, yet the programmer's own choices reach 11% (greedy) to 42% (beam search of width 16):
it has never seen a target several steps away, and its first choice toward such a target is often wrong.

What. Every training step, starting from training input grids, the model runs a dreamed program: a length L drawn
uniformly from 1..`max_dream`, each code drawn in proportion to how often the programmer itself names it on the batch's
real pairs (a moving average with decay `usage_decay`; codes it does not use are not dreamed). The frozen executor runs
the program, the grid decoded and re-encoded after every step. The last grid is the target. At every position t the
programmer sees the grid after t steps and the target, and is trained to name the code executed next: cross-entropy over
`scale` times its cosine similarity to every code. Positions are dropped when a step leaves the grid unchanged or the
program returns to a grid it has already passed (then the next code is not the way to the target). `examples` positions
are drawn from the kept ones every step, so the programmer always sees the same number. Only the programmer
learns; the state encoder and decoder, the executor and the codebook are frozen.

No label is used: no hidden program, no depth. The real pairs are used only for their input grids and for which codes
the programmer names on them.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from core.registry import METHODS
from methods.base import TrainingStep, unwrap_model
from methods.neo import NEOMethod


class NEODreamMethod(NEOMethod):
    init_prefixes = None

    def __init__(self, *args, max_dream: int = 8, scale: float = 10.0, usage_decay: float = 0.99, min_share: float = 0.25,
                 examples: int = 512, freeze: list[str] | None = None, **kwargs) -> None:
        super().__init__(*args, freeze=freeze if freeze is not None else ["encoder", "decoder", "executor", "code"], **kwargs)
        self.max_dream, self.scale, self.usage_decay, self.min_share, self.examples = max_dream, scale, usage_decay, min_share, examples
        self.usage: torch.Tensor | None = None

    def compute_loss(self, model: nn.Module, batch: dict, step: int) -> TrainingStep:
        module = unwrap_model(model)
        x, y = batch["x"], batch["y"]
        size = module.code.size
        table = module.code.codebook(torch.arange(size, device=x.device)).float()
        with torch.no_grad():
            named = (F.normalize(module.programmer(module.encoder(x), module.encoder(y)).flatten(1).float(), dim=-1) @ table.T).argmax(dim=1)
            share = torch.bincount(named, minlength=size).float() / x.shape[0]
            self.usage = share if self.usage is None else self.usage_decay * self.usage + (1 - self.usage_decay) * share
            codes_used = torch.nonzero(self.usage >= self.min_share / size).flatten()
        # the dreamed program, run by the frozen executor; every kept position is one training example
        lengths = torch.randint(1, self.max_dream + 1, (x.shape[0],), device=x.device)
        with torch.no_grad():
            weights = self.usage[codes_used] / self.usage[codes_used].sum()
            programs = codes_used[torch.multinomial(weights, x.shape[0] * self.max_dream, replacement=True)].view(x.shape[0], self.max_dream)
            grids = [x]
            for t in range(self.max_dream):
                out = module.decoder(module.executor(module.encoder(grids[-1]), table[programs[:, t]][:, None])).argmax(dim=-1)
                grids.append(torch.where((t < lengths).view(-1, 1, 1), out.to(x.dtype), grids[-1]))
            target = grids[-1]
            rows, states, labels = [], [], []
            for t in range(self.max_dream):
                moved = grids[t + 1].ne(grids[t]).flatten(1).any(dim=1)
                fresh = torch.ones_like(moved)
                for later in range(t + 1, self.max_dream + 1):
                    fresh &= ~(grids[later].eq(grids[t]).flatten(1).all(dim=1) & (later <= lengths))
                keep = (t < lengths) & moved & fresh & target.ne(grids[t]).flatten(1).any(dim=1)
                index = torch.nonzero(keep).flatten()
                rows.append(index)
                states.append(grids[t][index])
                labels.append(programs[index, t])
            rows, states, labels = torch.cat(rows), torch.cat(states), torch.cat(labels)
            kept = rows.numel()
            if kept and self.examples:  # a fixed number of positions per step, drawn from the kept ones, so shapes never change
                pick = torch.randperm(kept, device=x.device)[: self.examples] if kept >= self.examples else torch.randint(kept, (self.examples,), device=x.device)
                rows, states, labels = rows[pick], states[pick], labels[pick]
        if rows.numel() == 0:
            zero = sum(p.sum() for p in module.programmer.parameters()) * 0.0
            return TrainingStep(zero, {"loss": zero, "dream_examples": 0.0})
        action = module.programmer(module.encoder(states), module.encoder(target[rows]))
        logits = self.scale * (F.normalize(action.flatten(1).float(), dim=-1) @ table.T)
        loss = F.cross_entropy(logits, labels)
        metrics = {"loss": loss, "dream_examples": float(kept), "dream_codes": float(codes_used.numel()),
                   "dream_agree": float((logits.argmax(dim=1) == labels).float().mean()), "mean_length": float(lengths.float().mean()), "code_utilization": codes_used.numel() / size}
        return TrainingStep(loss, metrics)


METHODS["neo_dream"] = NEODreamMethod
