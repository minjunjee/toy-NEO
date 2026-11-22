"""Training methods for this round's code-assignment experiments.

METHODS["neo_em"]: the loss of models/neo_em.py (responsibility-weighted reconstruction plus the programmer's
prediction of the responsibilities) and nothing else; the lab method would add an unweighted reconstruction of the top
code on top. Starts from every part of `train.init_from`, as neo_warm.

METHODS["neo_alternate"]: the lab objective, with the programmer (and the codebook) and the executor taking turns:
for `period` steps only the programmer and the codebook are updated, then for `period` steps only the executor, and so
on. The parts that wait get no update at all (their gradients are dropped before the optimizer step). Starts from every
part of `train.init_from`.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from core.registry import METHODS
from methods.base import TrainingStep, unwrap_model
from methods.neo import NEOMethod


class NEOEMMethod(NEOMethod):
    init_prefixes = None

    def compute_loss(self, model: nn.Module, batch: dict, step: int) -> TrainingStep:
        out = model(batch, step)
        module = unwrap_model(model)
        loss = out.code_loss_by_length[0].mean()
        auto = F.cross_entropy(out.auto_logits.reshape(-1, out.auto_logits.shape[-1]), out.observations.reshape(-1).long())
        total = loss + self.auto_weight * auto
        logits = out.logits_by_length[0]
        decoded = logits.argmax(dim=-1).eq(out.targets)
        metrics = {"loss": total, "pixel_acc": float(decoded.float().mean()), "grid_acc": float(decoded.flatten(1).all(dim=1).float().mean()),
                   "mean_length": 1.0, "code_utilization": out.code_indices_by_length[0].unique().numel() / module.code.size, **module.last}
        return TrainingStep(total, metrics)


class NEOAlternateMethod(NEOMethod):
    init_prefixes = None

    def __init__(self, *args, period: int = 1000, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.period = period

    def train_step(self, model: nn.Module, batch: dict, step: int) -> dict[str, float]:
        module = unwrap_model(model)
        executor_turn = (step // self.period) % 2 == 1
        waiting = list(module.programmer.parameters()) + list(module.code.parameters()) if executor_turn else list(module.executor.parameters())
        for parameter in waiting:
            parameter.requires_grad_(False)
        try:
            metrics = super().train_step(model, batch, step)
        finally:
            for parameter in waiting:
                parameter.requires_grad_(True)
        metrics["executor_turn"] = float(executor_turn)
        return metrics


METHODS["neo_em"] = NEOEMMethod
METHODS["neo_alternate"] = NEOAlternateMethod
