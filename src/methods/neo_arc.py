"""Training of MODELS["neo_arc"]. Registered as METHODS["neo_arc"].

The loss is the cell cross-entropy of every pair of the episode, predicted from the code of the other pairs (or from the
table code of the task variant), averaged over the passes of the executor, plus `tie_weight` times the tie between the
programmer's codes and the table code. Pad cells are part of the target; `pad_weight` below 1 makes them count less than
the cells of the grid. The table has its own learning rate scale and no weight decay.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from core.registry import METHODS
from data.arc_tasks import PAD
from methods.base import Method, TrainingStep, cosine_with_warmup


class NEOArcMethod(Method):
    def __init__(self, lr: float, weight_decay: float, warmup_ratio: float, min_lr_ratio: float, grad_clip: float, total_steps: int,
                 tie_weight: float = 1.0, table_lr_scale: float = 1.0, pad_weight: float = 1.0, freeze: list[str] | None = None) -> None:
        self.lr, self.weight_decay, self.warmup_ratio, self.min_lr_ratio = lr, weight_decay, warmup_ratio, min_lr_ratio
        self.grad_clip, self.total_steps, self.tie_weight, self.table_lr_scale, self.pad_weight = grad_clip, total_steps, tie_weight, table_lr_scale, pad_weight
        self.freeze = tuple(freeze or ())

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        table = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("table.")]
        rest = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("table.")]
        groups = [{"params": rest, "lr": self.lr, "weight_decay": self.weight_decay}]
        if table:
            groups.append({"params": table, "lr": self.lr * self.table_lr_scale, "weight_decay": 0.0})
        return torch.optim.AdamW(groups, lr=self.lr, weight_decay=self.weight_decay)

    def lr_at(self, step: int) -> float:
        return cosine_with_warmup(step, self.total_steps, self.warmup_ratio, self.min_lr_ratio)

    def compute_loss(self, model: nn.Module, batch: dict, step: int) -> TrainingStep:
        out = model(batch, step)
        targets, valid = out.targets[out.valid], out.valid
        weight = torch.where(targets.eq(PAD), self.pad_weight, 1.0).flatten(1)
        per_pass = []
        for logits in out.logits_by_loop:
            logits = logits[valid].float()
            cell = F.cross_entropy(logits.flatten(0, 2), targets.flatten(), reduction="none").view(len(targets), -1)
            per_pass.append(((cell * weight).sum(dim=1) / weight.sum(dim=1)).mean())
        cells = sum(per_pass) / len(per_pass)
        total = cells + self.tie_weight * out.tie
        with torch.no_grad():
            from_table = out.from_table[valid]
            right = out.logits_by_loop[-1][valid].argmax(dim=-1).eq(targets)
            exact = right.flatten(1).all(dim=1).float()
            metrics = {"loss": total, "cell_loss": cells, "tie": out.tie, "code_norm": out.code_norm,
                       "cell_acc": right[targets.ne(PAD)].float().mean(), "grid_acc": exact.mean()}
            if from_table.any():
                metrics["grid_acc_table"] = exact[from_table].mean()
            if (~from_table).any():
                metrics["grid_acc_programmer"] = exact[~from_table].mean()
            if len(out.logits_by_loop) > 1:
                metrics["grid_acc_first_pass"] = out.logits_by_loop[0][valid].argmax(dim=-1).eq(targets).flatten(1).all(dim=1).float().mean()
            if "source" in batch:                    # a mixed batch: exact grids of each train file
                source = batch["source"].repeat_interleave(len(out.valid) // len(batch["source"]))[valid]
                for k in source.unique().tolist():
                    metrics[f"grid_acc_file{k}"] = exact[source == k].mean()
        return TrainingStep(total, metrics)


METHODS["neo_arc"] = NEOArcMethod
