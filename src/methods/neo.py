"""NEO objective: recon(selected) + grounding + code(selected) + auto-recon; 3 AdamW groups; code tau step."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from methods.base import Method, TrainingStep, cosine_with_warmup, grid_metrics, unwrap_model
from core.registry import METHODS
from models.neo import select_by_length


CODE_LOSS_SCOPES = ("selected", "prefix")  # code loss at the selected step only, or mean over steps 1..selected


class NEOMethod(Method):
    init_prefixes = ("encoder.", "decoder.")

    def __init__(
        self,
        lr: float,
        programmer_lr_scale: float,
        executor_lr_scale: float,
        weight_decay: float,
        warmup_ratio: float,
        min_lr_ratio: float,
        grad_clip: float,
        grounding_weight: float,
        code_weight: float,
        auto_weight: float,
        total_steps: int,
        freeze: tuple[str, ...] = ("encoder", "decoder"),
        code_loss_scope: str = "selected",
    ) -> None:
        if code_loss_scope not in CODE_LOSS_SCOPES:
            raise ValueError(f"code_loss_scope must be one of {CODE_LOSS_SCOPES}")
        self.lr = lr
        self.programmer_lr_scale = programmer_lr_scale
        self.executor_lr_scale = executor_lr_scale
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.grad_clip = grad_clip
        self.grounding_weight = grounding_weight
        self.code_weight = code_weight
        self.auto_weight = auto_weight
        self.total_steps = total_steps
        self.freeze = tuple(freeze)
        self.code_loss_scope = code_loss_scope

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        programmer = list(model.programmer.parameters())
        executor = list(model.executor.parameters())
        other = [
            p for name, p in model.named_parameters()
            if not name.startswith(("programmer.", "executor.")) and p.requires_grad
        ]
        groups = [
            {"params": programmer, "lr": self.lr * self.programmer_lr_scale, "name": "programmer"},
            {"params": executor, "lr": self.lr * self.executor_lr_scale, "name": "executor"},
        ]
        if other:
            groups.append({"params": other, "lr": self.lr, "name": "other"})
        return torch.optim.AdamW(groups, weight_decay=self.weight_decay)

    def lr_at(self, step: int) -> float:
        return cosine_with_warmup(step, self.total_steps, self.warmup_ratio, self.min_lr_ratio)

    def compute_loss(self, model: nn.Module, batch: dict, step: int) -> TrainingStep:
        out = model(batch, step)
        module = unwrap_model(model)
        logits = select_by_length(out.logits_by_length, out.selected_lengths)
        recon = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), out.targets.reshape(-1).long())
        if self.code_loss_scope == "prefix":
            losses = torch.stack(out.code_loss_by_length, dim=1)
            used = torch.arange(losses.shape[1], device=losses.device)[None] < out.selected_lengths[:, None]
            code = (losses * used).sum() / used.sum().clamp_min(1)
        else:
            code = select_by_length(out.code_loss_by_length, out.selected_lengths).mean()
        auto = F.cross_entropy(out.auto_logits.reshape(-1, out.auto_logits.shape[-1]), out.observations.reshape(-1).long())
        total = recon + self.grounding_weight * out.grounding + self.code_weight * code + self.auto_weight * auto
        metrics = {
            "loss": total, "recon": recon, "grounding": out.grounding, "code": code, "auto_recon": auto,
            "mean_length": out.selected_lengths.float().mean(), "code_tau": out.code_temperature,
            **grid_metrics(logits, out.targets),
        }
        counts = torch.bincount(out.selected_lengths, minlength=module.max_length + 1)[1:]
        for length, count in enumerate(counts.tolist(), start=1):
            metrics[f"length_{length}_frac"] = count / len(out.selected_lengths)
        if out.code_indices_by_length[0] is not None:
            used = torch.cat([i for i in out.code_indices_by_length if i is not None]).unique().numel()
            metrics["code_utilization"] = used / module.code.size
        return TrainingStep(total, metrics)

    def after_step(self, model: nn.Module) -> None:
        module = unwrap_model(model)
        step = getattr(module.code, "step", None)
        if step is not None:
            step()


METHODS["neo"] = NEOMethod
