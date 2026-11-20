"""Codec pretraining: reconstruction CE + kl_weight * KL (paper: kl_weight 0.0)."""

from __future__ import annotations

from torch import nn
from torch.nn import functional as F

from methods.base import Method, TrainingStep, cosine_with_warmup, grid_metrics
from core.registry import METHODS


class VAEPretrain(Method):
    def __init__(self, lr: float, weight_decay: float, warmup_ratio: float, min_lr_ratio: float, grad_clip: float, kl_weight: float, total_steps: int) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.grad_clip = grad_clip
        self.kl_weight = kl_weight
        self.total_steps = total_steps

    def lr_at(self, step: int) -> float:
        return cosine_with_warmup(step, self.total_steps, self.warmup_ratio, self.min_lr_ratio)

    def compute_loss(self, model: nn.Module, batch: dict, step: int) -> TrainingStep:
        out = model(batch, step)
        recon = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), out.targets.reshape(-1).long())
        kl = out.posterior.kl_to_standard_normal() if out.posterior is not None else recon.new_zeros(())
        total = recon + self.kl_weight * kl
        return TrainingStep(total, {"loss": total, "recon": recon, "kl": kl, **grid_metrics(out.logits, out.targets)})


METHODS["vae_pretrain"] = VAEPretrain
