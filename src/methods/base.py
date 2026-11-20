"""Method = how a model is trained: model construction, optimizer, schedule, one train step."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from core.registry import MODELS, build


@dataclass
class TrainingStep:
    total: Tensor
    metrics: dict[str, Tensor | float]


def load_init_from(model: nn.Module, path: str | Path, prefixes: tuple[str, ...] | None = None) -> dict:
    """Load matching keys from a checkpoint's model state. Returns the load report."""

    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    if prefixes:
        state = {k: v for k, v in state.items() if k.startswith(prefixes)}
    result = model.load_state_dict(state, strict=False)
    if prefixes:
        missing = [k for k in result.missing_keys if k.startswith(prefixes)]
        if missing:
            raise KeyError(f"init_from is missing keys under {prefixes}: {missing[:5]}")
    return {"loaded": len(state), "missing": result.missing_keys, "unexpected": result.unexpected_keys}


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when training through a wrapper such as DDP."""

    return getattr(model, "module", model)


class Method:
    """Default: single AdamW over trainable params, cosine/warmup handled by ``lr_at``."""

    freeze: tuple[str, ...] = ()
    init_prefixes: tuple[str, ...] | None = None

    def build_model(self, spec: dict, init_from: str | None) -> nn.Module:
        model = build(MODELS, spec)
        if init_from:
            self.init_report = load_init_from(model, init_from, self.init_prefixes)
        for name in self.freeze:
            for parameter in getattr(model, name).parameters():
                parameter.requires_grad = False
        return model

    def set_train_mode(self, model: nn.Module) -> None:
        model.train()
        module = unwrap_model(model)
        for name in self.freeze:
            getattr(module, name).eval()

    def attach(self, model: nn.Module, device: torch.device, precision: str) -> None:
        self.device = device
        self.amp = device.type == "cuda" and precision == "bf16"
        self.optimizer = self.build_optimizer(model)
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def lr_at(self, step: int) -> float:
        raise NotImplementedError

    def compute_loss(self, model: nn.Module, batch: dict, step: int) -> TrainingStep:
        raise NotImplementedError

    def after_step(self, model: nn.Module) -> None:
        pass

    def train_step(self, model: nn.Module, batch: dict, step: int) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            out = self.compute_loss(model, batch, step)
        out.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        multiplier = self.lr_at(step)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * multiplier
        self.optimizer.step()
        self.after_step(model)
        metrics = {k: (float(v.detach()) if isinstance(v, Tensor) else float(v)) for k, v in out.metrics.items()}
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        metrics["grad_norm"] = float(grad_norm)
        return metrics

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])


def cosine_with_warmup(step: int, total_steps: int, warmup_ratio: float, min_ratio: float) -> float:
    """Official learning-to-theorize schedule multiplier."""

    import math

    warmup = int(total_steps * warmup_ratio)
    if step < warmup:
        return step / warmup if warmup else 1.0
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def grid_metrics(logits: Tensor, targets: Tensor) -> dict[str, float]:
    decoded = logits.argmax(dim=-1)
    correct = decoded.eq(targets)
    return {
        "pixel_acc": float(correct.float().mean()),
        "grid_acc": float(correct.flatten(1).all(dim=1).float().mean()),
    }
