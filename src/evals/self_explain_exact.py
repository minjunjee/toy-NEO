"""Self-explanation protocol: theory from the support pair, applied to its own input."""

from __future__ import annotations

import torch
from torch import nn

from evals.transfer_exact import to_device
from core.registry import EVALS


def self_explain_exact(model: nn.Module, data, cfg: dict, step: int, device: torch.device) -> dict:
    infer = dict(cfg.get("infer", {}))
    solved = total = 0
    pixel_sum = steps_sum = 0.0
    for batch in data.eval_loader(cfg["split"], cfg.get("batch_size", 256), max_samples=cfg.get("max_samples")):
        batch = to_device(batch, device)
        out = model.self_explain(batch["support_x"], batch["support_y"], **infer)
        correct = out["pred"].eq(batch["support_y"])
        solved += int(correct.flatten(1).all(dim=1).sum())
        total += correct.shape[0]
        pixel_sum += float(correct.float().flatten(1).mean(dim=1).sum())
        steps_sum += float(out.get("n_steps", torch.zeros(1)).float().sum())
    return {"solved": solved, "total": total, "exact": solved / max(total, 1), "pixel": pixel_sum / max(total, 1), "mean_steps": steps_sum / max(total, 1)}


EVALS["self_explain_exact"] = self_explain_exact
