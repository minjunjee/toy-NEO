"""Transfer protocol: theory from the support pair, applied to the query input, exact match against the answer."""

from __future__ import annotations

import ast

import torch
from torch import nn

from core.registry import EVALS

ACTION_LABELS = {
    "move_down": "D", "move_left": "L", "move_right": "R", "move_up": "U",
    "rot90": "r", "flip_h": "h", "flip_v": "v", "next_color": "c",
}


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def program_label(transitions: str) -> str:
    """Compact program label from the paper HDF5 metadata, e.g. "[['move_up', 'rot90'], ...]" -> "Ur"."""

    parsed = ast.literal_eval(transitions)
    if not isinstance(parsed, list) or len(parsed) != 2 or not isinstance(parsed[0], list) or not all(isinstance(a, str) for a in parsed[0]):
        raise ValueError(f"invalid transitions metadata: {transitions!r}")
    return "".join(ACTION_LABELS.get(action, action) for action in parsed[0])


def transfer_exact(model: nn.Module, data, cfg: dict, step: int, device: torch.device) -> dict:
    infer = dict(cfg.get("infer", {}))
    solved = total = 0
    pixel_sum = extracted_sum = applied_sum = 0.0
    by_program: dict[str, list[int]] = {}
    for batch in data.eval_loader(cfg["split"], cfg.get("batch_size", 256), max_samples=cfg.get("max_samples")):
        batch = to_device(batch, device)
        out = model.infer(batch["support_x"], batch["support_y"], batch["query_x"], batch["query_y"], **infer)
        correct = out["pred"].eq(batch["query_y"])
        exact = correct.flatten(1).all(dim=1)
        solved += int(exact.sum())
        total += correct.shape[0]
        if cfg.get("group_by_program", False):
            for is_correct, transitions in zip(exact.tolist(), batch["transitions"], strict=True):
                counts = by_program.setdefault(program_label(transitions), [0, 0])
                counts[0] += int(is_correct)
                counts[1] += 1
        pixel_sum += float(correct.float().flatten(1).mean(dim=1).sum())
        extracted_sum += float(out.get("n_extracted", torch.zeros(1)).float().sum())
        applied_sum += float(out.get("n_applied", torch.zeros(1)).float().sum())
    return {
        "solved": solved, "total": total, "exact": solved / max(total, 1), "pixel": pixel_sum / max(total, 1),
        "mean_extracted": extracted_sum / max(total, 1), "mean_applied": applied_sum / max(total, 1),
        **{f"exact_program_{label}": correct / count for label, (correct, count) in sorted(by_program.items())},
    }


EVALS["transfer_exact"] = transfer_exact
