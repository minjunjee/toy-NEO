"""Rebuild a model from a checkpoint (its config travels inside) and run eval protocols."""

from __future__ import annotations

from pathlib import Path

import torch

from core.registry import DATASETS, METHODS, build, import_all


def latest_checkpoint(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "ckpt").glob("step-*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoints under {run_dir / 'ckpt'}")
    return candidates[-1]


def load_run(run_dir: Path, checkpoint: Path | None, device: torch.device):
    import_all()
    checkpoint = checkpoint or latest_checkpoint(run_dir)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    method = build(METHODS, config["method"])
    model = method.build_model(config["model"], init_from=None)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    data = build(DATASETS, config["data"])
    return model, data, config, int(payload["step"])
