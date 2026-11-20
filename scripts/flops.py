#!/usr/bin/env python3
"""Floating-point operations of one training step of a run config, counted with PyTorch's FlopCounterMode.

The count is of the forward pass on one training batch; a training step is taken as three forward passes (forward plus a
backward pass of about twice the forward cost), the usual convention. Only the trained parts need a backward pass, but the
frozen state encoder and decoder are small next to the programmer and executor, so the same factor is used throughout.
Evaluation is not counted. Counted on CPU with a small batch and scaled linearly to the config's batch size.

    python scripts/flops.py configs/babytheorist/<run>.yaml [...]      # prints and writes runs/flops.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from core.config import load_config  # noqa: E402
from core.registry import METHODS, build, import_all  # noqa: E402

BACKWARD_FACTOR = 3.0


def step_flops(path: Path, probe: int = 4) -> dict:
    config = load_config(path)
    method = build(METHODS, config["method"])
    model = method.build_model(config["model"], init_from=None)
    model.train()
    model_config = config["model"]
    grid = model_config["encoder"].get("grid_size") or model_config["decoder"]["spatial"][0]
    colors = model_config["encoder"].get("num_colors") or model_config["encoder"]["vocab"]
    batch = {"x": torch.randint(0, colors, (probe, grid, grid)), "y": torch.randint(0, colors, (probe, grid, grid))}
    with FlopCounterMode(display=False) as counter:
        model(batch, 0)
    forward = counter.get_total_flops() / probe * config["train"]["batch_size"]
    return {"forward_flops_per_step": forward, "train_flops_per_step": forward * BACKWARD_FACTOR, "batch_size": config["train"]["batch_size"], "steps": config["train"]["steps"],
            "train_flops_total": forward * BACKWARD_FACTOR * config["train"]["steps"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "flops.json")
    args = parser.parse_args()
    import_all()
    report = json.loads(args.out.read_text()) if args.out.exists() else {}
    for path in args.configs:
        report[path.stem] = step_flops(path)
        r = report[path.stem]
        print(f"{path.stem:46s} {r['train_flops_per_step'] / 1e12:9.3f} TFLOPs per step   {r['train_flops_total'] / 1e15:9.2f} PFLOPs for {r['steps']} steps")
    args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
