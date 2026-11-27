#!/usr/bin/env python3
"""Evaluate a trained ARC run with frozen weights, with any test-time budget.

    python scripts/arc/evaluate.py --run runs/arc16_programmer_s0 --split evaluation --fit-steps 300 --fit-report 10 30 100 --votes 16 --out runs/arc16_programmer_s0/frozen_evaluation.json

The evaluation is EVALS["arc_tasks"] (src/evals/arc.py); `--detail` adds the per-task results. `--file` evaluates a task
file that is not in the run's config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from core.registry import DATASETS, EVALS, MODELS, build, import_all  # noqa: E402


def load_run(run: Path, device: torch.device, checkpoint: Path | None = None):
    import_all()
    path = checkpoint or sorted((run / "ckpt").glob("step-*.pt"))[-1]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = saved["config"]
    model = build(MODELS, config["model"])
    model.load_state_dict(saved["model"])
    return model.to(device).eval(), config, int(saved["step"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--file", help="a task file to evaluate instead of the split's")
    parser.add_argument("--fit-steps", type=int, default=0)
    parser.add_argument("--fit-lr", type=float, default=0.05)
    parser.add_argument("--fit-report", type=int, nargs="*", default=[])
    parser.add_argument("--votes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, step = load_run(args.run, device, args.checkpoint)
    block = dict(config["data"])
    if args.file:
        block["evals"] = {**block.get("evals", {}), args.split: args.file}
    data = build(DATASETS, block)
    cfg = {"split": args.split, "batch_size": args.batch_size, "max_tasks": args.max_tasks, "fit_steps": args.fit_steps, "fit_lr": args.fit_lr,
           "fit_report": args.fit_report, "votes": args.votes, "table": args.table, "detail": args.detail}
    result = EVALS["arc_tasks"](model, data, cfg, step, device)
    result = {"run": args.run.name, "step": step, "config": cfg, **result}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "detail"}, indent=1))


if __name__ == "__main__":
    main()
