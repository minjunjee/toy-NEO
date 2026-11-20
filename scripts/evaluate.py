#!/usr/bin/env python3
"""Evaluate a run's checkpoint: python scripts/evaluate.py --run runs/<run> [--ckpt ...] [--eval transfer_exact] [--split id]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from training.evaluate import load_run  # noqa: E402
from training.trainer import run_evals  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--eval", action="append", help="eval type to run (default: every block in the run config)")
    parser.add_argument("--split", action="append", help="restrict to these splits")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data, config, step = load_run(args.run, args.ckpt, device)
    blocks = [dict(b) for b in config["evals"]]
    if args.eval:
        blocks = [b for b in blocks if b["type"] in args.eval]
    if args.split:
        blocks = [b for b in blocks if b["split"] in args.split]
    for block in blocks:
        if args.max_samples:
            block["max_samples"] = args.max_samples
        if args.batch_size:
            block["batch_size"] = args.batch_size
    metrics = run_evals(model, data, blocks, step, device, args.run / "eval")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
