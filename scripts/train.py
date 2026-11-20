#!/usr/bin/env python3
"""Train one run: python scripts/train.py --config configs/runs/<run>.yaml"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from core.config import load_config  # noqa: E402
from training.trainer import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO / "runs")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--steps", type=int, help="override train.steps (smoke runs only)")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.wandb_mode:
        config["train"]["wandb_mode"] = args.wandb_mode
    if args.steps:
        config["train"]["steps"] = args.steps
    output_dir = args.output_root / args.config.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_bytes(args.config.read_bytes())
    train(config, output_dir, resume=args.resume)


if __name__ == "__main__":
    main()
