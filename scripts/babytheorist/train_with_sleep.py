#!/usr/bin/env python3
"""Train a run in segments with an offline phase between them: wake for `--sleep-every` steps, sleep, wake again.

Each segment is scripts/train.py resumed from the last checkpoint and stopped at the next sleep step; the learning-rate
schedule, the optimizer state and the order of the data carry over, so without a phase that changes anything this is the
same training as one uninterrupted run. Between segments scripts/babytheorist/sleep_edit_checkpoint.py --in-place edits
the checkpoint (merge duplicate codes, hand each freed code to an unexplained pair). What each phase did is appended to
runs/<run>/sleep_log.jsonl and its compute to runs/<run>/extra_flops.json.

    python scripts/babytheorist/train_with_sleep.py --config configs/babytheorist/<run>.yaml --split lv1 --sleep-every 3600 --first-sleep 7200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--sleep-every", type=int, required=True)
    parser.add_argument("--first-sleep", type=int)
    parser.add_argument("--last-sleep", type=int, help="no phase after this step (default: none in the final tenth of training)")
    parser.add_argument("--output-root", type=Path, default=REPO / "runs")
    parser.add_argument("--dry-run", action="store_true")
    args, sleep_options = parser.parse_known_args()
    total = yaml.safe_load(args.config.read_text())["train"]["steps"]
    first = args.first_sleep or args.sleep_every
    last = args.last_sleep or int(total * 0.9)
    stops = [s for s in range(first, total, args.sleep_every) if s <= last] + [total]
    run = args.output_root / args.config.stem
    done = 0
    for stop in stops:
        checkpoint = run / "ckpt" / f"step-{stop:06d}.pt"
        if not checkpoint.exists():
            command = [sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(args.config), "--output-root", str(args.output_root), "--wandb-mode", "disabled", "--steps", str(stop)]
            if done:
                command += ["--resume", str(run / "ckpt" / f"step-{done:06d}.pt")]
            print("wake ", " ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True, cwd=REPO)
        if stop < total and not (run / "ckpt" / f"step-{stop:06d}.before_sleep.pt").exists():
            command = [sys.executable, str(REPO / "scripts" / "babytheorist" / "sleep_edit_checkpoint.py"), "--ckpt", str(checkpoint), "--in-place", "--split", args.split, *sleep_options]
            print("sleep", " ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True, cwd=REPO)
        done = stop


if __name__ == "__main__":
    main()
