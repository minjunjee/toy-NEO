#!/usr/bin/env python3
"""Delete intermediate training checkpoints of finished or stopped runs, keeping what is still needed.

For every run directory matching the given patterns, every ckpt/step-*.pt is deleted except
  - the checkpoint with the highest step,
  - any checkpoint a config under configs/ names (init_from, resume),
  - everything in a run that is still training (no DONE, DONE.stopped or STOPPED.txt file).
Evaluation files, logs and configs are never touched. Without --apply it only prints what it would free.

    python scripts/prune_checkpoints.py "neo_mlp_*" "onestep_*" [--apply]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--runs", type=Path, default=REPO / "runs")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    named = set()
    for config in (REPO / "configs").rglob("*.yaml"):
        named.update(re.findall(r"runs/[\w./-]+?\.pt", config.read_text()))
    for jobs in (args.runs / "queue").glob("jobs*.txt"):
        named.update(re.findall(r"runs/[\w./-]+?\.pt", jobs.read_text()))
    keep_named = {(REPO / n).resolve() for n in named}
    freed = 0
    for pattern in args.patterns:
        for run in sorted(args.runs.glob(pattern)):
            checkpoints = sorted((run / "ckpt").glob("step-*.pt"))
            if not checkpoints:
                continue
            if not any((run / marker).exists() for marker in ("DONE", "DONE.stopped", "STOPPED.txt")):
                print(f"skip (still training?) {run.name}")
                continue
            last = max((c for c in checkpoints if re.fullmatch(r"step-\d+", c.stem)), key=lambda c: int(c.stem.split("-")[1]), default=None)
            drop = [c for c in checkpoints if c != last and c.resolve() not in keep_named]
            size = sum(c.stat().st_size for c in drop)
            freed += size
            if drop:
                print(f"{run.name}: {len(drop)} of {len(checkpoints)} checkpoints, {size / 2**20:.0f} MB")
            if args.apply:
                for c in drop:
                    c.unlink()
    print(f"{'freed' if args.apply else 'would free'} {freed / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
