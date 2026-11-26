#!/usr/bin/env python3
"""Adaptation to a sibling world: a model of one world continuing on another, against a model learning the other from
nothing, from their light codebook records (evals/codebook_snapshot.py; split lv1 is the new world, split usual the old).

The value at each record is the share of the probe's one-primitive tasks (first 1,024 of the eval file) whose query the
programmer's chosen code gets right. The adapted run's starting point (step 0) is measured here with the same probe on the
checkpoint it started from. The x-axis is training steps; each step sees 128 tasks of two pairs.

    python scripts/babytheorist/sibling_curves.py --out docs/reports/2026-11-27-sibling/large \
        --pair runs/adapt_m24_large_s43 runs/scratch_m24_large_s43 [--pair ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from core.registry import DATASETS, EVALS, build  # noqa: E402
from training.evaluate import load_run  # noqa: E402


def records(run: Path, name: str) -> list[tuple[int, float]]:
    rows = [json.loads(Path(p).read_text()) for p in glob.glob(str(run / "eval" / f"{name}_step*.json"))]
    return sorted((r["step"], r["transfer"]) for r in rows)


def start_point(run: Path, device) -> dict[str, float]:
    config = json.loads((run / "resolved.json").read_text()) if (run / "resolved.json").exists() else None
    source = Path(config["train"]["init_from"])
    model, _, _, _ = load_run(source.parent.parent, source, device)
    data = build(DATASETS, config["data"])
    return {split: EVALS["codebook_snapshot"](model, data, {"split": split, "max_tasks": 1024, "max_samples": 4000}, 0, device)["transfer"] for split in ("lv1", "usual")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=2, action="append", type=Path, required=True, metavar=("ADAPTED", "SCRATCH"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig, axes = plt.subplots(1, len(args.pair), figsize=(5 * len(args.pair), 3.8), squeeze=False, sharey=True)
    summary = {}
    for k, (adapted, scratch) in enumerate(args.pair):
        ax = axes[0, k]
        start = start_point(adapted, device)
        new = [(0, start["lv1"])] + records(adapted, "snapshot_lv1")
        old = [(0, start["usual"])] + records(adapted, "snapshot_usual_usual")
        fresh = records(scratch, "snapshot_lv1")
        ax.plot(*zip(*new), color="#2a78d6", lw=1.6, label="adapted: new world")
        ax.plot(*zip(*old), color="#2a78d6", lw=1, ls=":", label="adapted: old world")
        if fresh:
            ax.plot(*zip(*fresh), color="#eb6834", lw=1.6, ls="--", label="from nothing: new world")
        ax.set_xlabel("training step (128 tasks of two pairs each)", fontsize=8)
        ax.set_title(adapted.name, fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, max(s for s, _ in new + fresh) if fresh else None)
        if k == 0:
            ax.set_ylabel("one-primitive query right", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        summary[adapted.name] = {"start": start, "adapted_new": new, "adapted_old": old, "scratch_new": fresh}
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=160)
    Path(f"{args.out}.json").write_text(json.dumps(summary) + "\n")
    for name, s in summary.items():
        print(name, "start", {k: round(v, 3) for k, v in s["start"].items()}, "new", [(a, round(b, 2)) for a, b in s["adapted_new"][:8]],
              "old", [(a, round(b, 2)) for a, b in s["adapted_old"][:6]], "scratch", [(a, round(b, 2)) for a, b in s["scratch_new"][:8]])


if __name__ == "__main__":
    main()
