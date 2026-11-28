#!/usr/bin/env python3
"""Learning curves of the 30x30 runs on one step axis: runs branched from a checkpoint are drawn from the step they started at.

Three panels, all solved from one reading of the demonstration pairs with frozen weights: unseen examples of training
tasks (re-ARC), the original ARC examples of training tasks, and evaluation tasks. For a run with given codes the dashed
line is the same examples executed with the table's code.

    python scripts/arc/curves30.py --runs arc30_colors512_s0="vector code" arc30_map_from50k_s0@50000="vector and map" --out docs/reports/2026-11-28-arc/runs30.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PANELS = (("heldout", "training tasks, unseen examples (re-ARC)"), ("training", "training tasks, original ARC examples"), ("evaluation", "evaluation tasks, weights frozen"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run[@start step][=label]")
    parser.add_argument("--root", type=Path, default=REPO / "runs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for index, item in enumerate(args.runs):
        name, label = item.split("=")[0], item.split("=")[-1]
        name, start = (name.split("@")[0], int(name.split("@")[1])) if "@" in name else (name, 0)
        rows = [json.loads(line) for line in (args.root / name / "train.jsonl").read_text().splitlines() if '"loss"' not in line]
        for ax, (split, _) in zip(axes, PANELS):
            points = [(start + row["step"], row[f"during/{split}/zero"]) for row in rows if f"during/{split}/zero" in row]
            ax.plot([p[0] for p in points], [100 * p[1] for p in points], marker="o", ms=3, color=f"C{index}", label=label)
            given = [(start + row["step"], row[f"during/{split}/given"]) for row in rows if f"during/{split}/given" in row]
            if given:
                ax.plot([p[0] for p in given], [100 * p[1] for p in given], marker="s", ms=3, ls="--", color=f"C{index}", label=f"{label}, table code")
        print(label, {split: [round(100 * row[f"during/{split}/zero"], 1) for row in rows if f"during/{split}/zero" in row][-1:] for split, _ in PANELS})
    for ax, (_, title) in zip(axes, PANELS):
        ax.set_title(f"{title}\nsolved from one reading, %", fontsize=10)
        ax.set_xlabel("training steps (64 episodes of 4 pairs each)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("30x30 canvas, 400 re-ARC training tasks: the first run and the two runs branched from its 50,000-step checkpoint")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(args.out)


if __name__ == "__main__":
    main()
