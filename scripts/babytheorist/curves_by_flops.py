#!/usr/bin/env python3
"""Learning curves against training compute instead of training steps.

x is the floating-point operations spent on training so far: steps times the per-step count from scripts/flops.py, plus
whatever a method reports as `extra_flops` in its training log (compute spent outside the training step, such as an
offline phase; cumulative). y is self-explanation and strict transfer on the light evaluation logged during training.

    python scripts/babytheorist/curves_by_flops.py --split lv1 --out docs/reports/<name>/curves_by_flops.png runs/<run> [...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"


def curve(run: Path, split: str, flops: dict) -> dict:
    per_step = flops[run.name]["train_flops_per_step"]
    rows = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
    extra, points = 0.0, []
    offline = json.loads((run / "extra_flops.json").read_text()) if (run / "extra_flops.json").exists() else None  # an offline phase applied to a checkpoint
    for row in rows:
        extra = row.get("extra_flops", extra)
        if f"observe/{split}/explain" in row:
            once = offline["flops"] if offline and row["step"] >= offline["at_step"] else 0.0
            points.append({"step": row["step"], "flops": row["step"] * per_step + extra + once, "explain": row[f"observe/{split}/explain"],
                           "transfer": row[f"observe/{split}/transfer_strict"], "transfer_depth1": row[f"observe/{split}/transfer_strict_depth1"]})
    return {"run": run.name, "points": points}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--flops", type=Path, default=REPO / "runs" / "flops.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    flops = json.loads(args.flops.read_text())
    curves = [curve(run, args.split, flops) for run in args.runs]
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "legend.frameon": False})
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharex=True, sharey=True)
    for index, c in enumerate(curves):
        x = [p["flops"] / 1e15 for p in c["points"]]
        for ax, key in zip(axes, ("explain", "transfer")):
            ax.plot(x, [p[key] for p in c["points"]], color=SERIES[index % len(SERIES)], linewidth=1.8, marker="o", markersize=3, label=c["run"])
    axes[0].set_title("support pair reproduced (self-explanation)")
    axes[1].set_title("transferred to the query, answer not shown")
    for ax in axes:
        ax.set_xlabel("training compute so far (PFLOPs = 1e15 floating-point operations)")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("share of eval tasks")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), bbox_to_anchor=(0.5, 1.0), fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.9 - 0.04 * ((len(labels) - 1) // 3)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150, bbox_inches="tight")
    args.out.with_suffix(".json").write_text(json.dumps(curves, indent=1) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
