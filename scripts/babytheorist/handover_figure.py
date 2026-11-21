#!/usr/bin/env python3
"""Self-explanation and transfer of finished runs under the two rules for what a program step hands to the next one.

Reads each run's full observation (the rule it was trained with) and the re-evaluation written by observe_rollout.py with
--rollout hard (only the decoded grid is handed over), and draws them side by side.

    python scripts/babytheorist/handover_figure.py --hard docs/reports/<name>/observe_rollout_hard_36k.json --out docs/reports/<name>/handover_rules.png \
        "16 codes=runs/<run>" "8 codes=runs/<run>" ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SOFT, HARD = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="label=run directory")
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    hard = {entry["run"]: entry for entry in json.loads(args.hard.read_text())}
    labels, soft_rows, hard_rows = [], [], []
    for item in args.runs:
        label, _, directory = item.partition("=")
        run = Path(directory)
        full = json.loads(sorted((run / "eval").glob("observe_full_*_step*.json"))[-1].read_text())
        labels.append(label.replace("\\n", "\n"))
        soft_rows.append(full)
        hard_rows.append(hard[run.name])
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": INK,
                         "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "legend.frameon": False})
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), sharey=True)
    width = 0.36
    for ax, key, title in zip(axes, ("explain", "transfer_strict"), ("support pair reproduced (self-explanation)", "transferred to the query, answer not shown")):
        for offset, rows, color, name in ((-width / 2 - 0.01, soft_rows, SOFT, "next step gets the decoded cell probabilities (as trained)"), (width / 2 + 0.01, hard_rows, HARD, "next step gets only the decoded grid")):
            xs = [i + offset for i in range(len(labels))]
            values = [row[key] for row in rows]
            ax.bar(xs, values, width=width, color=color, label=name)
            for x, value in zip(xs, values):
                ax.text(x, value + 0.015, f"{value:.0%}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("share of eval tasks")
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
