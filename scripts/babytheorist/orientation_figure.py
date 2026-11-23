#!/usr/bin/env python3
"""One dot per seed, a bar for the seed mean: the four orientation primitives that swap width and height, the seventeen
non-orientation primitives, and the code NMI over the seven orientation primitives, for groups of 24-primitive runs
(their last saved evaluation with the per-primitive table).

    python scripts/babytheorist/orientation_figure.py --out docs/reports/2026-11-23-em/orientation_by_condition.png \
        --group "lab=runs/onestep_s_m24_wide_s42,runs/onestep_s_m24_wide_s43" --group "soft EM=..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orientation_table import rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", required=True, help="label=run,run,...")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate", type=float, default=0.6)
    args = parser.parse_args()
    groups = []
    for spec in args.group:
        label, runs = spec.split("=", 1)
        last = [rows(Path(r))[-1] for r in runs.split(",") if rows(Path(r))]
        groups.append((label, last))
    panels = (("swap", "four that swap width and height", True), ("others", "the other seventeen", False), ("nmi_orientation", "code NMI, seven orientation", False))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    summary = {}
    for ax, (key, title, gate) in zip(axes, panels):
        for i, (label, last) in enumerate(groups):
            values = [r[key] for r in last]
            mean = sum(values) / len(values) if values else 0.0
            ax.bar(i, mean, color="#1a56b0" if "EM" in label or "pair" in label else "#9aa0a6", alpha=0.35, width=0.6)
            ax.scatter([i] * len(values), values, color="#1d1d1f", s=22, zorder=3)
            ax.text(i + 0.33, mean, f"{mean:.2f}", ha="left", va="center", fontsize=9)
            summary.setdefault(label, {})[key] = {"mean": mean, "seeds": values}
        if gate:
            ax.axhline(args.gate, color="#c0392b", lw=1, ls="--")
            ax.text(len(groups) - 0.5, args.gate + 0.02, "gate", color="#c0392b", fontsize=9, ha="right")
        ax.set_xticks(range(len(groups)), [g[0] for g in groups], rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("transfer on single-primitive tasks")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
    for label, values in summary.items():
        print(label, {k: round(v["mean"], 3) for k, v in values.items()})


if __name__ == "__main__":
    main()
