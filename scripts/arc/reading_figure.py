#!/usr/bin/env python3
"""What the programmer's reading buys in a per-task sleep: accuracy against sleep steps when the free code of every view
starts from the programmer's reading of the pairs, and when it starts from a random vector of the same length.

Reads <run>/reading_s<steps>_<read|random>.json (scripts/arc/sleep_per_task.py --code free --code-init ...); the
100-step reading result may be given with --extra 100=read=<file>.

    python scripts/arc/reading_figure.py --run runs/arc16_given_views_s0 --extra 100=read=grid_s100_free --out docs/reports/2026-11-28-arc/reading.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--extra", nargs="*", default=[], help="steps=read|random=result name")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    points: dict[str, dict[int, dict]] = {"read": {}, "random": {}}
    for path in args.run.glob("reading_s*_*.json"):
        found = re.fullmatch(r"reading_s(\d+)_(read|random)\.json", path.name)
        if found:
            points[found.group(2)][int(found.group(1))] = json.loads(path.read_text())
    for item in args.extra:
        steps, start, name = item.split("=")
        points[start][int(steps)] = json.loads((args.run / f"{name}.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    labels = {"read": "free code starts from the programmer's reading", "random": "free code starts from a random vector"}
    print("| sleep steps | " + " | ".join(f"{labels[s]}: first view / two voted guesses" for s in labels) + " |")
    print("|---|---|---|")
    for steps in sorted(set(points["read"]) | set(points["random"])):
        cells = [f"{100 * points[s][steps]['first_view']:.1f} / {100 * points[s][steps]['vote2']:.1f}" if steps in points[s] else "" for s in labels]
        print(f"| {steps} | " + " | ".join(cells) + " |")
    for ax, (key, title) in zip(axes, (("first_view", "first view, one guess"), ("vote2", "two guesses voted over 64 views"))):
        for index, start in enumerate(labels):
            steps = sorted(points[start])
            ax.plot(steps, [100 * points[start][k][key] for k in steps], marker="o", color=f"C{index}", label=labels[start])
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks([0, 10, 25, 50, 100, 400], ["0", "10", "25", "50", "100", "400"])
        ax.set_xlabel("steps of the per-task sleep")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("evaluation tasks' tests solved, %")
    axes[0].legend(fontsize=9)
    fig.suptitle("16x16 proxy, 105 evaluation tasks: the same per-task sleep with two starts of the code")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(args.out)


if __name__ == "__main__":
    main()
