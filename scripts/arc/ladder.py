#!/usr/bin/env python3
"""Accuracy on the evaluation tasks against the test-time work per task, for the ways of using a trained run.

From cheapest to dearest: one reading of the demonstration pairs; the code fitted on them; that voted over views; a sleep
on all evaluation tasks together; a sleep of its own for every task. Weights are frozen for the first three. Seconds are
wall-clock per task as measured (the GPU was shared, so they are upper bounds), on a log axis.

    python scripts/arc/ladder.py --base arc16_programmer_s0="6M, 64-dim code" --joint arc16_sleep_replay_s0 --per-task per_task_evaluation_s100 --out docs/reports/2026-11-28-arc/ladder.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def final(run: Path, split: str = "evaluation") -> dict:
    return json.loads(sorted((run / "eval").glob(f"final_{split}_step*.json"))[-1].read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", nargs="+", required=True, help="trained run, optionally =label")
    parser.add_argument("--joint", nargs="*", default=[], help="for each base, its joint sleep run (or - for none)")
    parser.add_argument("--per-task", nargs="*", default=[], help="for each base, the name of its per-task sleep result file (or - for none)")
    parser.add_argument("--root", type=Path, default=REPO / "runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    print("| model | one reading | code fitted 100 steps | fitted, two guesses voted over 8 views | joint sleep, two voted guesses | per-task sleep, two voted guesses |")
    print("|---|---|---|---|---|---|")
    for index, item in enumerate(args.base):
        name, label = item.split("=")[0], item.split("=")[-1]
        frozen = final(args.root / name)
        views = frozen["config"].get("votes", 1)
        points = [(frozen["seconds_per_task_zero"], frozen["zero"], "one reading"),
                  (frozen["seconds_per_task_fit"], frozen["fit100"], "code fitted"),
                  (frozen["seconds_per_task_fit"] * views, frozen.get("fit100_vote2", frozen["fit100"]), "fitted and voted")]
        row = [label] + [f"{100 * p[1]:.1f}" for p in points]
        joint = args.joint[index] if index < len(args.joint) else "-"
        if joint != "-":
            slept = final(args.root / joint)
            lines = [json.loads(line) for line in (args.root / joint / "train.jsonl").read_text().splitlines() if '"loss"' in line]
            seconds = lines[-1]["elapsed"] / slept["tasks"] + slept["seconds_per_task_fit"] * slept["config"].get("votes", 1)
            points.append((seconds, slept["fit100_vote2"], "joint sleep"))
            row.append(f"{100 * slept['fit100_vote2']:.1f}")
        else:
            row.append("")
        per_task = args.per_task[index] if index < len(args.per_task) else "-"
        if per_task != "-":
            own = json.loads((args.root / name / f"{per_task}.json").read_text())
            points.append((own["seconds_per_task"], own["vote2"], "sleep per task"))
            row.append(f"{100 * own['vote2']:.1f}")
        else:
            row.append("")
        ax.plot([p[0] for p in points], [100 * p[1] for p in points], marker="o", color=f"C{index}", label=label)
        for seconds, value, text in points:
            ax.annotate(f"{text}\n{100 * value:.1f}%", (seconds, 100 * value), textcoords="offset points", xytext=(6, -4 if index else 8), fontsize=8, color=f"C{index}")
        print("| " + " | ".join(row) + " |")
    ax.set_xscale("log")
    ax.set_xlabel("test-time seconds per task (log scale, shared GPU)")
    ax.set_ylabel("evaluation tasks' tests solved, %")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(args.title or "16x16 proxy: 210 evaluation tasks never trained on before test time")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(args.out)


if __name__ == "__main__":
    main()
