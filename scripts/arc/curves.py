#!/usr/bin/env python3
"""Learning curves and the final frozen evaluation of ARC runs.

One row of panels: exact grids on the training batches, then for each evaluation split the share of tasks' tests solved
from one reading of the demonstration pairs, all against training steps. Under it, for each split, the final evaluation
as a curve over the ways of naming the code (one reading, code fitted for 10, 30, 100 steps, and the vote over views).
Also prints the final numbers as a table.

    python scripts/arc/curves.py --runs arc16_programmer_s0=programmer arc16_table_s0=table --out docs/reports/2026-11-28-arc/proxy.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SPLITS = {"heldout": "training tasks, unseen examples (re-ARC)", "training": "training tasks, original ARC examples", "evaluation": "evaluation tasks (never trained on)"}


def read(run: Path) -> tuple[list[dict], list[dict]]:
    train, evals = [], []
    for line in (run / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        (train if "loss" in row else evals).append(row)
    return train, evals


def smooth(values: list[float], k: int = 10) -> list[float]:
    out = []
    for i in range(len(values)):
        window = values[max(0, i - k + 1): i + 1]
        out.append(sum(window) / len(window))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run name, optionally =label")
    parser.add_argument("--root", type=Path, default=REPO / "runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--train-key", default="grid_acc_programmer")
    args = parser.parse_args()
    runs = [(r.split("=")[0], r.split("=")[-1]) for r in args.runs]
    fig, axes = plt.subplots(2, 4, figsize=(21, 9))
    table = []
    for index, (name, label) in enumerate(runs):
        color = f"C{index}"
        train, evals = read(args.root / name)
        key = args.train_key if any(args.train_key in row for row in train) else "grid_acc"
        rows = [row for row in train if key in row]
        axes[0, 0].plot([row["step"] for row in rows], smooth([row[key] for row in rows]), color=color, label=label)
        for column, split in enumerate(SPLITS, start=1):
            points = [(row["step"], row[f"during/{split}/zero"]) for row in evals if f"during/{split}/zero" in row]
            axes[0, column].plot([p[0] for p in points], [100 * p[1] for p in points], color=color, marker="o", ms=3, label=label)
        final = next((row for row in reversed(evals) if any(k.startswith("final/") for k in row)), None)
        if final is None:
            continue
        line = {"run": label}
        for column, split in enumerate(SPLITS, start=1):
            get = lambda k: final.get(f"final/{split}/{k}")  # noqa: E731
            fits = sorted(int(k.split("/")[-1][3:]) for k in final if k.startswith(f"final/{split}/fit") and k.split("/")[-1][3:].isdigit())
            ways = ["zero"] + [f"fit{k}" for k in fits]
            xs = list(range(len(ways)))
            axes[1, column].plot(xs, [100 * get(w) for w in ways], color=color, marker="o", label=label)
            if get("zero_vote2") is not None:
                axes[1, column].plot(xs, [100 * get(f"{w}_vote2") for w in ways], color=color, marker="s", ls="--")
            axes[1, column].set_xticks(xs, ["one reading"] + [f"fit {k} steps" for k in fits])
            line[split] = " / ".join(f"{100 * get(w):.1f}" for w in ways) + (f" | vote2 {100 * get(ways[-1] + '_vote2'):.1f}" if get("zero_vote2") is not None else "")
            line[f"{split} verify"] = f"{100 * get('verify_tasks'):.1f}"
            if get("table") is not None:
                line[f"{split} table"] = f"{100 * get('table'):.1f}"
            line[f"{split} s/task"] = f"{get('seconds_per_task_zero'):.4f} / {get('seconds_per_task_fit') or 0:.3f}"
        table.append(line)
    axes[0, 0].set_title("exact grids on training batches (smoothed)")
    axes[0, 0].set_ylabel("share")
    for column, (split, text) in enumerate(SPLITS.items(), start=1):
        axes[0, column].set_title(f"{text}\nsolved from one reading, %", fontsize=10)
        axes[1, column].set_title(f"{text}\nfinal: solid first view, dashed two guesses voted over views, %", fontsize=9)
    for ax in axes[0]:
        ax.set_xlabel("training steps")
        ax.grid(alpha=0.3)
    for ax in axes[1, 1:]:
        ax.set_xlabel("how the code is named at test time (weights frozen)")
        ax.grid(alpha=0.3)
    axes[1, 0].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 0].legend(handles, labels, loc="center", fontsize=11, frameon=False)
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(args.out)
    if table:
        keys = list(dict.fromkeys(k for line in table for k in line))
        print("| " + " | ".join(keys) + " |")
        print("|" + "---|" * len(keys))
        for line in table:
            print("| " + " | ".join(str(line.get(k, "")) for k in keys) + " |")


if __name__ == "__main__":
    main()
