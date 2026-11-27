#!/usr/bin/env python3
"""Curves of sleeps on the evaluation tasks' demonstration pairs: what is gained on the evaluation tasks' tests and what
is kept on the training tasks, against sleep steps. Step 0 is the run the sleep started from (its final evaluation).

    python scripts/arc/sleep_curves.py --start arc16_programmer_s0 --runs arc16_sleep_replay_s0="half replay" arc16_sleep_alone_s0="no replay" --out docs/reports/2026-11-28-arc/sleep1.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PANELS = (("evaluation", "zero", "evaluation tasks' tests\nsolved from one reading, %"), ("evaluation", "fit30", "evaluation tasks' tests\nsolved after fitting the code 30 steps, %"),
          ("evaluation", "given", "evaluation tasks' tests\nsolved with the task's code in the table, %"),
          ("heldout", "zero", "training tasks, unseen examples\nsolved from one reading, %"), ("training", "zero", "training tasks, original ARC examples\nsolved from one reading, %"))


def rows(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines() if '"loss"' not in line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="the run the sleeps started from")
    parser.add_argument("--runs", nargs="+", required=True, help="sleep run, optionally =label")
    parser.add_argument("--root", type=Path, default=REPO / "runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    before = next(row for row in reversed(rows(args.root / args.start)) if any(k.startswith("final/") for k in row))
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.2 * len(PANELS), 4.6))
    table = []
    for index, item in enumerate(args.runs):
        name, label = item.split("=")[0], item.split("=")[-1]
        evals = rows(args.root / name)
        line = {"run": label}
        for ax, (split, way, _) in zip(axes, PANELS):
            points = [(0, before.get(f"final/{split}/{way}"))] + [(row["step"], row.get(f"during/{split}/{way}")) for row in evals if f"during/{split}/zero" in row]
            points = [(s, v) for s, v in points if v is not None]
            if not points:
                continue
            ax.plot([p[0] for p in points], [100 * p[1] for p in points], marker="o", ms=3, color=f"C{index}", label=label)
            line[f"{split} {way}"] = f"{100 * points[0][1]:.1f} to {100 * points[-1][1]:.1f}"
        final = next((row for row in reversed(evals) if any(k.startswith("final/") for k in row)), None)
        if final is not None:
            get = lambda k: final.get(f"final/evaluation/{k}")  # noqa: E731
            last = max((int(k.split("/")[-1][3:]) for k in final if k.startswith("final/evaluation/fit") and k.split("/")[-1][3:].isdigit()), default=None)
            parts = [f"one reading {100 * get('zero'):.1f}"]
            if last is not None:
                parts.append(f"code fitted {last} steps {100 * get(f'fit{last}'):.1f}")
                if get(f"fit{last}_vote2") is not None:
                    parts.append(f"two voted guesses {100 * get(f'fit{last}_vote2'):.1f}")
            if get("given") is not None:
                parts.append(f"table code {100 * get('given'):.1f}")
                if get("given_vote2") is not None:
                    parts.append(f"table code, two voted guesses {100 * get('given_vote2'):.1f}")
            line["evaluation, final"] = ", ".join(parts)
        table.append(line)
    for ax, (_, _, text) in zip(axes, PANELS):
        ax.set_title(text, fontsize=10)
        ax.set_xlabel("sleep steps")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(args.out)
    keys = list(dict.fromkeys(k for line in table for k in line))
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for line in table:
        print("| " + " | ".join(str(line.get(k, "")) for k in keys) + " |")


if __name__ == "__main__":
    main()
