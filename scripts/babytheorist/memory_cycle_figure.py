#!/usr/bin/env python3
"""Figure for rounds of waking and sleeping on one stream of tasks (memory_online.py, configs sleep_*).

A model that never sleeps meets the whole stream with its fast memory only (`--awake`, one online run). The other model
meets the same stream in parts (`--rounds`, online runs in stream order), sleeping between them: each sleep trains the
weights on the tasks its fast memory kept. Per window of `--window` tasks of the stream: accuracy on the tasks of new
primitives (new against the model that started the stream), and how many new codes the fast memory has made so far in
the current wake. Dashed vertical lines mark the sleeps.

    python scripts/babytheorist/memory_cycle_figure.py --awake runs/queue/memory_online_refit_2400.json \
        --rounds runs/queue/memory_online_refit.json runs/queue/memory_online_wake2_fresh.json \
        --out docs/reports/2026-11-27-memory/cycle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def windows(log, size):
    acc, known, codes, made = [], [], [], 0
    for start in range(0, len(log) - size + 1, size):
        w = log[start: start + size]
        new = [r["right"] for r in w if not r["known"]]
        old = [r["right"] for r in w if r["known"]]
        made += sum(r["kind"] == "allocated" for r in w)
        acc.append(float(np.mean(new)) if new else np.nan)
        known.append(float(np.mean(old)) if old else np.nan)
        codes.append(made)
    return acc, known, codes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--awake", type=Path, help="one online run over the whole stream, no sleep")
    parser.add_argument("--rounds", type=Path, nargs="+", required=True, help="online runs over successive parts of the stream, a sleep between them")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    summary = {}
    if args.awake and args.awake.exists():
        log = json.loads(args.awake.read_text())[-1]["log"]
        acc, _, codes = windows(log, args.window)
        xs = (np.arange(len(acc)) + 1) * args.window
        axes[0].plot(xs, acc, "-", color="#8b929c", lw=1.6, label="fast memory only, no sleep")
        axes[1].plot(xs, codes, "-", color="#8b929c", lw=1.6)
        summary["no sleep"] = {"new": [round(a, 3) for a in acc], "codes": codes}
    offset = 0
    for k, path in enumerate(args.rounds):
        log = json.loads(path.read_text())[-1]["log"]
        acc, known, codes = windows(log, args.window)
        xs = offset + (np.arange(len(acc)) + 1) * args.window
        axes[0].plot(xs, acc, "-o", color="#eb6834", lw=1.6, ms=3, label="fast memory, sleep between parts" if k == 0 else None)
        axes[1].plot(xs, codes, "-o", color="#eb6834", lw=1.6, ms=3)
        summary[f"wake {k + 1}"] = {"new": [round(a, 3) for a in acc], "known": [round(a, 3) for a in known], "codes": codes,
                                    "new_mean": round(float(np.mean([r["right"] for r in log if not r["known"]])), 3)}
        offset += len(log)
        if k + 1 < len(args.rounds):
            for ax in axes:
                ax.axvline(offset + args.window * 0.5, color="#444", lw=0.8, ls="--")
    axes[0].set_ylabel("tasks of new primitives, query exact", fontsize=8)
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", color="#eee", lw=0.6)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    axes[1].set_ylabel("new codes made in the current wake", fontsize=8)
    for ax in axes:
        ax.set_xlabel("tasks of the stream met so far", fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=160)
    Path(f"{args.out}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
