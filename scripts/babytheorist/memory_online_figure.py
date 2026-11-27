#!/usr/bin/env python3
"""Figure for the online fast memory (memory_online.py): accuracy on a new primitive's task against how many tasks of that
primitive the stream had shown before, for each online run, with the offline reference of a new code fitted on k labelled
examples (memory_new_codes.py) where given.

Occurrences are binned 0, 1, 2, 3-4, 5-8, 9 and more; each bin is the mean over the stream's tasks of new primitives in it.
New primitives are also split by the offline reference: those a new code learns from 16 labelled examples to 50% or more
(the executor can already compute them) and the rest of those the reference covers (it has no labelled examples of the
4-cell shifts, which the train file leaves out; they count only in the panel of all new primitives).

    python scripts/babytheorist/memory_online_figure.py --online runs/queue/memory_online_plain.json runs/queue/memory_online_refit.json \
        --labels "verify, else allocate" "verify, else refit, else allocate" --offline runs/queue/memory_new_codes.json \
        --out docs/reports/2026-11-27-memory/online
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 99)]
NAMES = ["0", "1", "2", "3-4", "5-8", "9+"]
COLORS = ("#8b929c", "#2a78d6", "#eb6834", "#1baf7a")


def curve(log, keep):
    seen = defaultdict(int)
    by_bin = defaultdict(list)
    for row in log:
        if row["known"] or not keep(row["primitive"]):
            continue
        i = seen[row["primitive"]]
        seen[row["primitive"]] += 1
        for b, (lo, hi) in enumerate(BINS):
            if lo <= i <= hi:
                by_bin[b].append(row["right"])
    return [float(np.mean(by_bin[b])) if by_bin[b] else np.nan for b in range(len(BINS))], [len(by_bin[b]) for b in range(len(BINS))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--online", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--offline", type=Path)
    parser.add_argument("--offline-run", default="onestep_s_m24_hardem_pairs2_price_s43")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    offline = None
    if args.offline and args.offline.exists():
        entries = [e for e in json.loads(args.offline.read_text()) if e["run"] == args.offline_run and e.get("init") == "nearest"]
        if entries:
            offline = entries[-1]
    easy = (lambda name: offline["primitives"].get(name, {}).get("k16", {}).get("held_out", 0) >= 0.5) if offline else None
    groups = [("all new primitives", lambda name: True)]
    if easy:
        hard = lambda name: name in offline["primitives"] and not easy(name)  # noqa: E731
        groups += [("computable by the executor", easy), ("needing a new computation", hard)]
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.4), sharey=True, squeeze=False)
    summary = {}
    for g, (title, keep) in enumerate(groups):
        ax = axes[0][g]
        for k, (path, label) in enumerate(zip(args.online, args.labels)):
            report = json.loads(path.read_text())[-1]
            ys, ns = curve(report["log"], keep)
            ax.plot(range(len(BINS)), ys, "-o", color=COLORS[(k + 1) % len(COLORS)], lw=1.6, ms=4, label=label)
            summary.setdefault(label, {})[title] = {"accuracy": [None if np.isnan(y) else round(y, 3) for y in ys], "tasks": ns,
                                                    "new_codes": report["new_codes"]}
        if offline:
            names = [n for n, e in offline["primitives"].items() if keep(n)]
            ks = [("k1", 1), ("k2", 2), ("k4", 3), ("k16", 5)]
            ys = [float(np.mean([offline["primitives"][n][key]["held_out"] for n in names])) for key, _ in ks]
            ax.plot([x for _, x in ks], ys, "--s", color=COLORS[0], lw=1.2, ms=4, label="new code from k labelled examples")
            summary.setdefault("offline", {})[title] = dict(zip([key for key, _ in ks], [round(y, 3) for y in ys]))
        ax.set_xticks(range(len(BINS)))
        ax.set_xticklabels(NAMES, fontsize=8)
        ax.set_xlabel("tasks of this primitive seen before", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#eee", lw=0.6)
    axes[0][0].set_ylabel("query exact", fontsize=8)
    axes[0][0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=160)
    Path(f"{args.out}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
