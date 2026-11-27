#!/usr/bin/env python3
"""Rounds of waking and sleeping at several seeds (memory_online.py runs over successive parts of one stream, a sleep
between them): per wake, the share of the new primitives' tasks answered right, split by kind of primitive, and the
number of new codes the fast memory made in that wake. Thin lines are seeds, the thick line their mean.

New primitives are split by the labelled diagnostic of memory_new_codes.py on the seed-43 price model: those a new code
alone learned to 50% or more from 16 examples (the executor can compute them) and the others it covers.

    python scripts/babytheorist/memory_rounds_figure.py --out docs/reports/2026-11-27-memory/rounds \
        --seed 42=runs/queue/memory_online_refit_s42.json,runs/queue/memory_online_wake2_memory_s42.json,... --seed 43=...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "babytheorist"))

PANELS = [("all new primitives", "#eb6834"), ("computable by the executor", "#2a78d6"), ("needing a new computation", "#6250d6"), ("known 24", "#8b929c")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", action="append", required=True, help="label=first wake json,second wake json,...")
    parser.add_argument("--offline", type=Path, default=REPO / "runs" / "queue" / "memory_new_codes.json")
    parser.add_argument("--offline-run", default="onestep_s_m24_hardem_pairs2_price_s43")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    entry = [e for e in json.loads(args.offline.read_text()) if e["run"] == args.offline_run and e.get("init") == "nearest"][-1]
    easy = {n for n, e in entry["primitives"].items() if e["k16"]["held_out"] >= 0.5}
    hard = set(entry["primitives"]) - easy
    table = {}
    for spec in args.seed:
        label, paths = spec.split("=")
        rows = []
        for path in paths.split(","):
            if not Path(path).exists():
                break
            report = json.loads(Path(path).read_text())[-1]
            acc = defaultdict(list)
            for r in report["log"]:
                if r["known"]:
                    acc["known 24"].append(r["right"])
                    continue
                acc["all new primitives"].append(r["right"])
                if r["primitive"] in easy:
                    acc["computable by the executor"].append(r["right"])
                elif r["primitive"] in hard:
                    acc["needing a new computation"].append(r["right"])
            rows.append({**{k: float(np.mean(v)) for k, v in acc.items()}, "codes": report["new_codes"]})
        table[label] = rows
    fig, axes = plt.subplots(1, 5, figsize=(16.0, 3.1))
    summary = {"seeds": {k: [{a: round(b, 3) for a, b in r.items()} for r in rows] for k, rows in table.items()}, "mean": {}}
    wakes = max(len(rows) for rows in table.values())
    for g, (title, color) in enumerate(PANELS + [("codes", "#eb6834")]):
        ax = axes[g]
        for rows in table.values():
            ax.plot(range(len(rows)), [r[title] for r in rows], "-", color=color, lw=0.8, alpha=0.45)
        mean = [float(np.mean([rows[w][title] for rows in table.values() if len(rows) > w])) for w in range(wakes)]
        count = [sum(len(rows) > w for rows in table.values()) for w in range(wakes)]
        ax.plot(range(wakes), mean, "-o", color=color, lw=2.0, ms=4)
        summary["mean"][title] = [round(m, 3) for m in mean]
        summary["seeds_per_wake"] = count
        ax.set_xticks(range(wakes))
        ax.set_xlabel("sleeps before this wake", fontsize=8)
        if title == "codes":
            ax.set_title("new codes made in the wake", fontsize=9)
        else:
            ax.set_title(title, fontsize=9)
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", color="#eee", lw=0.6)
    axes[0].set_ylabel("query exact over the wake's 800 tasks", fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=160)
    Path(f"{args.out}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["mean"]), summary["seeds_per_wake"])


if __name__ == "__main__":
    main()
