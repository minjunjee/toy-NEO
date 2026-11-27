#!/usr/bin/env python3
"""Curves of a sleep that consolidates an online fast memory (configs sleep_*): at every light codebook record, the share of
probe tasks whose query the programmer's chosen code gets right, averaged over groups of primitives: the 24 the model knew,
the new ones a new code alone learned to 50% or more from 16 labelled examples (memory_new_codes.py, the executor can
compute them), the new ones it did not, and the 4-cell shifts (no labelled examples). Each run is drawn as one line per
group; a reference run (growth on the full 52-primitive file) can be added and is cut to the same steps.

    python scripts/babytheorist/sleep_curves.py --runs runs/sleep_fresh_m24_to_m52_s43 runs/sleep_memory_m24_to_m52_s43 \
        --labels "fresh rows" "fast-memory rows" --reference runs/grow_m24_to_m52_price_s43 --out docs/reports/2026-11-27-memory/sleep
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "babytheorist"))
import generate_many_primitives as gen  # noqa: E402

COLORS = ("#2a78d6", "#eb6834", "#8b929c")


def records(run: Path, name: str = "snapshot_lv1"):
    out = {}
    for path in glob.glob(str(run / "eval" / f"{name}_step*.json")):
        step = int(re.search(r"step(\d+)", path).group(1))
        prims = json.loads(Path(path).read_text())["primitives"]
        out[step] = {n: e["transfer"] / max(e["tasks"], 1) for n, e in prims.items() if e["tasks"]}
    return dict(sorted(out.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--offline", type=Path, default=REPO / "runs" / "queue" / "memory_new_codes.json")
    parser.add_argument("--offline-run", default="onestep_s_m24_hardem_pairs2_price_s43")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    known = set(gen.SETS["24"])
    entry = [e for e in json.loads(args.offline.read_text()) if e["run"] == args.offline_run and e.get("init") == "nearest"][-1]
    easy = {n for n, e in entry["primitives"].items() if e["k16"]["held_out"] >= 0.5}
    hard = {n for n in entry["primitives"] if n not in easy}
    shift4 = {n for n in gen.SETS["52"] if n.endswith("_4") and n.startswith("shift")}
    groups = [("known 24", known), ("new, computable", easy), ("new, new computation", hard), ("4-cell shifts", shift4)]
    runs = list(zip(args.runs, args.labels))
    if args.reference:
        runs.append((args.reference, "growth, 80,000 tasks"))
    last = min(max(records(r)) for r, _ in zip(args.runs, args.labels) if records(r))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.3 * len(groups), 3.0), sharey=True)
    summary = {}
    for k, (run, label) in enumerate(runs):
        rec = {s: v for s, v in records(run).items() if s <= max(last, 100)}
        if not rec:
            continue
        steps = list(rec)
        for g, (title, names) in enumerate(groups):
            ys = [float(np.mean([v[n] for n in names if n in v])) for v in rec.values()]
            axes[g].plot(steps, ys, "-", color=COLORS[k % len(COLORS)], lw=1.6, label=label)
            summary.setdefault(label, {})[title] = {str(s): round(y, 3) for s, y in zip(steps, ys) if s in (steps[0], steps[len(steps) // 2], steps[-1])}
    for g, (title, _) in enumerate(groups):
        axes[g].set_title(title, fontsize=9)
        axes[g].set_xlabel("sleep steps", fontsize=8)
        axes[g].set_ylim(0, 1)
        axes[g].grid(axis="y", color="#eee", lw=0.6)
    axes[0].set_ylabel("query exact, chosen code", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=160)
    Path(f"{args.out}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
