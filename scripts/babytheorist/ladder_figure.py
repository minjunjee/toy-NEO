#!/usr/bin/env python3
"""Figure for the code-as-function ladder (code_function_ladder.py): for each model, the true code's one-step accuracy on
states moving away from the training distribution (new objects, objects after 1 to 5 random primitives, larger objects).
Runs are averaged within groups given as label=run,run,...

    python scripts/babytheorist/ladder_figure.py --out docs/reports/2026-11-27-ladder/ladder \
        --group "24, price=onestep_s_m24_hardem_pairs2_price_s42,onestep_s_m24_hardem_pairs2_price_s43" ...
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8b929c", "#e34948")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", required=True, help="label=run,run,...")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for path in glob.glob(str(REPO / "runs" / "queue" / "ladder_*.json")):
        for r in json.loads(Path(path).read_text()):
            rows[r["run"]] = r["mean_accuracy"]
    rungs = ["fresh", "after_1", "after_2", "after_3", "after_4", "after_5", "large"]
    names = ["new\nobject", "after 1", "after 2", "after 3", "after 4", "after 5", "larger\nobjects"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    summary = {}
    for k, spec in enumerate(args.group):
        label, runs = spec.split("=")
        found = [rows[r] for r in runs.split(",") if r in rows]
        if not found:
            continue
        mean = [float(np.mean([f[g] for f in found])) for g in rungs]
        summary[label] = {"runs": len(found), "mean": dict(zip(rungs, mean))}
        ax.plot(range(6), mean[:6], "-o", color=COLORS[k % len(COLORS)], lw=1.6, ms=4, label=f"{label} ({len(found)})")
        ax.plot([6], [mean[6]], "s", color=COLORS[k % len(COLORS)], ms=6)
    ax.axvline(5.5, color="#ccc", lw=0.8)
    ax.set_xticks(range(7))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("true code, one step, exact", fontsize=8)
    ax.set_ylim(0.4, 1.01)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=160)
    Path(f"{args.out}.json").write_text(json.dumps(summary, indent=1) + "\n")
    for label, s in summary.items():
        print(label, {k: round(v, 2) for k, v in s["mean"].items()})


if __name__ == "__main__":
    main()
