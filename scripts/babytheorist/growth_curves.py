#!/usr/bin/env python3
"""Learning curves when an environment grows: a run that continues from a smaller environment's model against a run that
starts from nothing on the same data, from their light codebook records (evals/codebook_snapshot.py).

For each record step, each primitive's share of probe tasks whose query the programmer's chosen code gets right (about
twenty probe tasks per primitive at 52 primitives, so each value moves in steps of about 5%; curves are smoothed over three
records). Primitives are split into those of the smaller set (already known to the grown run), the new ones it has to
learn, and new ones that the train file leaves out (shift by four cells, never trained by either run).

Reported per pair of runs: for every new primitive, the first step from which its smoothed rate stays at or above
`--level` (default 0.8) for the rest of the run, and the median over new primitives; for the known primitives, their mean
at the first record and the lowest mean over the run (how much the old knowledge dips while the new is learned).

    python scripts/babytheorist/growth_curves.py --small 24 --large 52 --out docs/reports/2026-11-27-growth/m24_to_m52 \
        --pair runs/grow_m24_to_m52_price_s43 runs/onestep_s_m52_hardem_pairs2_price_s43 [--pair ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "babytheorist"))

import generate_many_primitives as gen  # noqa: E402

UNTRAINED = ("shift_up_4", "shift_down_4", "shift_left_4", "shift_right_4")


def curves(run: Path) -> tuple[list[int], dict[str, np.ndarray]]:
    records = sorted((json.loads(Path(p).read_text()) for p in glob.glob(str(run / "eval" / "snapshot_*_step*.json"))), key=lambda r: r["step"])
    steps = [r["step"] for r in records]
    names = sorted({n for r in records for n in r["primitives"]})
    out = {}
    for n in names:
        raw = np.array([r["primitives"][n]["transfer"] / max(r["primitives"][n]["tasks"], 1) if n in r["primitives"] else np.nan for r in records])
        smooth = np.convolve(np.nan_to_num(raw), np.ones(3) / 3, mode="same")
        smooth[0], smooth[-1] = raw[0], raw[-1]
        out[n] = smooth
    return steps, out


def first_stay(steps: list[int], values: np.ndarray, level: float) -> int | None:
    for i in range(len(values)):
        if (values[i:] >= level).all():
            return steps[i]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--small", default="24")
    parser.add_argument("--large", default="52")
    parser.add_argument("--pair", nargs=2, action="append", type=Path, required=True, metavar=("GROWN", "SCRATCH"))
    parser.add_argument("--level", type=float, default=0.8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    old = set(gen.SETS[args.small])
    new = [n for n in gen.SETS[args.large] if n not in old and n not in UNTRAINED]
    untrained = [n for n in gen.SETS[args.large] if n in UNTRAINED]
    summary = {}
    fig, axes = plt.subplots(1, len(args.pair), figsize=(5.2 * len(args.pair), 4.2), squeeze=False, sharey=True)
    for k, (grown, scratch) in enumerate(args.pair):
        ax = axes[0, k]
        entry = {}
        for run, style in ((grown, "-"), (scratch, "--")):
            steps, by = curves(run)
            if not steps:
                continue
            kind = "grown" if run == grown else "scratch"
            for group, names, color in (("known", sorted(old), "#2a78d6"), ("new", new, "#eb6834"), ("untrained", untrained, "#999999")):
                present = [n for n in names if n in by]
                if not present:
                    continue
                mean = np.mean([by[n] for n in present], axis=0)
                ax.plot(steps, mean, style, color=color, lw=1.4, label=f"{kind}: {group} ({len(present)})")
            reach = {n: first_stay(steps, by[n], args.level) for n in new if n in by}
            got = [v for v in reach.values() if v is not None]
            known_mean = np.mean([by[n] for n in old if n in by], axis=0)
            entry[kind] = {"last_step": steps[-1], "new_reached": f"{len(got)}/{len(reach)}", "new_median_step": float(np.median(got)) if got else None,
                           "new_steps": reach, "new_mean_last": float(np.mean([by[n][-1] for n in new if n in by])),
                           "known_first": float(known_mean[0]), "known_lowest": float(known_mean.min()), "known_last": float(known_mean[-1]),
                           "untrained_last": float(np.mean([by[n][-1] for n in untrained if n in by])) if untrained else None}
        summary[grown.name] = entry
        ax.set_title(f"{grown.name}\nsolid: grown from {args.small}, dashed: from nothing", fontsize=8)
        ax.set_xlabel("training step", fontsize=8)
        ax.set_ylim(0, 1.02)
        if k == 0:
            ax.set_ylabel("query right with the chosen code", fontsize=8)
        ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=160)
    Path(f"{args.out}.json").write_text(json.dumps(summary, indent=1) + "\n")
    for name, entry in summary.items():
        for kind, e in entry.items():
            print(name, kind, {k: v for k, v in e.items() if k != "new_steps"})


if __name__ == "__main__":
    main()
