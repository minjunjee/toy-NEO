#!/usr/bin/env python3
"""How the codebook settles during training, from the light records of evals/codebook_snapshot.py (every 100 steps).

The hidden primitive of each probe task only sorts the record. Figures, one column per run:
  1. assignment over time: for each primitive, the share of its probe tasks for which the programmer picked each code,
     at a few steps; columns are codes ordered by the primitive they serve at the end
  2. the four orientation primitives that swap width and height: how many codes they effectively use together (the
     exponential of the entropy of their pooled choices), and how often each is solved on the query
  3. geometry: cosine similarity between the codes the four primitives choose most, and how far the codebook moves
     between records
  4. the executor's meaning: for each of the four, the share of its tasks that its own most-chosen code reproduces,
     against the share that ANY of the four's codes reproduces
  5. the hard EM side: number of codes no task chose recently (usage below a tenth of uniform) and the largest bonus

    python scripts/babytheorist/codebook_settling.py --out docs/reports/2026-11-26-codebook/settling runs/<run> [runs/<run> ...]
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

SWAP = ("rot90", "rot270", "transpose", "anti_transpose")
SWAP_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")
GROUPS = (("swap width and height", SWAP), ("flips and half turn", ("flip_h", "flip_v", "rot180")),
          ("colour", ("next_color", "prev_color", "color_plus3")), ("shape", ("hollow", "add_border")))


def load(run: Path) -> list[dict]:
    records = []
    for path in sorted(glob.glob(str(run / "eval" / "snapshot_*_step*.json"))):
        records.append(json.loads(Path(path).read_text()))
    return sorted(records, key=lambda r: r["step"])


def choice_matrix(record: dict, names: list[str]) -> np.ndarray:
    rows = []
    for name in names:
        entry = record["primitives"].get(name)
        if entry is None or not entry["tasks"]:
            rows.append(np.zeros(len(record["codes"])))
        else:
            rows.append(np.array(entry["chosen"], dtype=float) / entry["tasks"])
    return np.array(rows)


def effective_codes(record: dict, names=SWAP) -> float:
    pooled = sum(np.array(record["primitives"][n]["chosen"], dtype=float) for n in names if n in record["primitives"])
    p = pooled / max(pooled.sum(), 1)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def top_code(record: dict, name: str) -> int:
    return int(np.argmax(record["primitives"][name]["chosen"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="path without extension; .png and .json are written")
    parser.add_argument("--labels", nargs="*", help="one label per run")
    parser.add_argument("--steps", type=int, nargs="*", default=[100, 300, 600, 1000, 2000, 3000, 5000, 8000, 24000])
    args = parser.parse_args()
    runs = [(run, load(run)) for run in args.runs]
    labels = args.labels or [run.name for run in args.runs]
    names = [n for _, group in GROUPS for n in group]
    rest = sorted(n for n in runs[0][1][-1]["primitives"] if n not in names)
    names += rest
    summary = {}

    # figure 1: assignment heatmaps at a few steps, one row of panels per run
    steps = [s for s in args.steps if any(r["step"] == s for _, recs in runs for r in recs)]
    fig, axes = plt.subplots(len(runs), len(steps), figsize=(2.1 * len(steps), 4.6 * len(runs)), squeeze=False)
    for i, ((run, records), label) in enumerate(zip(runs, labels)):
        last = records[-1]
        final = choice_matrix(last, names)
        order = sorted(range(final.shape[1]), key=lambda c: (int(np.argmax(final[:, c])) if final[:, c].sum() else 999, -final[:, c].sum()))
        for j, step in enumerate(steps):
            ax = axes[i, j]
            rec = next((r for r in records if r["step"] == step), None)
            if rec is None:
                ax.axis("off")
                continue
            ax.imshow(choice_matrix(rec, names)[:, order], aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"{label}\nstep {step}" if j == 0 else f"step {step}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks(range(len(names)) if j == 0 else [])
            if j == 0:
                ax.set_yticklabels(names, fontsize=6)
                for k, n in enumerate(names):
                    if n in SWAP:
                        ax.get_yticklabels()[k].set_color(SWAP_COLORS[SWAP.index(n)])
            ax.axhline(len(SWAP) - 0.5, color="#d33", lw=0.6)
    fig.suptitle("Which code the programmer picks for each primitive (rows), codes ordered by their final primitive", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{args.out}_assignment.png", dpi=160)
    plt.close(fig)

    # figure 2 to 5: curves over steps
    fig, axes = plt.subplots(5, len(runs), figsize=(5.2 * len(runs), 13), squeeze=False, sharex=True)
    for i, ((run, records), label) in enumerate(zip(runs, labels)):
        x = [r["step"] for r in records]
        ax = axes[0, i]
        ax.plot(x, [effective_codes(r) for r in records], color="#222", lw=1.4, label="the four together")
        ax.plot(x, [effective_codes(r, names=tuple(n for n in names if n not in SWAP)) / (len(names) - len(SWAP)) * len(SWAP) for r in records],
                color="#999", lw=1, ls="--", label="the other primitives, rescaled to four")
        ax.axhline(4, color="#bbb", lw=0.6)
        ax.set_ylabel("codes in use\n(exp entropy)", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax = axes[1, i]
        for n, color in zip(SWAP, SWAP_COLORS):
            ax.plot(x, [r["primitives"][n]["transfer"] / max(r["primitives"][n]["tasks"], 1) for r in records], color=color, lw=1, label=n)
        ax.set_ylabel("query solved\nwith the chosen code", fontsize=8)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax = axes[2, i]
        sims, moves = [], []
        prev = None
        for r in records:
            codes = np.array(r["codes"])
            tops = [top_code(r, n) for n in SWAP]
            distinct = sorted(set(tops))
            pair = [float(codes[a] @ codes[b]) for k, a in enumerate(distinct) for b in distinct[k + 1:]]
            sims.append(np.mean(pair) if pair else 1.0)
            moves.append(float(np.linalg.norm(codes - prev, axis=1).mean()) if prev is not None else np.nan)
            prev = codes
        ax.plot(x, sims, color="#222", lw=1.2, label="cosine between the four's top codes (1 when shared)")
        ax.set_ylabel("cosine", fontsize=8)
        ax.set_ylim(-0.2, 1.05)
        twin = ax.twinx()
        twin.plot(x, moves, color="#c77", lw=0.8, label="codebook movement per 100 steps")
        twin.set_ylabel("mean code movement", fontsize=7, color="#c77")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax = axes[3, i]
        for n, color in zip(SWAP, SWAP_COLORS):
            own = [r["primitives"][n]["reproduces"][top_code(r, n)] / max(r["primitives"][n]["tasks"], 1) for r in records]
            ax.plot(x, own, color=color, lw=1, label=f"{n}: its top code")
        ax.set_ylabel("support reproduced by\nits own top code", fontsize=8)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax = axes[4, i]
        if "em_usage" in records[0]:
            size = len(records[0]["em_usage"])
            ax.plot(x, [sum(u < 0.1 / size for u in r["em_usage"]) for r in records], color="#222", lw=1, label="codes with usage below a tenth of uniform")
            twin = ax.twinx()
            twin.plot(x, [max(r["em_bonus"]) for r in records], color="#c77", lw=0.8)
            twin.set_ylabel("largest dead-code bonus", fontsize=7, color="#c77")
            ax.legend(fontsize=7, frameon=False)
        ax.set_ylabel("unused codes", fontsize=8)
        ax.set_xlabel("training step", fontsize=8)
        ax.set_xscale("log")
        summary[label] = {"steps": x, "effective_codes_swap": [effective_codes(r) for r in records], "code_nmi": [r["code_nmi"] for r in records],
                          "swap_top_cosine": sims, "codebook_movement": moves}
    fig.tight_layout()
    fig.savefig(f"{args.out}_curves.png", dpi=160)
    plt.close(fig)
    # figure 3: only the four. Top: for each of them, the code the programmer picks most often, step by step (the lines
    # lie on top of each other while the four share a code and part when they split; a dot's size is the share of the
    # primitive's tasks on that code). Bottom: the share of each one's query answers the chosen code gets right.
    fig, axes = plt.subplots(2, len(runs), figsize=(5.4 * len(runs), 6.4), squeeze=False, sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    for i, ((run, records), label) in enumerate(zip(runs, labels)):
        x = [r["step"] for r in records]
        used = sorted({top_code(r, n) for r in records for n in SWAP})
        rank = {c: k for k, c in enumerate(used)}
        ax = axes[0, i]
        for j, (n, color) in enumerate(zip(SWAP, SWAP_COLORS)):
            ys = [rank[top_code(r, n)] + (j - 1.5) * 0.08 for r in records]
            share = [max(r["primitives"][n]["chosen"]) / max(r["primitives"][n]["tasks"], 1) for r in records]
            ax.plot(x, ys, color=color, lw=1.0, alpha=0.8, label=n)
            ax.scatter(x, ys, s=[4 + 22 * v for v in share], color=color, alpha=0.8)
        ax.set_yticks(range(len(used)))
        ax.set_yticklabels([f"code {c}" for c in used], fontsize=6)
        ax.set_title(f"{label}: the code each of the four picks most", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=4)
        ax = axes[1, i]
        for n, color in zip(SWAP, SWAP_COLORS):
            ax.plot(x, [r["primitives"][n]["transfer"] / max(r["primitives"][n]["tasks"], 1) for r in records], color=color, lw=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("query right with\nthe chosen code", fontsize=8)
        ax.set_xlabel("training step", fontsize=8)
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(f"{args.out}_four.png", dpi=160)
    plt.close(fig)
    Path(f"{args.out}.json").write_text(json.dumps(summary) + "\n")
    print("wrote", f"{args.out}_assignment.png", f"{args.out}_curves.png")


if __name__ == "__main__":
    main()
