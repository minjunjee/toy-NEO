#!/usr/bin/env python3
"""Figures and a table from the support-verified search (scripts/babytheorist/support_search.py).

Results of every eval level are pooled by the depth of the hidden program, weighting each level by its number of tasks
at that depth. Writes into --out:
    search_by_length.png   transfer by program length, one line per width, seed mean (thin lines: each seed at the
                           widest and narrowest width)
    search_checks.png      by length, seed mean: share of tasks whose first program for support 1 fails support 2, and
                           share of programs that pass both supports and still miss the query answer
    search_summary.json    the pooled numbers
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def pooled(files: list[str]) -> dict:
    # (run, width) -> depth -> sums
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for f in files:
        for r in json.load(open(f)):
            key = (r["run"], r["width"])
            for d, tasks in enumerate(r["tasks_by_depth"]):
                if not tasks:
                    continue
                cell = acc[key][d + 1]
                found_s1 = (r["found_s1_by_depth"][d] or 0.0) * tasks
                found_all = (r["found_all_by_depth"][d] or 0.0) * tasks
                cell["tasks"] += tasks
                cell["transfer"] += (r["transfer_by_depth"][d] or 0.0) * tasks
                cell["found_s1"] += found_s1
                cell["found_all"] += found_all
                cell["s1_rejected"] += (r["s1_rejected_by_depth"][d] or 0.0) * found_s1
                cell["query_wrong_verified"] += (r["query_wrong_verified_by_depth"][d] or 0.0) * found_all
    out: dict = {}
    for (run, width), depths in acc.items():
        for d, c in depths.items():
            out.setdefault(run, {}).setdefault(width, {})[d] = {
                "tasks": int(c["tasks"]), "transfer": c["transfer"] / c["tasks"],
                "s1_rejected": c["s1_rejected"] / c["found_s1"] if c["found_s1"] else None,
                "query_wrong_verified": c["query_wrong_verified"] / c["found_all"] if c["found_all"] else None}
    return out


def seed_mean(data: dict, width: int, depth: int, key: str):
    values = [data[run][width][depth][key] for run in data if width in data[run] and depth in data[run][width] and data[run][width][depth][key] is not None]
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", default="runs/queue/support_search_lv*_gpu*.json")
    parser.add_argument("--out", type=Path, default=Path("docs/reports/2026-11-23-search"))
    parser.add_argument("--max-length", type=int, default=10)
    args = parser.parse_args()
    data = pooled(sorted(glob.glob(args.inputs)))
    widths = sorted({w for run in data for w in data[run]})
    lengths = list(range(1, args.max_length + 1))
    args.out.mkdir(parents=True, exist_ok=True)
    colors = {1: "#9aa0a6", 4: "#7fa7e0", 8: "#3f7fd6", 16: "#1a56b0", 32: "#0b2f6b"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for run in sorted(data):
        for w in (widths[0], widths[-1]):
            ys = [data[run][w].get(d, {}).get("transfer") for d in lengths]
            ax.plot(lengths, ys, color=colors.get(w, "k"), lw=0.8, alpha=0.35)
    for w in widths:
        ys = [seed_mean(data, w, d, "transfer") for d in lengths]
        ax.plot(lengths, ys, color=colors.get(w, "k"), lw=2.5, marker="o", label=f"width {w}")
    ax.set_xlabel("length of the hidden program")
    ax.set_ylabel("transfer")
    ax.set_ylim(0, 1)
    ax.set_xticks(lengths)
    ax.set_title(f"Support-verified search, one-step models, mean of {len(data)} seeds (thin: each seed)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "search_by_length.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, key, title in zip(axes, ("s1_rejected", "query_wrong_verified"),
                              ("first program for support 1 fails support 2", "passes both supports, misses the query")):
        for w in widths:
            ax.plot(lengths, [seed_mean(data, w, d, key) for d in lengths], color=colors.get(w, "k"), lw=2.2, marker="o", label=f"width {w}")
        ax.set_title(title)
        ax.set_xlabel("length of the hidden program")
        ax.set_xticks(lengths)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("share, seed mean")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "search_checks.png", dpi=150)

    summary = {"runs": sorted(data), "widths": widths, "by_length": {
        str(d): {str(w): {k: seed_mean(data, w, d, k) for k in ("transfer", "s1_rejected", "query_wrong_verified")} for w in widths} for d in lengths},
        "per_run": {run: {str(w): {str(d): v for d, v in sorted(data[run][w].items())} for w in data[run]} for run in data}}
    (args.out / "search_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for d in lengths:
        row = "  ".join(f"w{w} {seed_mean(data, w, d, 'transfer'):.2f}" for w in widths if seed_mean(data, w, d, "transfer") is not None)
        print(f"length {d}: {row}")


if __name__ == "__main__":
    main()
