#!/usr/bin/env python3
"""Before and after dream training of the programmer: transfer by length of the hidden program, seed mean, for the
searches of support_search.py on the one-step models (before) and on their dream-trained continuations (after).

    python scripts/babytheorist/dream_figures.py [--out docs/reports/2026-11-23-dream]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_figures import pooled, seed_mean  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", default="runs/queue/support_search_lv*_gpu*.json")
    parser.add_argument("--after", default="runs/queue/dream_search_lv*_gpu*.json")
    parser.add_argument("--out", type=Path, default=Path("docs/reports/2026-11-23-dream"))
    parser.add_argument("--max-length", type=int, default=10)
    args = parser.parse_args()
    before, after = pooled(sorted(glob.glob(args.before))), pooled(sorted(glob.glob(args.after)))
    seeds = sorted({run[-3:] for run in after})
    before = {run: v for run, v in before.items() if run[-3:] in seeds}
    lengths = list(range(1, args.max_length + 1))
    widths = sorted({w for run in after for w in after[run]})
    args.out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {1: "#9aa0a6", 4: "#7fa7e0", 16: "#1a56b0"}
    for w in widths:
        ax.plot(lengths, [seed_mean(before, w, d, "transfer") for d in lengths], color=colors.get(w, "k"), lw=1.8, ls="--", marker="o", ms=4)
        ax.plot(lengths, [seed_mean(after, w, d, "transfer") for d in lengths], color=colors.get(w, "k"), lw=2.6, marker="o", label=f"width {w}")
    ax.plot([], [], color="k", ls="--", label="before dreaming")
    ax.set_xlabel("length of the hidden program")
    ax.set_ylabel("transfer, seed mean")
    ax.set_xticks(lengths)
    ax.set_ylim(0, 1)
    ax.set_title(f"Programmer trained on its own dreamed programs, {len(after)} seeds (dashed: before)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "dream_by_length.png", dpi=150)
    summary = {}
    for w in widths:
        summary[str(w)] = {str(d): {"before": seed_mean(before, w, d, "transfer"), "after": seed_mean(after, w, d, "transfer"),
                                    "query_wrong_verified_before": seed_mean(before, w, d, "query_wrong_verified"),
                                    "query_wrong_verified_after": seed_mean(after, w, d, "query_wrong_verified")} for d in lengths}
        print(f"width {w}: " + "  ".join(f"{d}: {summary[str(w)][str(d)]['before']:.2f}->{summary[str(w)][str(d)]['after']:.2f}" for d in lengths
                                      if summary[str(w)][str(d)]["after"] is not None))
    per_seed = {}
    for run in sorted(after):
        twin = next(r for r in before if r[-3:] == run[-3:])
        per_seed[run] = {str(w): {str(d): {"before": before[twin][w].get(d, {}).get("transfer"), "after": after[run][w].get(d, {}).get("transfer")} for d in lengths} for w in widths}
    (args.out / "dream_summary.json").write_text(json.dumps({"by_width": summary, "per_seed": per_seed}, indent=1) + "\n")


if __name__ == "__main__":
    main()
