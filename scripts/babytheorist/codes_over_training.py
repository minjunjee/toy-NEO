#!/usr/bin/env python3
"""Follow the code a model uses for each true primitive across the checkpoints of a run.

For every checkpoint, on the eval tasks whose hidden program is one primitive: per primitive, the share of tasks explained
with exactly one code, which code that is, and the share transferred to the query. The hidden program only sorts the
results. Writes a JSON and a figure with one panel per primitive.

    python scripts/babytheorist/codes_over_training.py --split lv1 --out docs/reports/<name>/codes_over_training runs/<run> [runs/<run> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from evals.babytheorist import babytheorist_observe  # noqa: E402
from training.evaluate import load_run  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict = {}
    for run in args.runs:
        rows = []
        for checkpoint in sorted((run / "ckpt").glob("step-*.pt")):
            model, data, config, step = load_run(run, checkpoint, device)
            infer = next((dict(b.get("infer", {})) for b in config["evals"] if b["type"] == "babytheorist_observe"), {})
            result = babytheorist_observe(model, data, {"split": args.split, "batch_size": 500, "max_samples": args.max_samples, "infer": infer}, step, device)
            rows.append({"step": step, "explain": result["explain"], "transfer_strict": result["transfer_strict"], "code_usage": result.get("code_usage"), "one_step_tasks": result.get("one_step_tasks", {})})
            print(run.name, step, f"explain {result['explain']:.3f} transfer {result['transfer_strict']:.3f}", flush=True)
        report[run.name] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")

    primitives = sorted({p for rows in report.values() for r in rows for p in r["one_step_tasks"]})
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 8.5, "legend.frameon": False})
    figure, axes = plt.subplots(2, len(primitives), figsize=(2.35 * len(primitives), 4.6), sharex=True, sharey=True, squeeze=False)
    for column, primitive in enumerate(primitives):
        for index, (name, rows) in enumerate(report.items()):
            steps = [r["step"] for r in rows]
            entry = lambda r: r["one_step_tasks"].get(primitive, {"tasks": 1, "one_code_extracted": 0, "one_code_exact": 0, "transfer_strict": 0})  # noqa: E731
            axes[0, column].plot(steps, [entry(r).get("one_code_exact", 0) / entry(r)["tasks"] for r in rows], color=SERIES[index % len(SERIES)], linewidth=1.8, label=name)
            axes[1, column].plot(steps, [entry(r)["transfer_strict"] / entry(r)["tasks"] for r in rows], color=SERIES[index % len(SERIES)], linewidth=1.8, label=name)
        axes[0, column].set_title(primitive, fontsize=9)
        axes[1, column].set_xlabel("training step")
    axes[0, 0].set_ylabel("support reproduced\nexactly with one code")
    axes[1, 0].set_ylabel("transferred to the query")
    axes[0, 0].set_ylim(-0.03, 1.03)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 1.0))
    figure.suptitle("Tasks whose hidden program is a single primitive: share reproduced exactly with one code, and share transferred", y=1.06, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(args.out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"-> {args.out.with_suffix('.json')} and .png")


if __name__ == "__main__":
    main()
