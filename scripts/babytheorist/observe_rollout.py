#!/usr/bin/env python3
"""The run's observation evaluation again, with a different rule for the state handed from one program step to the next.

The transformer runs are trained and evaluated with rollout `carry` under state_update reencode_soft: the next step starts
from the encoding of the decoded cell probabilities, which also tells it how sure each cell was. With `hard` the next step
starts from the encoding of the decoded grid itself (argmax), so every step can only use what is visible on the grid.
Programs of one code are the same under both; the difference shows how much of the multi-code programs leans on the part
that is not visible on the grid. Nothing is written into the run directory.

    python scripts/babytheorist/observe_rollout.py --split lv1 --rollout hard [--out file.json] runs/<run> [...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from evals.babytheorist import babytheorist_observe  # noqa: E402
from training.evaluate import load_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = []
    for run in args.runs:
        model, data, config, step = load_run(run, args.ckpt, device)
        block = next(dict(b) for b in config["evals"] if b["type"] == "babytheorist_observe" and b["split"] == args.split)
        block.update(infer={**block.get("infer", {}), "rollout": args.rollout}, tables=True, max_samples=args.max_samples)
        with torch.no_grad():
            result = babytheorist_observe(model, data, block, step, device)
        report.append({"run": run.name, "step": step, "infer": block["infer"], **result})
        by_depth = "  ".join(f"{result[f'explain_depth{d}']:.3f}/{result[f'transfer_strict_depth{d}']:.3f}" for d in (1, 2, 3, 4, 5))
        print(f"{run.name} step {step} rollout {args.rollout}: explain {result['explain']:.3f} transfer {result['transfer_strict']:.3f} | explain/transfer by depth 1..5+: {by_depth}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
