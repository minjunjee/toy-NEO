#!/usr/bin/env python3
"""The run's observation evaluation again, with a different rule for the state handed from one program step to the next.

The transformer runs are trained and evaluated with rollout `carry` under state_update reencode_soft: the next step starts
from the encoding of the decoded cell probabilities, which also tells it how sure each cell was. With `hard` the next step
starts from the encoding of the decoded grid itself (argmax), so every step can only use what is visible on the grid.
Programs of one code are the same under both; the difference shows how much of the multi-code programs leans on the part
that is not visible on the grid. Nothing is written into the run directory.

With --eval-file the tasks come from another eval file than the ones in the run's config, for example programs longer than
the run was trained on; --max-steps lets the program be longer than the run's max_length, and --deepest-bucket reports
depths one by one further than 5.

    python scripts/babytheorist/observe_rollout.py --split lv1 --rollout hard [--out file.json] runs/<run> [...]
    python scripts/babytheorist/observe_rollout.py --split lv3 --eval-file data/babytheorist/single_a/lv3_eval.h5 --rollout carry \
        --max-steps 10 --deepest-bucket 10 runs/<run>
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
    parser.add_argument("--eval-file", type=Path, help="evaluate on this eval file under the name given by --split")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--deepest-bucket", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = []
    for run in args.runs:
        model, data, config, step = load_run(run, args.ckpt, device)
        if args.eval_file:
            data.eval_paths[args.split] = str(args.eval_file)
        block = next(dict(b) for b in config["evals"] if b["type"] == "babytheorist_observe" and (args.eval_file or b["split"] == args.split))
        block.update(split=args.split, infer={**block.get("infer", {}), "rollout": args.rollout}, tables=True, max_samples=args.max_samples)
        if args.max_steps:
            block["infer"]["max_steps"] = args.max_steps
        if args.deepest_bucket:
            block["deepest_bucket"] = args.deepest_bucket
        if args.batch_size:
            block["batch_size"] = args.batch_size
        with torch.no_grad():
            result = babytheorist_observe(model, data, block, step, device)
        report.append({"run": run.name, "step": step, "infer": block["infer"], **result})
        depths = range(1, (args.deepest_bucket or 5) + 1)
        by_depth = "  ".join(f"{result[f'explain_depth{d}']:.3f}/{result[f'transfer_strict_depth{d}']:.3f}" for d in depths)
        print(f"{run.name} step {step} split {args.split} {block['infer']}: explain {result['explain']:.3f} transfer {result['transfer_strict']:.3f} | explain/transfer by depth 1..{depths[-1]}+: {by_depth}", flush=True)
        print("    tasks by depth: " + "  ".join(str(result[f"tasks_depth{d}"]) for d in depths), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
