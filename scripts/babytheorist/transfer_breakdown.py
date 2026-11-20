#!/usr/bin/env python3
"""Why transfer fails on single-primitive tasks: compare the code that explains the support pair with the code that
explains the query pair on its own.

Each pair is explained separately (the query pair with its answer, for this diagnosis only). A task falls into one of:
  same code        both pairs are reproduced exactly with one code, and it is the same code
  different codes  both pairs are reproduced exactly with one code, but not the same one: the primitive is split
  not covered      the support pair is reproduced with one code; the query pair is not reproduced with one code
  not one code     the support pair is not reproduced exactly with one code (several codes, or not at all)
and for each group the share of tasks transferred (support program replayed on the query, answer not shown).

    python scripts/babytheorist/transfer_breakdown.py --split lv1 [--out file.json] runs/<run> [...]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from training.evaluate import load_run  # noqa: E402

GROUPS = ("same code", "different codes", "not covered", "not one code")


@torch.no_grad()
def breakdown(run: Path, split: str, checkpoint: Path | None, device: torch.device) -> dict:
    model, data, config, step = load_run(run, checkpoint, device)
    infer = next((dict(b.get("infer", {})) for b in config["evals"] if b["type"] == "babytheorist_observe"), {})
    counts: dict = collections.defaultdict(lambda: {g: [0, 0] for g in GROUPS})
    for batch in data.eval_loader(split, 500):
        keep = torch.nonzero(batch["depth"] == 1).flatten()
        if len(keep) == 0:
            continue
        sx, sy, qx, qy = (batch[k][keep].to(device) for k in ("support_x", "support_y", "query_x", "query_y"))
        transferred = model.infer(sx, sy, qx, None, **infer)["pred"].eq(qy).flatten(1).all(1).cpu()
        support, query = model.self_explain(sx, sy, **infer), model.self_explain(qx, qy, **infer)
        one = lambda out, target: (out["pred"].eq(target).flatten(1).all(1) & (out["n_steps"] == 1)).cpu()  # noqa: E731
        code = lambda out: out["indices"][0].view(len(keep), -1)[:, 0].cpu()  # noqa: E731
        s_one, q_one, s_code, q_code = one(support, sy), one(query, qy), code(support), code(query)
        for i, row in enumerate(keep.tolist()):
            primitive = json.loads(batch["transitions"][row])[0]["primitive"]
            if s_one[i] and q_one[i]:
                group = "same code" if int(s_code[i]) == int(q_code[i]) else "different codes"
            else:
                group = "not covered" if s_one[i] else "not one code"
            for name in (primitive, "all"):
                counts[name][group][0] += int(transferred[i])
                counts[name][group][1] += 1
    return {"run": run.name, "step": step, "by_primitive": {k: v for k, v in counts.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = []
    for run in args.runs:
        result = breakdown(run, args.split, args.ckpt, device)
        report.append(result)
        total = sum(v[1] for v in result["by_primitive"]["all"].values())
        print(f"{result['run']} step {result['step']}: {total} single-primitive tasks")
        for group in GROUPS:
            hit, n = result["by_primitive"]["all"][group]
            print(f"   {group:16s} {n:5d} tasks ({n / max(total, 1):5.1%})  transferred {hit / max(n, 1):6.1%}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
