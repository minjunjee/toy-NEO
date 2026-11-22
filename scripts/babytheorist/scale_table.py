#!/usr/bin/env python3
"""How a model holds up as the environment grows: one line per run, split by what kind of task it is.

For every eval task the program is extracted from the first support pair and replayed on the query (the answer is not
shown). Two ways of writing the program are measured:

  own       the model's own form: one code for the lab one-step model, one code and a count for neo_repeat / neo_counted
  stepwise  the lab NEO's rollout with the same parts: the programmer looks again after every single application of the
            executor and picks a code, up to --max-steps applications, the grid handed between them, and the lab's
            length rule picks where to stop. This is the only way these models can write a task made of several
            different primitives, and no model here was trained on such a task.

Tasks are grouped by the depth of their hidden program (1 = one primitive) and by whether the program uses a primitive
that the train file did not have (--held-out, matched against the end of the primitive's name). The grouping only labels
the rows; nothing of it reaches the model.

    python scripts/babytheorist/scale_table.py runs/ladder_repeat_p52_s42 [...] [--held-out _4] [--max-steps 12] [--out file.json]
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

SHIFT_FOUR = ("shift_up_4", "shift_down_4", "shift_left_4", "shift_right_4")


@torch.no_grad()
def measure(run: Path, device: torch.device, split: str, max_steps: int, max_tasks: int, held_out: tuple[str, ...]) -> dict:
    model, data, config, step = load_run(run, None, device)
    counted = config["model"]["type"] in ("neo_repeat", "neo_counted")
    own = {"max_steps": model.max_length, "rollout": "hard"} if counted else {"max_steps": 1, "rollout": "carry"}
    stepwise = {"max_steps": max_steps, "rollout": "hard"}
    totals = collections.defaultdict(lambda: collections.Counter())
    seen = 0
    for batch in data.eval_loader(split, 500):
        if seen >= max_tasks:
            break
        sx, sy, qx, qy = (batch[k].to(device) for k in ("support_x", "support_y", "query_x", "query_y"))
        seen += len(sx)
        results = {}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for name, settings, replan in (("own", own, False), ("stepwise", stepwise, True)):
                if counted:
                    model.replan = replan
                explained = model.self_explain(sx, sy, **settings)
                answered = model.infer(sx, sy, qx, **settings)
                results[name] = (explained["pred"].eq(sy).flatten(1).all(dim=1).cpu(), answered["pred"].eq(qy).flatten(1).all(dim=1).cpu(), explained["n_steps"].cpu())
        for i in range(len(sx)):
            program = [t["primitive"] for t in json.loads(batch["transitions"][i])]
            depth = min(int(batch["depth"][i]), 4)
            group = f"depth {depth}" + (", uses a primitive left out of training" if any(p in held_out for p in program) else "")
            row = totals[group]
            row["tasks"] += 1
            for name, (explain, transfer, steps) in results.items():
                row[f"{name}_explain"] += int(explain[i])
                row[f"{name}_transfer"] += int(transfer[i])
                row[f"{name}_steps"] += int(steps[i])
    groups = {g: {k: (v / row["tasks"] if k != "tasks" else v) for k, v in row.items()} for g, row in sorted(totals.items())}
    return {"run": run.name, "step": step, "model": config["model"]["type"], "codes": int(model.code.size), "groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--held-out", nargs="*", default=list(SHIFT_FOUR))
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-tasks", type=int, default=6000)
    parser.add_argument("--split", default="lv1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = []
    for run in args.runs:
        entry = measure(run, device, args.split, args.max_steps, args.max_tasks, tuple(args.held_out))
        report.append(entry)
        print(f"{entry['run']}  ({entry['model']}, {entry['codes']} codes, step {entry['step']})", flush=True)
        for group, r in entry["groups"].items():
            print(f"  {group:48s} {int(r['tasks']):5d} tasks   own: explain {r['own_explain']:.3f} transfer {r['own_transfer']:.3f}   "
                  f"stepwise: explain {r['stepwise_explain']:.3f} transfer {r['stepwise_transfer']:.3f} (mean applications {r['stepwise_steps']:.1f})", flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
