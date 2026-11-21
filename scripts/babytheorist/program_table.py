#!/usr/bin/env python3
"""Which programs the model writes for single-primitive tasks, and which of them carry over to a new input.

For every eval task whose hidden program is one primitive, the program is extracted from the first support pair. If it
reproduces that pair exactly, the code sequence is counted under the task's primitive together with whether replaying it
on the query gives the right answer (answer not shown). The table shows, per primitive, the most frequent sequences.
The primitive's name only labels the rows; nothing here feeds it to the model.

With --rollout hard the state handed from one program step to the next is the encoding of the decoded grid (argmax), so a
step can only see what is visible on the grid. The runs are trained and normally evaluated with the soft re-encoding, which
also passes how sure each cell was; comparing the two shows which programs lean on that extra information.

    python scripts/babytheorist/program_table.py --split lv1 [--top 6] [--rollout hard] [--out file.json] runs/<run> [...]
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


@torch.no_grad()
def programs(model, data, split: str, infer: dict, device: torch.device) -> dict:
    table: dict = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    tasks: dict = collections.Counter()
    for batch in data.eval_loader(split, 500):
        keep = torch.nonzero(batch["depth"] == 1).flatten()
        if len(keep) == 0:
            continue
        sx, sy, qx, qy = (batch[k][keep].to(device) for k in ("support_x", "support_y", "query_x", "query_y"))
        transferred = model.infer(sx, sy, qx, None, **infer)["pred"].eq(qy).flatten(1).all(1).cpu()
        support = model.self_explain(sx, sy, **infer)
        exact = support["pred"].eq(sy).flatten(1).all(1).cpu()
        # the model returns one index tensor per program step; a program is the first n_steps of them
        codes, lengths = torch.stack([step.view(len(keep), -1)[:, 0] for step in support["indices"]], dim=1).cpu(), support["n_steps"].cpu()
        for i, row in enumerate(keep.tolist()):
            primitive = json.loads(batch["transitions"][row])[0]["primitive"]
            tasks[primitive] += 1
            if exact[i]:
                sequence = " ".join(str(int(c)) for c in codes[i, : int(lengths[i])])
                table[primitive][sequence][0] += int(transferred[i])
                table[primitive][sequence][1] += 1
    return {p: {"tasks": tasks[p], "programs": dict(sorted(table[p].items(), key=lambda kv: -kv[1][1]))} for p in sorted(tasks)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--rollout", help="override the evaluation's rollout setting, e.g. hard")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = []
    for run in args.runs:
        model, data, config, step = load_run(run, args.ckpt, device)
        infer = next((dict(b.get("infer", {})) for b in config["evals"] if b["type"] == "babytheorist_observe"), {})
        if args.rollout:
            infer["rollout"] = args.rollout
        result = {"run": run.name, "step": step, "infer": infer, "by_primitive": programs(model, data, args.split, infer, device)}
        report.append(result)
        tasks = sum(e["tasks"] for e in result["by_primitive"].values())
        reproduced = sum(n for e in result["by_primitive"].values() for _, n in e["programs"].values())
        carried = sum(hit for e in result["by_primitive"].values() for hit, _ in e["programs"].values())
        print(f"{run.name} step {step} {infer}: {tasks} single-primitive tasks, reproduced {reproduced / tasks:.1%}, reproduced and transferred {carried / tasks:.1%}")
        for primitive, entry in result["by_primitive"].items():
            reproduced = sum(n for _, n in entry["programs"].values())
            print(f"  {primitive}: {entry['tasks']} tasks, {reproduced} reproduced, {len(entry['programs'])} distinct programs")
            for sequence, (hit, n) in list(entry["programs"].items())[: args.top]:
                print(f"      [{sequence}]  {n:4d} tasks  transferred {hit / n:6.1%}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
