#!/usr/bin/env python3
"""Per primitive: what the programmer's code does, and what the best code in the codebook could do.

For runs on the family of fixed shifts (shift_<direction>_<cells>), on the single-primitive eval tasks. Per primitive:

  explain    share of tasks whose first support pair is reproduced by the code the programmer picks for it
  transfer   share of tasks whose query is answered by that code (the answer is not shown)
  picked     the code picked most often, as kind:amount when the codebook has that index, and its share; for a model
             whose program is one code and a count (neo_repeat), the code and count picked most often, and explain and
             transfer are those of that program (the two-steps columns then repeat them)
  best code  the single code that reproduces most support pairs of this primitive when the executor is simply given it,
             its share on the support pairs and on the queries: can the executor do this primitive with some code at all
  search     per task, try every code on the support pair and keep the first that reproduces it: share of tasks where
             one does, and share of tasks where that code also answers the query
  repeat     try every code applied 1 to --repeats times in a row, the grid handed between the applications, and keep
             the first (fewest applications) that reproduces the support pair: share of tasks where one does, share
             where that code and count also answer the query, and the (code x count) that reproduces most support
             pairs of the primitive
  two steps  the same model allowed a program of up to two codes, the grid handed between the steps: explain and
             transfer. This is what the model can already do for an amount it has no code for, by joining two it has

The primitive's name only groups the rows. `--held-out` names the amounts that the train file did not have.

    python scripts/babytheorist/amount_table.py runs/onestep_k_rot_add_s42 [...] [--held-out 4] [--out file.json]
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
def table_for(run: Path, device: torch.device, per_primitive: int, split: str, repeats: int) -> dict:
    model, data, config, step = load_run(run, None, device)
    counted = config["model"]["type"] == "neo_repeat"   # its program is one code and a count: explain and transfer use that
    own = {"max_steps": model.max_length, "rollout": "hard"} if counted else {"max_steps": 1, "rollout": "carry"}
    tasks = collections.defaultdict(lambda: collections.defaultdict(list))
    for batch in data.eval_loader(split, 1000):
        for i in torch.nonzero(batch["depth"] == 1).flatten().tolist():
            name = json.loads(batch["transitions"][i])[0]["primitive"]
            if len(tasks[name]["support_x"]) < per_primitive:
                for key in ("support_x", "support_y", "query_x", "query_y"):
                    tasks[name][key].append(batch[key][i])
    codes = model.code.codebook(torch.arange(model.code.size, device=device)).float()
    indexed = hasattr(model.code, "kind_and_amount")

    def label(index: int) -> str:
        if not indexed:
            return str(index)
        kind, amount = model.code.kind_and_amount(torch.tensor(index, device=device))
        return f"{int(kind)}:{int(amount)}"

    def run_code(x: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return model.decoder(model.executor(model.encoder(x), vectors[:, None])).argmax(dim=-1)

    rows = {}
    for name in sorted(tasks):
        sx, sy, qx, qy = (torch.stack(tasks[name][key]).to(device) for key in ("support_x", "support_y", "query_x", "query_y"))
        n = len(sx)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            explained = model.self_explain(sx, sy, **own)
            answered = model.infer(sx, sy, qx, **own)
            explained_2 = explained if counted else model.self_explain(sx, sy, max_steps=2, rollout="hard")
            answered_2 = answered if counted else model.infer(sx, sy, qx, max_steps=2, rollout="hard")
        picked = explained["indices"][0].view(n, -1)[:, 0]
        # [task, code, applications - 1]: does the code applied that many times in a row give the answer
        support_many = torch.zeros(n, len(codes), repeats, dtype=torch.bool, device=device)
        query_many = torch.zeros(n, len(codes), repeats, dtype=torch.bool, device=device)
        for c, vector in enumerate(codes):
            vectors = vector.expand(n, -1)
            s_grid, q_grid = sx, qx
            for t in range(repeats):
                s_grid, q_grid = run_code(s_grid, vectors), run_code(q_grid, vectors)
                support_many[:, c, t] = s_grid.eq(sy).flatten(1).all(dim=1)
                query_many[:, c, t] = q_grid.eq(qy).flatten(1).all(dim=1)
        support_ok, query_ok = support_many[:, :, 0], query_many[:, :, 0]
        by_count = support_many.transpose(1, 2).reshape(n, -1)               # fewest applications first, then the code's number
        found_many = by_count.any(dim=1)
        first_many = by_count.float().argmax(dim=1)
        query_by_count = query_many.transpose(1, 2).reshape(n, -1)
        rate = support_many.float().mean(dim=0)                              # [code, applications - 1]
        best_many = int(rate.flatten().argmax())
        best_c, best_t = best_many // repeats, best_many % repeats
        best = int(support_ok.float().mean(dim=0).argmax())
        found = support_ok.any(dim=1)
        first = support_ok.float().argmax(dim=1)
        most, count = collections.Counter(zip(picked.tolist(), explained["n_steps"].tolist())).most_common(1)[0] if counted else collections.Counter(picked.tolist()).most_common(1)[0]
        times = ""
        if counted:
            most, times = most[0], f" x{most[1]}"
        rows[name] = {
            "tasks": n,
            "explain": float(explained["pred"].eq(sy).flatten(1).all(dim=1).float().mean()),
            "transfer": float(answered["pred"].eq(qy).flatten(1).all(dim=1).float().mean()),
            "picked": label(most) + times, "picked_share": count / n,
            "best_code": label(best), "best_support": float(support_ok[:, best].float().mean()), "best_query": float(query_ok[:, best].float().mean()),
            "search_explain": float(found.float().mean()),
            "search_transfer": float((found & query_ok[torch.arange(n, device=device), first]).float().mean()),
            "repeat_explain": float(found_many.float().mean()),
            "repeat_transfer": float((found_many & query_by_count[torch.arange(n, device=device), first_many]).float().mean()),
            "repeat_best": f"{label(best_c)} x{best_t + 1}", "repeat_best_support": float(rate[best_c, best_t]),
            "repeat_best_query": float(query_many[:, best_c, best_t].float().mean()),
            "two_step_explain": float(explained_2["pred"].eq(sy).flatten(1).all(dim=1).float().mean()),
            "two_step_transfer": float(answered_2["pred"].eq(qy).flatten(1).all(dim=1).float().mean()),
            "two_step_used": float((explained_2["n_steps"] == 2).float().mean()),
        }
    return {"run": run.name, "step": step, "indexed": indexed, "primitives": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--held-out", nargs="*", default=["4"], help="amounts (the number at the end of the primitive's name) missing from the train file")
    parser.add_argument("--per-primitive", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=6, help="how many times in a row one code is applied at most")
    parser.add_argument("--split", default="lv1")
    parser.add_argument("--quiet", action="store_true", help="print only the two summary lines per run")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = []
    keys = ("explain", "transfer", "best_support", "best_query", "search_explain", "search_transfer", "repeat_explain", "repeat_transfer", "repeat_best_support", "repeat_best_query",
            "two_step_explain", "two_step_transfer", "two_step_used")
    for run in args.runs:
        entry = table_for(run, device, args.per_primitive, args.split, args.repeats)
        groups = {"trained": [], "held out": []}
        for name, row in entry["primitives"].items():
            groups["held out" if name.rsplit("_", 1)[-1] in args.held_out else "trained"].append(row)
        entry["summary"] = {group: {key: sum(r[key] for r in rows) / len(rows) for key in keys} for group, rows in groups.items() if rows}
        report.append(entry)
        print(f"{entry['run']}  step {entry['step']}")
        if not args.quiet:
            print(f"  {'primitive':16s} explain transfer  picked (share)   best code  on support  on query   search explain  search transfer   repeat explain transfer  best code x count (support, query)   "
                  f"two steps explain transfer (share using 2)")
            for name, r in entry["primitives"].items():
                mark = "*" if name.rsplit("_", 1)[-1] in args.held_out else " "
                print(f" {mark}{name:16s}  {r['explain']:.2f}    {r['transfer']:.2f}   {r['picked']:>9s} ({r['picked_share']:.2f})     {r['best_code']:>6s}      {r['best_support']:.2f}       {r['best_query']:.2f}         "
                      f"{r['search_explain']:.2f}            {r['search_transfer']:.2f}              {r['repeat_explain']:.2f}    {r['repeat_transfer']:.2f}    {r['repeat_best']:>10s} ({r['repeat_best_support']:.2f}, {r['repeat_best_query']:.2f})"
                      f"                 {r['two_step_explain']:.2f}     {r['two_step_transfer']:.2f}    ({r['two_step_used']:.2f})")
        for group, values in entry["summary"].items():
            print(f"  {group:9s} mean: explain {values['explain']:.3f}  transfer {values['transfer']:.3f}  best code on support {values['best_support']:.3f} on query {values['best_query']:.3f}  "
                  f"search explain {values['search_explain']:.3f} transfer {values['search_transfer']:.3f}\n{'':17s}repeat search explain {values['repeat_explain']:.3f} transfer {values['repeat_transfer']:.3f}  "
                  f"best code x count on support {values['repeat_best_support']:.3f} on query {values['repeat_best_query']:.3f}  two steps explain {values['two_step_explain']:.3f} transfer {values['two_step_transfer']:.3f}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
