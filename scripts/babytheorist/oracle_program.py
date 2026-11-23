#!/usr/bin/env python3
"""How well does the executor run the hidden program itself? A diagnostic that reads the labels.

Each primitive is mapped to the code that most often reproduced it exactly on single-primitive tasks (the `exact_code`
histogram of the run's full evaluation). The hidden program of every task is written in those codes and executed on the
first support input and on the query input, the grid decoded and re-encoded between steps. The share of tasks whose
output is exactly right, by the length of the hidden program, is the ceiling a search over these codes could reach
with this executor, if it always proposed the right code. Comparing it with the search (support_search.py) separates two
causes of failure on long programs: the executor drifting over many steps, and the right code not being proposed.

    python scripts/babytheorist/oracle_program.py --split lv5 --eval-file data/babytheorist/single_a/lv5_eval.h5 \
        --max-samples 2000 --out runs/queue/oracle_program.json runs/<run> [...]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "babytheorist"))

from support_search import code_table, pad, run_programs  # noqa: E402
from training.evaluate import load_run  # noqa: E402


def primitive_codes(run: Path) -> dict[str, int]:
    for path in sorted(glob.glob(str(run / "eval" / "*full*_step*.json")), reverse=True):
        table = json.loads(Path(path).read_text()).get("one_step_tasks")
        if table:
            return {name: max(range(len(entry["exact_code"])), key=lambda c: entry["exact_code"][c]) for name, entry in table.items() if sum(entry["exact_code"])}
    raise SystemExit(f"{run}: no full evaluation with the single-primitive table")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--deepest", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = json.loads(args.out.read_text()) if args.out.exists() else []
    for run in args.runs:
        mapping = primitive_codes(run)
        model, data, config, step = load_run(run, None, device)
        model.eval()
        if args.eval_file:
            data.eval_paths[args.split] = str(args.eval_file)
        table = code_table(model)
        counts = torch.zeros(args.deepest)
        right = {"support": torch.zeros(args.deepest), "query": torch.zeros(args.deepest)}
        unmapped = 0
        for batch in data.eval_loader(args.split, 1024, max_samples=args.max_samples):
            programs = []
            for text in batch["transitions"]:
                names = [step["primitive"] for step in json.loads(text)]
                unmapped += sum(name not in mapping for name in names)
                programs.append([mapping.get(name, 0) for name in names])
            longest = max(len(p) for p in programs)
            codes = torch.tensor(pad(programs, longest), device=device)
            sx, sy = batch["supports_x"][:, 0].to(device), batch["supports_y"][:, 0].to(device)
            qx, qy = batch["query_x"].to(device), batch["query_y"].to(device)
            support = run_programs(model, table, sx, codes).eq(sy).flatten(1).all(dim=1).cpu()
            query = run_programs(model, table, qx, codes).eq(qy).flatten(1).all(dim=1).cpu()
            depth = batch["depth"].clamp(max=args.deepest) - 1
            for i in range(len(programs)):
                counts[depth[i]] += 1
                right["support"][depth[i]] += float(support[i])
                right["query"][depth[i]] += float(query[i])
        entry = {"run": run.name, "split": args.split, "mapping": mapping, "unmapped_steps": unmapped, "tasks_by_depth": counts.int().tolist(),
                 **{f"{k}_by_depth": [float(r / c) if c else None for r, c in zip(v, counts)] for k, v in right.items()}}
        report.append(entry)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(run.name, args.split, "query by depth", " ".join("-" if v is None else f"{v:.2f}" for v in entry["query_by_depth"]), flush=True)


if __name__ == "__main__":
    main()
