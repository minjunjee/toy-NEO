#!/usr/bin/env python3
"""Give new tasks codes of their own before a sleep: a checkpoint whose table holds one code for every new task and view.

The code of a task seen through a view is what the frozen model would use for it: the mean of the programmer's codes of
the view's demonstration pairs, optionally moved by `--fit-steps` steps of Adam on those pairs. The weights are copied
unchanged; a table the run already had is replaced, since the sleep that follows names only the new tasks. Test inputs
and answers are never read.

    python scripts/arc/grow_table.py --run runs/arc16_programmer_s0 --tasks data/arc/sleep_evaluation_16.npz --out runs/arc16_programmer_s0/ckpt/grown_evaluation.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "arc"))

from core.registry import DATASETS, MODELS, build  # noqa: E402
from evals.arc import fit_codes, read_codes, task_mean, view  # noqa: E402
from evaluate import load_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tasks", type=Path, required=True, help="train-format file of the new tasks (scripts/arc/sleep_data.py)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit-steps", type=int, default=0)
    parser.add_argument("--fit-lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--color-perms", type=int, help="colour permutations of the table's views, when fewer than the run was trained with")
    parser.add_argument("--pairs", type=int, help="read at most this many pairs of each task (the first ones), for train files with many examples")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, step = load_run(args.run, device, args.checkpoint)
    data = build(DATASETS, {**config["data"], "evals": {}, **({"color_perms": args.color_perms} if args.color_perms else {})})
    with np.load(args.tasks) as f:
        x, y, owner = f["inputs"], f["outputs"], f["task"].astype(np.int64)
        tasks = len(f["names"])
    if args.pairs:
        rank = np.zeros(len(owner), dtype=np.int64)          # the position of every pair among its task's pairs
        order = np.argsort(owner, kind="stable")
        rank[order] = np.arange(len(owner)) - np.searchsorted(owner[order], owner[order])
        x, y, owner = x[rank < args.pairs], y[rank < args.pairs], owner[rank < args.pairs]
    task = torch.from_numpy(owner).to(device)
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).long().to(device)  # noqa: E731
    table = torch.zeros(tasks, data.variants, model.code_dim)
    for color, colors in enumerate(data.tables):
        for symmetry in range(data.symmetries):
            vx, vy = tensor(view(x, symmetry, colors)), tensor(view(y, symmetry, colors))
            code = task_mean(read_codes(model, vx, vy, args.batch_size), task, tasks)
            if args.fit_steps:
                code = fit_codes(model, vx, vy, task, code, args.fit_steps, args.fit_lr, args.batch_size)
            table[:, color * data.symmetries + symmetry] = code.cpu()
    spec = {**config["model"], "tasks": tasks, "variants": data.variants}
    grown = build(MODELS, spec)
    state = {key: value for key, value in model.state_dict().items() if not key.startswith("table.")}
    state["table.weight"] = table.flatten(0, 1)
    grown.load_state_dict(state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": grown.state_dict(), "config": {**config, "model": spec, "data": {**config["data"], "color_perms": len(data.tables)}}}, args.out)
    print(f"{args.out}: {tasks} tasks x {data.variants} views, mean code length {float(table.norm(dim=-1).mean()):.2f}")


if __name__ == "__main__":
    main()
