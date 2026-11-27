#!/usr/bin/env python3
"""How a trained ARC run fails on a task file, from one reading of the demonstration pairs (weights frozen).

For every test: is the predicted canvas exact; if not, is the predicted shape right (the non-pad cells form the answer's
rectangle), and how many cells inside the answer are wrong. Tasks are split by whether the output keeps the input's shape
in all their demonstration pairs. Also the distribution of per-task scores, to see whether tasks are solved or not as wholes.

    python scripts/arc/failures.py --run runs/arc16_programmer_s0 --split heldout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "arc"))

from core.registry import DATASETS, build  # noqa: E402
from data.arc_tasks import PAD  # noqa: E402
from evals.arc import crop, predict, read_codes, task_mean  # noqa: E402
from evaluate import load_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, step = load_run(args.run, device)
    file = build(DATASETS, config["data"]).eval_tasks(args.split)
    tasks = len(file["names"])
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).long().to(device)  # noqa: E731
    d_task, t_task = tensor(file["demo_task"]), tensor(file["test_task"])
    codes = task_mean(read_codes(model, tensor(file["demo_inputs"]), tensor(file["demo_outputs"]), 256), d_task, tasks)
    guess = predict(model, tensor(file["test_inputs"]), codes[t_task], 256).cpu().numpy().astype(np.uint8)
    answer = file["test_outputs"].astype(np.uint8)
    same_shape = np.ones(tasks, dtype=bool)
    for x, y, t in zip(file["demo_inputs"], file["demo_outputs"], file["demo_task"]):
        same_shape[t] &= crop(x).shape == crop(y).shape
    exact = (guess == answer).reshape(len(guess), -1).all(axis=1)
    shape_right = np.array([((g != PAD) == (a != PAD)).all() for g, a in zip(guess, answer)])
    inside = answer != PAD
    wrong_cells = np.array([int(((g != a) & m).sum()) for g, a, m in zip(guess, answer, inside)])
    cells = inside.reshape(len(answer), -1).sum(axis=1)
    kept = same_shape[file["test_task"]]
    report = {"run": args.run.name, "step": step, "split": args.split, "tasks": tasks, "tests": int(len(guess))}
    for name, group in (("all", np.ones(len(guess), dtype=bool)), ("output keeps the input's shape", kept), ("output changes shape", ~kept)):
        n = int(group.sum())
        if not n:
            continue
        miss = group & ~exact
        report[name] = {
            "tests": n, "exact": round(float(exact[group].mean()), 4), "shape right": round(float(shape_right[group].mean()), 4),
            "of misses: shape wrong": round(float((~shape_right[miss]).mean()), 4) if miss.any() else None,
            "of misses with the right shape: wrong cells 1-2 / 3-10 / more": [
                round(float(((wrong_cells[miss & shape_right] >= lo) & (wrong_cells[miss & shape_right] <= hi)).mean()), 4) if (miss & shape_right).any() else None
                for lo, hi in ((1, 2), (3, 10), (11, 10**6))],
            "of misses with the right shape: median share of cells wrong": round(float(np.median(wrong_cells[miss & shape_right] / cells[miss & shape_right])), 4) if (miss & shape_right).any() else None,
        }
    per_task = np.bincount(file["test_task"], weights=exact.astype(float), minlength=tasks) / np.bincount(file["test_task"], minlength=tasks)
    report["tasks by score: 0 / between / 1"] = [int((per_task == 0).sum()), int(((per_task > 0) & (per_task < 1)).sum()), int((per_task == 1).sum())]
    report["tasks keeping shape"] = int(same_shape.sum())
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps({**report, "per_task": per_task.tolist(), "names": [str(n) for n in file["names"]]}, indent=1) + "\n")


if __name__ == "__main__":
    main()
