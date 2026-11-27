#!/usr/bin/env python3
"""Turn re-ARC and ARC-AGI-1 into the arrays the ARC experiments read.

A grid is placed at the top left of a square canvas of `--size` cells a side; the cells outside it hold the pad value 10
(colours are 0 to 9). Grids that do not fit are left out, and so are tasks that then keep too few examples.

Written to `--out`:
  rearc_train_<size>.npz    re-ARC examples of the ARC training tasks for training: inputs, outputs (N, size, size) uint8,
                            shapes (N, 4) = input height, width, output height, width, task (N,) index, names (tasks,)
  rearc_heldout_<size>.npz  for each of those tasks, `--heldout` re-ARC examples kept out of training, arranged as an
                            ARC task (three demonstration pairs and the rest as tests): the in-distribution check
  arc_training_<size>.npz   the original ARC-AGI-1 training tasks, demonstrations and tests
  arc_evaluation_<size>.npz the ARC-AGI-1 public evaluation tasks, demonstrations and tests
Task files hold demo_inputs, demo_outputs, demo_task, test_inputs, test_outputs, test_task, names, and for the training
tasks train_index (the row of the task in rearc_train, -1 if it was left out).

    python scripts/arc/prepare_arc.py --source /data/minjun_dev/toy-NEO/arc --out data/arc --size 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAD = 10


def canvas(grid, size: int) -> np.ndarray | None:
    g = np.asarray(grid, dtype=np.uint8)
    if g.ndim != 2 or g.shape[0] > size or g.shape[1] > size or g.size == 0:
        return None
    out = np.full((size, size), PAD, dtype=np.uint8)
    out[: g.shape[0], : g.shape[1]] = g
    return out


def pairs(examples, size: int):
    xs, ys, shapes = [], [], []
    for e in examples:
        x, y = canvas(e["input"], size), canvas(e["output"], size)
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        shapes.append((len(e["input"]), len(e["input"][0]), len(e["output"]), len(e["output"][0])))
    return xs, ys, shapes


def task_file(tasks: dict[str, dict], size: int, train_index: dict[str, int] | None = None) -> dict:
    """tasks: name -> {"train": [...], "test": [...]}; a task is kept only if every one of its grids fits."""
    out = {k: [] for k in ("demo_inputs", "demo_outputs", "demo_task", "test_inputs", "test_outputs", "test_task", "names", "train_index")}
    for name, task in tasks.items():
        dx, dy, _ = pairs(task["train"], size)
        tx, ty, _ = pairs(task["test"], size)
        if len(dx) != len(task["train"]) or len(tx) != len(task["test"]) or len(dx) < 2 or not tx:
            continue
        t = len(out["names"])
        out["names"].append(name)
        out["train_index"].append(-1 if train_index is None else train_index.get(name, -1))
        out["demo_inputs"] += dx; out["demo_outputs"] += dy; out["demo_task"] += [t] * len(dx)
        out["test_inputs"] += tx; out["test_outputs"] += ty; out["test_task"] += [t] * len(tx)
    return {k: np.asarray(v) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="folder holding re-arc/re_arc/tasks and ARC-AGI/data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--heldout", type=int, default=8, help="re-ARC examples of each task kept out of training")
    parser.add_argument("--min-examples", type=int, default=200, help="a task needs this many fitting re-ARC examples to be trained on")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs, outputs, shapes, task, names, heldout = [], [], [], [], [], {}
    for path in sorted((args.source / "re-arc" / "re_arc" / "tasks").glob("*.json")):
        xs, ys, sh = pairs(json.loads(path.read_text()), args.size)
        if len(xs) < args.min_examples + args.heldout:
            continue
        t = len(names)
        names.append(path.stem)
        keep = len(xs) - args.heldout
        inputs += xs[:keep]; outputs += ys[:keep]; shapes += sh[:keep]; task += [t] * keep
        held = [{"input": x[: s[0], : s[1]].tolist(), "output": y[: s[2], : s[3]].tolist()} for x, y, s in zip(xs[keep:], ys[keep:], sh[keep:])]
        heldout[path.stem] = {"train": held[:3], "test": held[3:]}
    np.savez_compressed(args.out / f"rearc_train_{args.size}.npz", inputs=np.asarray(inputs), outputs=np.asarray(outputs),
                        shapes=np.asarray(shapes, dtype=np.uint8), task=np.asarray(task, dtype=np.int32), names=np.asarray(names))
    index = {name: i for i, name in enumerate(names)}
    np.savez_compressed(args.out / f"rearc_heldout_{args.size}.npz", **task_file(heldout, args.size, index))
    report = {"size": args.size, "train_tasks": len(names), "train_examples": len(inputs)}
    for split in ("training", "evaluation"):
        tasks = {p.stem: json.loads(p.read_text()) for p in sorted((args.source / "ARC-AGI" / "data" / split).glob("*.json"))}
        data = task_file(tasks, args.size, index if split == "training" else None)
        np.savez_compressed(args.out / f"arc_{split}_{args.size}.npz", **data)
        report[f"{split}_tasks"] = f"{len(data['names'])} of {len(tasks)}"
        report[f"{split}_tests"] = int(len(data["test_task"]))
    (args.out / f"prepared_{args.size}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(report)


if __name__ == "__main__":
    main()
