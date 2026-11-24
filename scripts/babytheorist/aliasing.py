#!/usr/bin/env python3
"""How often does a train task fail to tell two primitives apart? A diagnostic that reads the hidden program.

For every single-primitive task of a train file with several pairs, every other primitive of the set is applied to each
pair's input. A primitive is an alias of the task's primitive on that task when it reproduces every pair of the task.
Reported per primitive: the share of its tasks that have at least one alias, and the aliases met most often, for the
first pair alone and for all the task's pairs.

    python scripts/babytheorist/aliasing.py data/babytheorist/many_p24_pairs2/lv1_train.h5 --primitives 24 --tasks 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_many_primitives import SETS, function_of, step, tight  # noqa: E402


def object_of(grid: np.ndarray):
    return tight(grid, (0, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train", type=Path)
    parser.add_argument("--primitives", required=True)
    parser.add_argument("--tasks", type=int, default=8000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    names = SETS[args.primitives]
    functions = {n: function_of(n) for n in names}
    with h5py.File(args.train, "r") as f:
        ins, outs = f["inputs"][: args.tasks], f["outputs"][: args.tasks]
        programs = [json.loads(p) for p in f["transformation_suites"][: args.tasks]]
    first, every = defaultdict(Counter), defaultdict(Counter)
    tasks = Counter()
    for i, program in enumerate(programs):
        if len(program) != 1:
            continue
        name = program[0]["primitive"]
        tasks[name] += 1
        for other in names:
            if other == name:
                continue
            ok_first, ok_all = None, True
            for j in range(ins.shape[1]):
                made = step(object_of(ins[i, j]), functions[other])
                same = made is not None and np.array_equal(made[1], outs[i, j])
                if j == 0:
                    ok_first = same
                ok_all &= same
            if ok_first:
                first[name][other] += 1
            if ok_all:
                every[name][other] += 1
    report = {}
    for name in sorted(tasks):
        any_first = sum(first[name].values()) and None
        report[name] = {"tasks": tasks[name], "aliases_first_pair": dict(first[name].most_common(3)), "aliases_all_pairs": dict(every[name].most_common(3))}
        print(f"{name:16s} {tasks[name]:5d} first pair {dict(first[name].most_common(3))} | all pairs {dict(every[name].most_common(3))}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
