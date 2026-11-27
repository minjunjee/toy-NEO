#!/usr/bin/env python3
"""Vote several per-task sleep results together: for every test the views of all runs are counted as one ballot box.

Each result file (scripts/arc/sleep_per_task.py) holds, per test, its five most frequent guesses with the number of views
behind each. The counts of the same grid are added over the files (optionally weighted), and the most frequent grid
(vote1) or the two most frequent (vote2) are scored. Only tasks present in every file are used.

    python scripts/arc/vote_together.py runs/a/per_task.json runs/b/per_task.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--weights", nargs="*", type=float)
    args = parser.parse_args()
    runs = [{task["index"]: task for task in json.loads(path.read_text())["per_task"]} for path in args.files]
    weights = args.weights or [1.0] * len(runs)
    shared = sorted(set.intersection(*(set(run) for run in runs)))
    one, two = [], []
    for index in shared:
        solved1, solved2 = [], []
        for test in range(len(runs[0][index]["ballots"])):
            box: Counter = Counter()
            right = set()
            for run, weight in zip(runs, weights):
                for digest, count, is_answer in run[index]["ballots"][test]:
                    box[digest] += weight * count
                    if is_answer:
                        right.add(digest)
            top = [digest for digest, _ in box.most_common(2)]
            solved1.append(bool(top) and top[0] in right)
            solved2.append(any(digest in right for digest in top))
        one.append(np.mean(solved1))
        two.append(np.mean(solved2))
    for path, run in zip(args.files, runs):
        print(f"{path}: vote1 {100 * np.mean([run[i]['vote1'] for i in shared]):.1f} vote2 {100 * np.mean([run[i]['vote2'] for i in shared]):.1f}")
    print(f"together on {len(shared)} tasks: vote1 {100 * np.mean(one):.1f} vote2 {100 * np.mean(two):.1f}")


if __name__ == "__main__":
    main()
