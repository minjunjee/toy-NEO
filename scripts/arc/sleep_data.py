#!/usr/bin/env python3
"""Write the train file of a sleep: the demonstration pairs of tasks from an ARC task file, in the format of the re-ARC
train file, so they can be mixed with replayed training tasks (data `arc_tasks` with several train files).

Only the demonstration pairs are written; test inputs and answers are never read. With `--failed` and the detail file of a
frozen evaluation (scripts/arc/evaluate.py --detail), only tasks that failed a check that needs no answers are kept:
`verify` (some demonstration pair is not reproduced from the code of the task's other pairs) or `fit_demo` (not all are
reproduced by the fitted code).

    python scripts/arc/sleep_data.py --tasks data/arc/arc_evaluation_16.npz --out data/arc/sleep_evaluation_16.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAD = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--failed", type=Path, help="detail file of a frozen evaluation of the same task file")
    parser.add_argument("--check", choices=("verify", "fit_demo"), default="verify")
    args = parser.parse_args()
    with np.load(args.tasks) as f:
        x, y, task, names = f["demo_inputs"], f["demo_outputs"], f["demo_task"].astype(np.int64), f["names"]
    keep = np.ones(len(names), dtype=bool)
    if args.failed:
        detail = json.loads(args.failed.read_text())["detail"]
        if detail["names"] != [str(n) for n in names[: len(detail["names"])]]:
            raise ValueError("the detail file is of another task file")
        keep[: len(detail["names"])] = ~np.asarray(detail[args.check], dtype=bool)
    rows = keep[task]
    renumber = np.cumsum(keep) - 1
    shape = lambda grids: np.stack([(grids[:, :, 0] != PAD).sum(axis=1), (grids[:, 0, :] != PAD).sum(axis=1)], axis=1)  # noqa: E731
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, inputs=x[rows], outputs=y[rows], shapes=np.concatenate([shape(x[rows]), shape(y[rows])], axis=1).astype(np.uint8),
                        task=renumber[task[rows]].astype(np.int32), names=names[keep])
    print(f"{args.out}: {int(keep.sum())} of {len(names)} tasks, {int(rows.sum())} pairs")


if __name__ == "__main__":
    main()
