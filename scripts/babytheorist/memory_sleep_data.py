#!/usr/bin/env python3
"""Files for a sleep that consolidates an online fast memory (memory_online.py --save-memory) into the weights.

From the saved memory (or several, from successive parts of one stream): the tasks of the stream that no code of the frozen model reproduced (the tasks that went into the
new codes' memory) become a two-pair train file, their two support pairs only, repeated to `--rows` rows so that, mixed
with a replay file of the same size, half of each batch is memory and half is the old world. The eval file of the stream
is copied without the stream's rows, so the consolidated model is scored on tasks it never saw. No label is read to pick
the tasks; the hidden primitive is copied along only because the file format carries it.

    python scripts/babytheorist/memory_sleep_data.py --memory runs/queue/memory_online_refit.pt \
        --train-out data/babytheorist/memory_refit_s43_rep/lv1_train.h5 \
        --eval-out data/babytheorist/many_p52_without4_unstreamed/lv1_eval.h5 --rows 40000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--memory", type=Path, nargs="+", required=True, help="one saved memory, or several of the same eval file (their tasks are joined)")
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--eval-out", type=Path)
    parser.add_argument("--rows", type=int, default=40000)
    args = parser.parse_args()
    loaded = [torch.load(path, map_location="cpu", weights_only=False) for path in args.memory]
    assert len({m["eval"] for m in loaded}) == 1, "the memories must come from the same eval file"
    saved = {"eval": loaded[0]["eval"], "rows": sorted({r for m in loaded for r in m["rows"]})}
    rows = sorted({r for m in loaded for code in m["memory_rows"] for r in code})
    with h5py.File(saved["eval"], "r") as f:
        data = {k: f[k][:] for k in f.keys()}
    reps = int(np.ceil(args.rows / len(rows)))
    pick = np.tile(np.array(rows), reps)[: args.rows]
    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.train_out, "w") as out:
        for key, value in data.items():
            if key in ("inputs", "outputs", "n_shapes"):
                out[key] = value[pick][:, :2]
            elif key == "n_support":
                out[key] = np.full(len(pick), 1, dtype=value.dtype)   # as in the two-pair train files of generate_many_primitives.py
            elif key == "transformation_suites":
                out.create_dataset(key, data=value[pick], dtype=h5py.string_dtype())
            else:
                out[key] = value[pick]
    print(f"{len(rows)} memory tasks, repeated to {len(pick)} rows: {args.train_out}")
    if args.eval_out:
        keep = np.setdiff1d(np.arange(len(data["depths"])), np.array(saved["rows"]))
        args.eval_out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.eval_out, "w") as out:
            for key, value in data.items():
                if key == "transformation_suites":
                    out.create_dataset(key, data=value[keep], dtype=h5py.string_dtype())
                else:
                    out[key] = value[keep]
        print(f"eval without the {len(saved['rows'])} stream rows: {len(keep)} tasks, {args.eval_out}")


if __name__ == "__main__":
    main()
