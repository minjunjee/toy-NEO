#!/usr/bin/env python3
"""A long-program eval file for a primitive set: three pairs per task (two supports and a query), hidden lengths drawn from
--depths, every pair's program the shortest explanation of its own result (as in generate_many_primitives.py). Pairs that
appear in --exclude-from (a train file) are left out. Only the eval file is written, so no train file is generated first.

    python scripts/babytheorist/generate_long_eval.py --primitives 24 --depths 5:1,6:1,7:1 --n-eval 1500 \
        --exclude-from data/babytheorist/many_p24_pairs2/lv1_train.h5 --out data/babytheorist/many_p24_long
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_many_primitives import SETS, generate, pair_hash, write  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primitives", choices=sorted(SETS), required=True)
    parser.add_argument("--depths", required=True, help='e.g. "5:1,6:1,7:1"')
    parser.add_argument("--n-eval", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--exclude-from", type=Path, nargs="*", default=[])
    parser.add_argument("--large", action="store_true", help="objects of 16 to 29 cells spanning up to 7, larger than any in training")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parts = [part.split(":") for part in args.depths.split(",")]
    weights = {int(p[0]): float(p[1]) if len(p) > 1 else 1.0 for p in parts}
    depths = {d: w / sum(weights.values()) for d, w in weights.items()}
    exclude = set()
    for path in args.exclude_from:
        with h5py.File(path, "r") as f:
            ins, outs = f["inputs"][:], f["outputs"][:]
        exclude |= {pair_hash(ins[i, j].astype(np.int8), outs[i, j].astype(np.int8)) for i in range(len(ins)) for j in range(ins.shape[1])}
    args.out.mkdir(parents=True, exist_ok=True)
    data = generate(SETS[args.primitives], args.n_eval, 3, args.seed, args.workers, exclude, depths, large=args.large)
    write(args.out / "lv1_eval.h5", SETS[args.primitives], data, 3)


if __name__ == "__main__":
    main()
