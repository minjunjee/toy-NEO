#!/usr/bin/env python3
"""A label-free score for choosing among runs: on held-out tasks of two pairs, the share whose code, named by the
programmer from the first pair, reproduces both pairs. The tasks mix programs of one and of several primitives, so the
score is low in absolute terms; it is only compared between runs. Neither the hidden program nor its depth is read.

    python scripts/babytheorist/pair_validation.py --file data/babytheorist/many_p24_pairs2_valid/lv1_train.h5 runs/<run> [...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from training.evaluate import load_run  # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(args.file, "r") as f:
        x = torch.from_numpy(f["inputs"][:].astype(np.int64)).to(device)
        y = torch.from_numpy(f["outputs"][:].astype(np.int64)).to(device)
    report = []
    for run in args.runs:
        model, data, config, step = load_run(run, None, device)
        model.eval()
        table = F.normalize(model.code.embedding.weight.float(), dim=-1)
        hits = 0
        for i in range(0, len(x), 500):
            sx, sy = x[i:i + 500], y[i:i + 500]
            code = (F.normalize(model.programmer(model.encoder(sx[:, 0]), model.encoder(sy[:, 0])).flatten(1).float(), dim=-1) @ table.T).argmax(dim=1)
            ok = torch.ones(len(sx), dtype=torch.bool, device=device)
            for j in range(sx.shape[1]):
                out = model.decoder(model.executor(model.encoder(sx[:, j]), table[code][:, None])).argmax(dim=-1)
                ok &= out.eq(sy[:, j]).flatten(1).all(dim=1)
            hits += int(ok.sum())
        report.append({"run": run.name, "both_pairs": hits / len(x)})
        print(f"{run.name:44s} both pairs reproduced {hits / len(x):.4f}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
