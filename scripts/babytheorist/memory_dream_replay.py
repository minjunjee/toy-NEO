#!/usr/bin/env python3
"""A replay file for the sleep made by the model itself, in place of the old world's train file.

The frozen model dreams: for each row a code of its own codebook is drawn uniformly, two grids are drawn from the grids it
saw while awake (the inputs and outputs of the stream's tasks, saved by memory_online.py --save-memory), and the executor
runs the code on both. The two (grid, result) pairs are one two-pair task of the replay file. Training on it asks the
sleeping model to keep doing with each code what it did before the sleep, on the states of the new world; no stored data
of the old world and no label is used. Every code is dreamt, used or not.

    python scripts/babytheorist/memory_dream_replay.py --run runs/onestep_s_m24_hardem_pairs2_price_s43 \
        --memory runs/queue/memory_online_refit.pt --out data/babytheorist/dream_replay_s43/lv1_train.h5 --rows 40000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "babytheorist"))

from support_search import code_table  # noqa: E402
from training.evaluate import load_run  # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--memory", type=Path, nargs="+", required=True, help="saved memories of the wakes so far (their streams' grids are pooled)")
    parser.add_argument("--rows", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--by-use", action="store_true", help="draw each code in proportion to how often it was the code that answered a task of the "
                        "stream (the first code in the programmer's order that reproduces both supports), plus one, instead of uniformly")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _ = load_run(args.run, None, device)
    model.eval()
    table = code_table(model)
    loaded = [torch.load(path, map_location="cpu", weights_only=False) for path in args.memory]
    saved = {"eval": loaded[0]["eval"], "rows": [r for m in loaded for r in m["rows"]]}
    with h5py.File(saved["eval"], "r") as f:
        rows = sorted(set(saved["rows"]))
        seen = np.concatenate([f["inputs"][:][rows], f["outputs"][:][rows]]).reshape(-1, *f["inputs"].shape[2:])
        grid_size = f["grid_sizes"][0]
    rng = np.random.default_rng(args.seed)
    weights = np.ones(table.shape[0])
    if args.by_use:
        with h5py.File(saved["eval"], "r") as f:
            xs = torch.tensor(f["inputs"][:][rows][:, :2], dtype=torch.long, device=device)
            ys = torch.tensor(f["outputs"][:][rows][:, :2], dtype=torch.long, device=device)
        k = table.shape[0]
        for start in range(0, len(rows), 16):
            x, y = xs[start: start + 16], ys[start: start + 16]
            n = len(x)
            state = model.encoder(x.flatten(0, 1))                                              # n * 2 states
            out = model.decoder(model.executor(state.repeat_interleave(k, dim=0), table.repeat(2 * n, 1)[:, None])).argmax(dim=-1)
            ok = out.view(n, 2, k, -1).eq(y.flatten(2)[:, :, None]).all(dim=3).all(dim=1)      # n, k
            action = model.programmer(model.encoder(x[:, 0]), model.encoder(y[:, 0]))
            rank = (torch.nn.functional.normalize(action.flatten(1), dim=-1) @ table.T).argsort(dim=1, descending=True)
            for i in range(n):
                good = [c for c in rank[i].tolist() if bool(ok[i, c])]
                if good:
                    weights[good[0]] += 1
        print(f"codes that answered a task of the stream: {int((weights > 1).sum())} of {k}; the most used holds {weights.max() / weights.sum():.3f} of the dreams")
    codes = rng.choice(table.shape[0], size=args.rows, p=weights / weights.sum())
    picks = rng.integers(0, len(seen), size=(args.rows, 2))
    inputs = seen[picks]                                           # rows, 2, H, W
    outputs = np.zeros_like(inputs)
    for start in range(0, args.rows, 1024):
        x = torch.tensor(inputs[start: start + 1024], dtype=torch.long, device=device)
        c = table[torch.tensor(codes[start: start + 1024], device=device)]
        state = model.encoder(x.flatten(0, 1))
        out = model.decoder(model.executor(state, c.repeat_interleave(2, dim=0)[:, None])).argmax(dim=-1)
        outputs[start: start + 1024] = out.view(-1, 2, *out.shape[1:]).cpu().numpy().astype(outputs.dtype)
    changed = float((outputs != inputs).reshape(args.rows, -1).any(axis=1).mean())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as out:
        out["inputs"], out["outputs"] = inputs.astype(np.int8), outputs.astype(np.int8)
        out["depths"] = np.ones(args.rows, dtype=np.int32)
        out["grid_sizes"] = np.tile(np.asarray(grid_size, dtype=np.int32), (args.rows, 1))
        out["n_shapes"] = np.ones((args.rows, 2), dtype=np.int32)
        out["n_support"] = np.ones(args.rows, dtype=np.int32)
        suites = [json.dumps([{"selector": {"type": "by_index", "param": 1}, "primitive": f"dream_code_{int(c)}", "target": None}]) for c in codes]
        out.create_dataset("transformation_suites", data=suites, dtype=h5py.string_dtype())
    print(f"{args.rows} dreamt tasks from {len(seen)} seen grids and {table.shape[0]} codes; {changed:.2f} of them change a grid: {args.out}")


if __name__ == "__main__":
    main()
