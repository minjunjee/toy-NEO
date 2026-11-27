#!/usr/bin/env python3
"""Start a larger environment from a model trained on a smaller one: every part is copied, the codebook keeps its trained
rows and gets new rows, initialised as a fresh model of the target config would initialise them.

The target config says the new codebook size (and everything else). Parameters whose shapes agree are copied from the
source checkpoint; the code vectors are copied into the first rows; anything else keeps the fresh initialisation (the
script prints what was copied and what was not). The result is a checkpoint that `train.init_from` can load.
With `--rows-from`, the new rows right after the trained ones start from the code vectors of an online fast memory.

    python scripts/babytheorist/grow_checkpoint.py --source runs/<24-primitive run>/ckpt/step-024000.pt \
        --config configs/babytheorist/<target>.yaml --out runs/grow_init/<name>.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from core.config import load_config  # noqa: E402
from core.registry import METHODS, build, import_all  # noqa: E402

EMBEDDING = "code.embedding.weight"


def sphere_refit(model, state: dict, saved: dict, chosen: list[int], steps: int) -> torch.Tensor:
    import h5py
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with h5py.File(saved["eval"], "r") as f:
        inputs, outputs = torch.tensor(f["inputs"][:], dtype=torch.long), torch.tensor(f["outputs"][:], dtype=torch.long)
    out, kept, total = [], 0, 0
    for j in chosen:
        rows = saved["memory_rows"][j]
        x, y = inputs[rows][:, :2].flatten(0, 1).to(device), outputs[rows][:, :2].flatten(0, 1).to(device)
        with torch.no_grad():
            state_x = model.encoder(x)
        v = saved["vectors"][j].clone().to(device).requires_grad_(True)
        opt = torch.optim.Adam([v], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            logits = model.decoder(model.executor(state_x, F.normalize(v, dim=-1).expand(len(x), -1)[:, None]))
            F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1)).backward()
            opt.step()
        unit = F.normalize(v.detach(), dim=-1)
        with torch.no_grad():
            done = model.decoder(model.executor(state_x, unit.expand(len(x), -1)[:, None])).argmax(-1).eq(y).flatten(1).all(1).view(-1, 2).all(1)
        kept, total = kept + int(done.sum()), total + len(rows)
        out.append(unit.cpu())
    print(f"unit-vector refit: {kept} of {total} memory tasks reproduced by the placed codes")
    return torch.stack(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows-from", type=Path, help="a saved online fast memory (memory_online.py --save-memory): its new code vectors "
                        "fill the rows right after the trained ones (the codebook normalises them when it reads them)")
    parser.add_argument("--min-memory", type=int, default=2, help="with --rows-from: only codes holding at least this many memory tasks, "
                        "the largest memories first, as many as the new rows allow")
    parser.add_argument("--sphere-steps", type=int, default=0, help="with --rows-from: refit each placed code on its memory tasks as a unit "
                        "vector (the codebook reads its rows normalised, and a fast-memory code fitted without that bound loses its "
                        "function when normalised), from its own direction, Adam lr 0.05, this many steps")
    args = parser.parse_args()
    import_all()
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    method = build(METHODS, config["method"])
    model = method.build_model(config["model"], None)
    target = model.state_dict()
    source = torch.load(args.source, map_location="cpu", weights_only=False)["model"]
    copied, partial, kept = [], [], []
    for key, value in target.items():
        if key not in source:
            kept.append(key)
        elif source[key].shape == value.shape:
            target[key] = source[key].clone()
            copied.append(key)
        elif key == EMBEDDING and source[key].shape[1:] == value.shape[1:] and source[key].shape[0] < value.shape[0]:
            target[key][: source[key].shape[0]] = source[key]
            partial.append(f"{key}: {source[key].shape[0]} trained rows of {value.shape[0]}")
        else:
            kept.append(f"{key} {tuple(source[key].shape)} -> {tuple(value.shape)}")
    if args.rows_from:
        saved = torch.load(args.rows_from, map_location="cpu", weights_only=False)
        start = source[EMBEDDING].shape[0]
        sizes = [len(m) for m in saved["memory_rows"]]
        chosen = sorted((j for j, n in enumerate(sizes) if n >= args.min_memory), key=lambda j: -sizes[j])[: target[EMBEDDING].shape[0] - start]
        vectors = saved["vectors"][chosen]
        if args.sphere_steps:
            vectors = sphere_refit(model, target, saved, chosen, args.sphere_steps)
        print(f"fast memory: {len(sizes)} codes, {sum(n >= args.min_memory for n in sizes)} with at least {args.min_memory} tasks, {len(chosen)} placed")
        target[EMBEDDING][start: start + len(vectors)] = vectors.to(target[EMBEDDING].dtype)
        partial.append(f"rows {start} to {start + len(vectors) - 1} from the fast memory {args.rows_from}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": target, "step": 0, "config": config, "source": str(args.source)}, args.out)
    print(f"copied {len(copied)} tensors; partly: {partial}; kept fresh: {kept}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
