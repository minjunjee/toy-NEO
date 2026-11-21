#!/usr/bin/env python3
"""Turn a training checkpoint of a fixed codebook into the same model with a codebook that can grow.

The code vectors get extra rows that are switched off (never chosen), so the widened model computes exactly what the
original does; the optimizer's moments for the codebook are padded with zeros; the config inside the checkpoint is changed
to code type vq_growable. Training can then be resumed from the widened checkpoint with a config that uses vq_growable of
the same size, and an offline phase can switch rows on.

    python scripts/babytheorist/widen_codebook.py --ckpt runs/<run>/ckpt/step-NNNNNN.pt --new-run <name> --size 16
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
EMBEDDING = "code.embedding.weight"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--new-run", required=True)
    parser.add_argument("--size", type=int, required=True)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    old = payload["model"][EMBEDDING].shape[0]
    if args.size <= old:
        raise SystemExit(f"the codebook already has {old} rows")
    generator = torch.Generator().manual_seed(0)
    extra = (torch.rand(args.size - old, payload["model"][EMBEDDING].shape[1], generator=generator) * 2 - 1) / args.size  # the lab VQ's initial range
    for key in ("model", "raw_model"):
        if key in payload:
            weights = payload[key][EMBEDDING]
            payload[key][EMBEDDING] = torch.cat([weights, extra.to(weights.dtype)])
            payload[key]["code.active"] = torch.arange(args.size) < old
    widened = 0
    for state in payload["method"]["optimizer"]["state"].values():
        for name, value in list(state.items()):
            if isinstance(value, torch.Tensor) and tuple(value.shape) == (old, extra.shape[1]):
                state[name] = torch.cat([value, torch.zeros(args.size - old, extra.shape[1], dtype=value.dtype)])
                widened += 1
    config = copy.deepcopy(payload["config"])
    config["model"]["code"] = {**config["model"]["code"], "type": "vq_growable", "size": args.size, "initial": old}
    payload["config"] = config
    output = REPO / "runs" / args.new_run / "ckpt" / args.ckpt.name
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    (output.parents[1] / "widened_from.txt").write_text(f"{args.ckpt}\n")
    print(f"{old} -> {args.size} rows, {widened} optimizer tensors padded -> {output}")


if __name__ == "__main__":
    main()
