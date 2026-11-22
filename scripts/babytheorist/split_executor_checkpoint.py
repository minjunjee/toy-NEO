#!/usr/bin/env python3
"""Turn a trained run's shared executor into one network per code: every code's network starts as a copy of the shared
executor, everything else (state encoder and decoder, programmer, codebook) is copied as it is. Writes a checkpoint that
a run with a `per_code` executor of the same width and depth (models/neo_expert.py, method neo_warm) starts from.

The idea under test: share first, split later. With one network per code from the start the networks learn too little
too late (each sees only its own pairs while the codes are still being found); started from a working shared executor,
each copy is already competent and is then free to specialise without touching the others.

    python scripts/babytheorist/split_executor_checkpoint.py runs/<run>/ckpt/step-024000.pt <out.pt> --codes 48
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--codes", type=int, required=True)
    args = parser.parse_args()
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    state = payload["model"]
    shared = {k[len("executor.stack."):]: v for k, v in state.items() if k.startswith("executor.stack.")}
    if not shared:
        raise SystemExit("the source has no shared executor (executor.stack.*)")
    split = {k: v for k, v in state.items() if not k.startswith("executor.")}
    for code in range(args.codes):
        for key, value in shared.items():
            split[f"executor.experts.{code}.{key}"] = value.clone()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": split, "step": 0, "config": payload.get("config"), "split_from": str(args.source)}, args.out)
    print(f"{args.out}: {len(shared)} shared tensors copied into {args.codes} networks")


if __name__ == "__main__":
    main()
