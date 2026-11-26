#!/usr/bin/env python3
"""Start a larger environment from a model trained on a smaller one: every part is copied, the codebook keeps its trained
rows and gets new rows, initialised as a fresh model of the target config would initialise them.

The target config says the new codebook size (and everything else). Parameters whose shapes agree are copied from the
source checkpoint; the code vectors are copied into the first rows; anything else keeps the fresh initialisation (the
script prints what was copied and what was not). The result is a checkpoint that `train.init_from` can load.

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": target, "step": 0, "config": config, "source": str(args.source)}, args.out)
    print(f"copied {len(copied)} tensors; partly: {partial}; kept fresh: {kept}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
