#!/usr/bin/env python3
"""Time real training steps of a run config and record GPU memory, set up exactly as scripts/train.py sets it up (same
data loader, precision and compiled parts), and optionally time one call of each evaluation block of the config.

    python scripts/bench_train.py configs/babytheorist/<run>.yaml [--warmup 40] [--steps 150] [--evals] [--tag name]

Prints one JSON line: seconds per step (after warmup), peak memory allocated and reserved by this process during the
timed steps, and per evaluation block its seconds and the extra peak memory it needed. Several copies started at once
on one GPU show how runs sharing a GPU slow each other down.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from core.config import load_config  # noqa: E402
from core.registry import DATASETS, METHODS, build, import_all  # noqa: E402
from training.trainer import run_evals, seed_everything, to_device, train_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--evals", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    import_all()
    config = load_config(args.config)
    tc = train_config(config["train"])
    device = torch.device("cuda")
    seed_everything(tc.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = build(DATASETS, config["data"])
    method = build(METHODS, config["method"])
    model = method.build_model(config["model"], None).to(device)
    method.attach(model, device, tc.precision)
    if tc.compile == "parts":
        for name in ("programmer", "executor"):
            part = getattr(model, name, None)
            if part is not None:
                part.forward = torch.compile(part.forward)
    method.set_train_mode(model)
    loader = iter(data.train_loader(tc.batch_size, args.warmup + args.steps, 0, tc.seed, 0, 1, tc.num_workers, True))
    for step in range(args.warmup):
        method.train_step(model, to_device(next(loader), device), step)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for step in range(args.warmup, args.warmup + args.steps):
        method.train_step(model, to_device(next(loader), device), step)
    torch.cuda.synchronize()
    seconds = (time.time() - start) / args.steps
    out = {"config": args.config.stem, "tag": args.tag, "batch": tc.batch_size, "codes": config["model"]["code"]["size"], "seconds_per_step": round(seconds, 4),
           "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2)}
    if args.evals:
        out["evals"] = {}
        for block in config["evals"]:
            base = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(2):  # the first call may compile; the second is the steady cost
                torch.cuda.synchronize()
                start = time.time()
                run_evals(model, data, [block], args.warmup + args.steps, device)
                torch.cuda.synchronize()
                if repeat == 0:
                    first = time.time() - start
            method.set_train_mode(model)
            out["evals"][block.get("name", block["type"])] = {"every": block.get("every"), "first_seconds": round(first, 2), "seconds": round(time.time() - start, 2),
                                                              "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "train_peak_gb": round(base / 2**30, 2)}
        out["reserved_after_evals_gb"] = round(torch.cuda.memory_reserved() / 2**30, 2)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
