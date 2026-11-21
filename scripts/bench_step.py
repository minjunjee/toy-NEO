#!/usr/bin/env python3
"""Time one training step of a run config on the GPU, with optional speed-ups, and check they do not change the numbers.

    python scripts/bench_step.py configs/babytheorist/<run>.yaml --batch 128 [--compile parts|model] [--fused] [--steps 20]

--compile parts   torch.compile on the programmer and the executor (the two parts that take the time)
--compile model   torch.compile on the whole model
--fused           AdamW with fused=True
The same random batch and the same initial weights are used for every setting; the loss of the first step is printed so
settings can be compared (it must agree to a few decimals; bf16 and kernel fusion change the last digits).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from core.config import load_config  # noqa: E402
from core.registry import METHODS, build, import_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--compile", choices=("none", "parts", "model"), default="none")
    parser.add_argument("--mode", default=None, help="torch.compile mode, e.g. max-autotune-no-cudagraphs")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    import_all()
    config = load_config(args.config)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    method = build(METHODS, config["method"])
    model = method.build_model(config["model"], init_from=None).to(device)
    if args.compile == "parts":
        model.programmer = torch.compile(model.programmer, mode=args.mode)
        model.executor = torch.compile(model.executor, mode=args.mode)
    elif args.compile == "model":
        model = torch.compile(model, mode=args.mode)
    method.attach(model, device, "bf16")
    if args.fused:
        method.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=method.lr, weight_decay=method.weight_decay, fused=True)
        method.base_lrs = [group["lr"] for group in method.optimizer.param_groups]
    method.set_train_mode(model)
    generator = torch.Generator().manual_seed(1)
    grid = int(config["model"]["executor"].get("grid_dim", 15))
    batch = {"x": torch.randint(0, 10, (args.batch, grid, grid), generator=generator).to(device), "y": torch.randint(0, 10, (args.batch, grid, grid), generator=generator).to(device)}
    started = time.time()
    first = method.train_step(model, batch, 0)
    torch.cuda.synchronize()
    warm = time.time() - started
    for i in range(1, 6):
        method.train_step(model, batch, i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for i in range(args.steps):
        method.train_step(model, batch, i + 6)
    torch.cuda.synchronize()
    per_step = (time.time() - started) / args.steps
    print(f"{args.config.stem} batch {args.batch} compile={args.compile}{'/' + args.mode if args.mode else ''} fused={args.fused}: "
          f"{per_step * 1000:.1f} ms/step, {args.batch / per_step:.0f} pairs/s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, "
          f"first step {warm:.1f} s, first loss {first['loss']:.5f}", flush=True)


if __name__ == "__main__":
    main()
