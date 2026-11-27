#!/usr/bin/env python3
"""A sleep of its own for every task: from the same trained weights, train all parts on one task's demonstration pairs
seen through the views (symmetries and colour permutations), answer its tests, then throw the changed weights away.

Training on a task. A batch is `--batch` views of the task drawn at random; in each, every demonstration pair is
predicted from the mean of the programmer's codes of the other pairs, as in the run's own training. Nothing is replayed:
the weights are dropped after the task, so nothing can be forgotten.

Answering. For each of the first `--votes` views the code is the mean of the programmer's codes of all the demonstration
pairs (optionally moved by `--fit-steps` steps of Adam on them), the test inputs are executed under it, the predictions
are mapped back and voted. Reported: the first view alone, the most frequent grid (vote1), either of the two most
frequent (vote2); each a mean over tasks of the share of the task's tests solved. `--steps 0` answers with the trained
weights unchanged. Test answers are read only to score.

    python scripts/arc/sleep_per_task.py --run runs/arc16_programmer_s0 --split evaluation --steps 200 --shard 0/4 --out runs/arc16_programmer_s0/per_task_s200.json
    python scripts/arc/sleep_per_task.py --merge 4 --out runs/arc16_programmer_s0/per_task_s200.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "arc"))

from core.registry import DATASETS, METHODS, build  # noqa: E402
from evals.arc import autocast, fit_codes, predict, read_codes, undo, view  # noqa: E402
from evaluate import load_run  # noqa: E402


def merge(out: Path, shards: int) -> None:
    parts = [json.loads(out.with_suffix(f".shard{i}.json").read_text()) for i in range(shards)]
    tasks = sorted((task for part in parts for task in part["per_task"]), key=lambda task: task["index"])
    result = {**{k: v for k, v in parts[0].items() if k not in ("per_task", "shard")}, "tasks": len(tasks)}
    for key in ("first_view", "vote1", "vote2"):
        result[key] = float(np.mean([task[key] for task in tasks]))
    result["seconds_per_task"] = float(np.mean([task["seconds"] for task in tasks]))
    result["per_task"] = tasks
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_task"}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--votes", type=int, default=64)
    parser.add_argument("--colors", type=int, default=8, help="colour permutations used for the views (the first of the run's own)")
    parser.add_argument("--fit-steps", type=int, default=0)
    parser.add_argument("--fit-lr", type=float, default=0.05)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--every", type=int, default=1, help="use every k-th task only, for quick comparisons of settings")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--merge", type=int, help="combine this many shard files written under --out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true", help="compile the programmer and the executor once (weights are reset in place, so the compiled code stays valid)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.merge:
        merge(args.out, args.merge)
        return
    shard, shards = (int(v) for v in args.shard.split("/"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, step = load_run(args.run, device, args.checkpoint)
    data = build(DATASETS, config["data"])
    method = build(METHODS, config["method"])
    file = data.eval_tasks(args.split)
    tasks = len(file["names"]) if args.max_tasks is None else min(args.max_tasks, len(file["names"]))
    views = [(s, c) for c in range(min(args.colors, len(data.tables))) for s in range(data.symmetries)]
    trained = {key: value.clone() for key, value in model.state_dict().items()}
    if args.compile and device.type == "cuda":
        for part in (model.programmer, model.executor):
            part.forward = torch.compile(part.forward)
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).long().to(device)  # noqa: E731
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    per_task = []
    chosen = [t for t in range(tasks) if t % args.every == 0]
    for place, t in enumerate(chosen):
        if place % shards != shard:
            continue
        started = time.monotonic()
        demo_x, demo_y = file["demo_inputs"][file["demo_task"] == t], file["demo_outputs"][file["demo_task"] == t]
        test_x, test_y = file["test_inputs"][file["test_task"] == t], file["test_outputs"][file["test_task"] == t].astype(np.uint8)
        seen_x = tensor(np.stack([view(demo_x, s, data.tables[c]) for s, c in views]))          # (V,n,S,S)
        seen_y = tensor(np.stack([view(demo_y, s, data.tables[c]) for s, c in views]))
        model.load_state_dict(trained)
        if args.steps:
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            for k in range(args.steps):
                pick = torch.randint(0, len(views), (args.batch,), generator=generator).to(device)
                batch = {"inputs": seen_x[pick], "outputs": seen_y[pick], "task": torch.full((args.batch,), -1, device=device), "variant": pick}
                optimizer.zero_grad(set_to_none=True)
                with autocast(device):
                    loss = method.compute_loss(model, batch, k).total
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        model.eval()
        used = views[: args.votes]
        v, n = len(used), len(demo_x)
        vx, vy = seen_x[:v].flatten(0, 1), seen_y[:v].flatten(0, 1)
        owner = torch.arange(v, device=device).repeat_interleave(n)
        code = read_codes(model, vx, vy, 256).view(v, n, -1).mean(dim=1)
        if args.fit_steps:
            code = fit_codes(model, vx, vy, owner, code, args.fit_steps, args.fit_lr, 256)
        seen_tests = tensor(np.stack([view(test_x, s, data.tables[c]) for s, c in used]))        # (V,q,S,S)
        q = len(test_x)
        guess = predict(model, seen_tests.flatten(0, 1), code.repeat_interleave(q, dim=0), 256).view(v, q, *test_y.shape[1:]).cpu().numpy().astype(np.uint8)
        first, one, two, misses = [], [], [], []
        for j in range(q):
            answer = test_y[j].tobytes()
            back = [undo(guess[i, j], s, data.tables[c]) for i, (s, c) in enumerate(used)]
            back = [None if b is None else b.tobytes() for b in back]
            counts = Counter(b for b in back if b is not None).most_common(2)
            top = [grid for grid, _ in counts]
            first.append(back[0] == answer)
            one.append(bool(top) and top[0] == answer)
            two.append(answer in top)
            # how the most frequent guess misses: -1 no valid guess, -2 wrong shape, else the number of wrong cells; and how many views agree on it
            if not top:
                misses.append([-1, 0])
            else:
                best = np.frombuffer(top[0], dtype=np.uint8).reshape(test_y[j].shape)
                shape_right = ((best != 10) == (test_y[j] != 10)).all()
                misses.append([int((best != test_y[j]).sum()) if shape_right else -2, int(counts[0][1])])
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_task.append({"index": t, "name": str(file["names"][t]), "pairs": int(n), "tests": int(q), "first_view": float(np.mean(first)),
                         "vote1": float(np.mean(one)), "vote2": float(np.mean(two)), "seconds": time.monotonic() - started, "top_guess": misses})
        if len(per_task) % 20 == 0:
            print(f"{len(per_task)} tasks, vote2 so far {100 * np.mean([p['vote2'] for p in per_task]):.1f}%, {np.mean([p['seconds'] for p in per_task]):.1f} s per task", flush=True)
    result = {"run": args.run.name, "step": step, "split": args.split, "sleep_steps": args.steps, "batch": args.batch, "lr": args.lr, "votes": args.votes,
              "fit_steps": args.fit_steps, "every": args.every, "shard": args.shard, "per_task": per_task}
    target = args.out if shards == 1 else args.out.with_suffix(f".shard{shard}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1) + "\n")
    if shards == 1:
        merge_one = {key: float(np.mean([p[key] for p in per_task])) for key in ("first_view", "vote1", "vote2", "seconds")}
        print(json.dumps(merge_one, indent=1))


if __name__ == "__main__":
    main()
