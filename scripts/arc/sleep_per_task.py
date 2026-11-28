#!/usr/bin/env python3
"""A sleep of its own for every task: from the same trained weights, train all parts on one task's demonstration pairs
seen through the views (symmetries and colour permutations), answer its tests, then throw the changed weights away.

Training on a task. A batch is `--batch` views of the task drawn at random; in each, every demonstration pair is
predicted from the mean of the programmer's codes of the other pairs, as in the run's own training. Nothing is replayed:
the weights are dropped after the task, so nothing can be forgotten. `--code` changes where the code comes from: `all`
uses the programmer's reading of all the pairs of the view, the predicted one included (what is used when answering);
`free` gives every view a code vector of its own, started from that reading and trained together with the weights, and
answers with it; `both` adds the losses of `others` and `all`.

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
import hashlib
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
from evals.arc import autocast, crop, fit_codes, predict, read_codes, undo, view  # noqa: E402
from evaluate import load_run  # noqa: E402


def enlarged(canvases: np.ndarray, symmetry: int, table: np.ndarray, scale: int) -> np.ndarray:
    """A view of the grids with every cell drawn as a scale x scale block (after the symmetry and the colour permutation)."""

    out = view(canvases, symmetry, table)
    if scale == 1:
        return out
    size = out.shape[-1]
    for i, canvas in enumerate(out):
        g = np.kron(crop(canvas), np.ones((scale, scale), dtype=np.uint8))
        out[i] = 10
        out[i, : g.shape[0], : g.shape[1]] = g
    return out


def shrunk(canvas: np.ndarray, symmetry: int, table: np.ndarray, scale: int) -> np.ndarray | None:
    """Undo `enlarged` on a predicted canvas; None when the prediction is not made of whole blocks of one colour."""

    if scale > 1:
        g = crop(canvas)
        if g is None or g.shape[0] % scale or g.shape[1] % scale:
            return None
        blocks = g.reshape(g.shape[0] // scale, scale, g.shape[1] // scale, scale)
        small = blocks[:, 0, :, 0]
        if not (blocks == small[:, None, :, None]).all():
            return None
        canvas = np.full_like(canvas, 10)
        canvas[: small.shape[0], : small.shape[1]] = small
    return undo(canvas, symmetry, table)


def fits(canvases: np.ndarray, scale: int) -> bool:
    size = canvases.shape[-1]
    return all(max(crop(c).shape) * scale <= size for c in canvases)


def merge(out: Path, shards: int) -> None:
    parts = [json.loads(out.with_suffix(f".shard{i}.json").read_text()) for i in range(shards)]
    tasks = sorted((task for part in parts for task in part["per_task"]), key=lambda task: task["index"])
    result = {**{k: v for k, v in parts[0].items() if k not in ("per_task", "shard")}, "tasks": len(tasks)}
    for key in ("first_view", "vote1", "vote2", "cell_vote", "cell_or_vote1", "cell_or_vote2"):
        if all(key in task for task in tasks):
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
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--max-pairs", type=int, default=64, help="at most this many pairs in a training step: tasks with many demonstration pairs use fewer views per step")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--code", choices=("others", "all", "free", "both"), default="others",
                        help="where a pair's code comes from while training on the task: the programmer's reading of the other pairs (as in pretraining), its "
                             "reading of all the pairs, a free vector per view started from that reading and trained with the weights, or the first two together")
    parser.add_argument("--code-lr", type=float, default=0.01, help="learning rate of the free codes")
    parser.add_argument("--votes", type=int, default=64)
    parser.add_argument("--symmetries", type=int, help="symmetries used for the views, when the run was trained with fewer")
    parser.add_argument("--color-perms", type=int, help="colour permutations to draw the views from, when the run was trained with fewer")
    parser.add_argument("--code-init", choices=("read", "first", "random"), default="read",
                        help="start of a free code: the programmer's reading of that view, its reading of the task as given for every view, or a random "
                             "vector of the same length as the readings (no programmer: what a learned task token starts from)")
    parser.add_argument("--scales", type=int, default=1, help="also use views with every cell enlarged up to this factor, for tasks whose grids then still fit the canvas")
    parser.add_argument("--colors", type=int, default=8, help="colour permutations used for the views (the first of the run's own)")
    parser.add_argument("--fit-steps", type=int, default=0)
    parser.add_argument("--fit-lr", type=float, default=0.05)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--every", type=int, default=1, help="use every k-th task only, for quick comparisons of settings")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--merge", type=int, help="combine this many shard files written under --out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump", type=Path, help="folder to save, for every task, the test inputs, the answers and the most frequent guess of each test")
    parser.add_argument("--compile", action="store_true", help="compile the programmer and the executor once (weights are reset in place, so the compiled code stays valid)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.merge:
        merge(args.out, args.merge)
        return
    shard, shards = (int(v) for v in args.shard.split("/"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, step = load_run(args.run, device, args.checkpoint)
    views_of_run = {"color_perms": len(build(DATASETS, {**config["data"], "evals": {}}).tables), "symmetries": config["data"].get("symmetries", 8)}
    data = build(DATASETS, {**config["data"], "color_perms": args.color_perms or views_of_run["color_perms"], "symmetries": args.symmetries or views_of_run["symmetries"]})
    method = build(METHODS, config["method"])
    file = data.eval_tasks(args.split)
    tasks = len(file["names"]) if args.max_tasks is None else min(args.max_tasks, len(file["names"]))
    plain_views = [(s, c, 1) for c in range(min(args.colors, len(data.tables))) for s in range(data.symmetries)]
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
        # views: the symmetries and colour permutations, and with --scales the same again with every cell enlarged, where everything fits
        views = list(plain_views)
        for scale in range(2, args.scales + 1):
            if fits(demo_x, scale) and fits(demo_y, scale) and fits(test_x, scale):
                views += [(s, c, scale) for s, c, _ in plain_views]
        seen_x = tensor(np.stack([enlarged(demo_x, s, data.tables[c], k) for s, c, k in views]))          # (V,n,S,S)
        seen_y = tensor(np.stack([enlarged(demo_y, s, data.tables[c], k) for s, c, k in views]))
        model.load_state_dict(trained)
        free = None
        if args.code == "free":          # one code of its own for every view, started from what the trained programmer reads in all the pairs
            model.eval()
            pairs = len(demo_x)
            free = torch.cat([read_codes(model, seen_x[i: i + 16].flatten(0, 1), seen_y[i: i + 16].flatten(0, 1), args.eval_batch).view(-1, pairs, model.code_dim).mean(dim=1)
                              for i in range(0, len(views), 16)])
            if args.code_init == "first":
                free = free[:1].expand(len(views), -1).clone()
            elif args.code_init == "random":
                noise = torch.randn(free.shape, generator=generator).to(device)
                free = noise / noise.norm(dim=-1, keepdim=True) * free.norm(dim=-1, keepdim=True)
            free.requires_grad_(True)
        if args.steps:
            model.train()
            groups = [{"params": list(model.parameters()), "lr": args.lr, "weight_decay": args.weight_decay}]
            if free is not None:
                groups.append({"params": [free], "lr": args.code_lr, "weight_decay": 0.0})
            optimizer = torch.optim.AdamW(groups)
            size = max(2, min(args.batch, args.max_pairs // len(demo_x)))
            for k in range(args.steps):
                pick = torch.randint(0, len(views), (size,), generator=generator).to(device)
                batch = {"inputs": seen_x[pick], "outputs": seen_y[pick], "task": torch.full((size,), -1, device=device), "variant": pick}
                optimizer.zero_grad(set_to_none=True)
                with autocast(device):
                    loss = torch.zeros((), device=device)
                    if args.code in ("others", "both"):       # every pair from the programmer's code of the other pairs, as in the run's own training
                        loss = loss + method.compute_loss(model, batch, k).total
                    if args.code in ("free", "all", "both"):  # every pair from one code for the whole view
                        x, y = batch["inputs"].flatten(0, 1), batch["outputs"].flatten(0, 1)
                        state = model.encoder(x)
                        if args.code == "free":
                            whole = free[pick]
                        else:                                 # the programmer's code of all the pairs, the predicted pair included: what is used to answer
                            whole = model.programmer(state, model.encoder(y)).flatten(1).float().view(size, len(demo_x), -1).mean(dim=1)
                        logits = model.run(state, whole.repeat_interleave(len(demo_x), dim=0))[-1]
                        loss = loss + torch.nn.functional.cross_entropy(logits.float().flatten(0, 2), y.flatten())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        model.eval()
        keep = list(range(min(args.votes, len(plain_views)))) + list(range(len(plain_views), len(views)))     # the views that vote
        used = [views[i] for i in keep]
        v, n = len(used), len(demo_x)
        vx, vy = seen_x[keep].flatten(0, 1), seen_y[keep].flatten(0, 1)
        owner = torch.arange(v, device=device).repeat_interleave(n)
        own = read_codes(model, vx, vy, args.eval_batch).view(v, n, -1)
        code = own.mean(dim=1) if free is None else free.detach()[keep]
        # checks that need no answers, on the first view: demonstration pairs reproduced from the other pairs' code, and from all pairs' code
        others = (own[0].sum(dim=0, keepdim=True) - own[0]) / max(n - 1, 1)
        demo_others = float(predict(model, seen_x[0], others, args.eval_batch).eq(seen_y[0]).flatten(1).all(dim=1).float().mean())
        demo_all = float(predict(model, seen_x[0], code[:1].expand(n, -1), args.eval_batch).eq(seen_y[0]).flatten(1).all(dim=1).float().mean())
        del own
        if args.fit_steps:
            code = fit_codes(model, vx, vy, owner, code, args.fit_steps, args.fit_lr, args.eval_batch)
        seen_tests = tensor(np.stack([enlarged(test_x, s, data.tables[c], k) for s, c, k in used]))        # (V,q,S,S)
        q = len(test_x)
        guess = predict(model, seen_tests.flatten(0, 1), code.repeat_interleave(q, dim=0), args.eval_batch).view(v, q, *test_y.shape[1:]).cpu().numpy().astype(np.uint8)
        first, one, two, misses, ballots, by_cell, shown = [], [], [], [], [], [], []
        for j in range(q):
            answer = test_y[j].tobytes()
            back = [shrunk(guess[i, j], s, data.tables[c], k) for i, (s, c, k) in enumerate(used)]
            # a vote cell by cell: among the views that agree on the most frequent output shape, the most frequent colour of every cell
            valid = [b for b in back if b is not None]
            cell_grid = None
            if valid:
                shapes = Counter(((b != 10).sum(axis=0)[0], (b != 10).sum(axis=1)[0]) for b in valid)
                shape = shapes.most_common(1)[0][0]
                same = np.stack([b for b in valid if ((b != 10).sum(axis=0)[0], (b != 10).sum(axis=1)[0]) == shape])
                counts_by_colour = np.stack([(same == colour).sum(axis=0) for colour in range(11)])
                cell_grid = counts_by_colour.argmax(axis=0).astype(np.uint8).tobytes()
            back = [None if b is None else b.tobytes() for b in back]
            tally = Counter(b for b in back if b is not None)
            counts = tally.most_common(2)
            top = [grid for grid, _ in counts]
            # the five most frequent guesses as (digest of the grid, number of views, is it the answer), so runs can be voted together later
            ballots.append([[hashlib.md5(grid).hexdigest()[:16], int(count), grid == answer] for grid, count in tally.most_common(5)])
            shown.append(np.frombuffer(top[0], dtype=np.uint8).reshape(test_y[j].shape) if top else np.full_like(test_y[j], 10))
            first.append(back[0] == answer)
            one.append(bool(top) and top[0] == answer)
            two.append(answer in top)
            # [the cell-by-cell grid is right, it or the most frequent whole grid is right, either of those or the second most frequent is right]
            by_cell.append([cell_grid == answer, cell_grid == answer or (bool(top) and top[0] == answer), cell_grid == answer or answer in top])
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
                         "vote1": float(np.mean(one)), "vote2": float(np.mean(two)), "seconds": time.monotonic() - started, "top_guess": misses, "ballots": ballots,
                         "cell_vote": float(np.mean([b[0] for b in by_cell])), "cell_or_vote1": float(np.mean([b[1] for b in by_cell])),
                         "cell_or_vote2": float(np.mean([b[2] for b in by_cell])),
                         "demo_from_others": demo_others, "demo_from_all": demo_all})
        if args.dump:
            args.dump.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.dump / f"task{t:03d}.npz", demo_inputs=demo_x, demo_outputs=demo_y, test_inputs=test_x, answers=test_y, guesses=np.stack(shown))
        if len(per_task) % 20 == 0:
            print(f"{len(per_task)} tasks, vote2 so far {100 * np.mean([p['vote2'] for p in per_task]):.1f}%, {np.mean([p['seconds'] for p in per_task]):.1f} s per task", flush=True)
    result = {"run": args.run.name, "step": step, "split": args.split, "sleep_steps": args.steps, "batch": args.batch, "lr": args.lr, "votes": args.votes,
              "fit_steps": args.fit_steps, "every": args.every, "code": args.code, "code_init": args.code_init, "scales": args.scales, "shard": args.shard, "per_task": per_task}
    target = args.out if shards == 1 else args.out.with_suffix(f".shard{shard}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1) + "\n")
    if shards == 1:
        merge_one = {key: float(np.mean([p[key] for p in per_task])) for key in ("first_view", "vote1", "vote2", "seconds")}
        print(json.dumps(merge_one, indent=1))


if __name__ == "__main__":
    main()
