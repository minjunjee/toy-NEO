#!/usr/bin/env python3
"""Sleep step that reopens codes which stayed mixed under two-pair hard EM. No label is used, except by --diagnose.

1. Read train tasks of two pairs. For every task, execute every code on both pairs and take the code with the smallest
   summed cross-entropy (the hard-EM assignment), and whether that code reproduces both pairs exactly.
2. A code is mixed when it is assigned at least --min-share of the tasks and reproduces both pairs of fewer than
   --max-rate of them.
3. The programmer's outputs (normalized) on the first pairs of a mixed code's tasks are split by k-means into
   --children + 1 groups. The group nearest the code keeps it; every other group gets one of the least used codes, set to
   the group's mean direction.
4. Sleep training: for --sleep-steps steps the executor and the codebook are trained on the mixed codes' tasks, each task
   with its group's code on both pairs, and the programmer to name that code. Other parts are frozen.
5. The new checkpoint is written for waking training (two-pair hard EM from it) and a report beside it.

With --diagnose the hidden primitive of each task is read, only to report how the k-means groups line up with the
primitives (purity); nothing else uses it.

    python scripts/babytheorist/reopen_mixed_codes.py runs/<run> --out runs/split_init/<name>.pt [--diagnose]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from training.evaluate import load_run  # noqa: E402


def kmeans(points: torch.Tensor, k: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    centers = points[torch.randperm(len(points), generator=generator)[:k].to(points.device)].clone()
    for _ in range(iters):
        assign = (points @ centers.T).argmax(dim=1)
        for j in range(k):
            mine = points[assign == j]
            if len(mine):
                centers[j] = F.normalize(mine.mean(dim=0), dim=0)
    return (points @ centers.T).argmax(dim=1), centers


@torch.no_grad()
def assign(model, table, x, y, x2, y2, batch=256):
    codes, exact, actions = [], [], []
    size = table.shape[0]
    for i in range(0, len(x), batch):
        sx, sy, tx, ty = x[i:i + batch], y[i:i + batch], x2[i:i + batch], y2[i:i + batch]
        n = len(sx)
        total = 0
        for source, goal in ((sx, sy), (tx, ty)):
            state = model.encoder(source)
            logits = model.decoder(model.executor(state.repeat_interleave(size, dim=0), table.repeat(n, 1)[:, None]))
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), goal.repeat_interleave(size, dim=0).reshape(-1), reduction="none")
            total = total + loss.view(n, size, -1).sum(dim=2)
        best = total.argmin(dim=1)
        ok = torch.ones(n, dtype=torch.bool, device=x.device)
        for source, goal in ((sx, sy), (tx, ty)):
            out = model.decoder(model.executor(model.encoder(source), table[best][:, None])).argmax(dim=-1)
            ok &= out.eq(goal).flatten(1).all(dim=1)
        action = model.programmer(model.encoder(sx), model.encoder(sy)).flatten(1).float()
        codes.append(best)
        exact.append(ok)
        actions.append(F.normalize(action, dim=-1))
    return torch.cat(codes), torch.cat(exact), torch.cat(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=20000)
    parser.add_argument("--min-share", type=float, default=0.01)
    parser.add_argument("--max-rate", type=float, default=0.5)
    parser.add_argument("--children", type=int, default=3)
    parser.add_argument("--sleep-steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--diagnose", action="store_true", help="report purity of the groups against the hidden primitive (reads the label)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model, data, config, step = load_run(args.run, None, device)
    model.eval()
    path = config["data"]["train"][0]
    with h5py.File(REPO / path, "r") as f:
        x = torch.from_numpy(f["inputs"][: args.tasks].astype(np.int64)).to(device)
        y = torch.from_numpy(f["outputs"][: args.tasks].astype(np.int64)).to(device)
        names = [json.loads(p)[0]["primitive"] for p in f["transformation_suites"][: args.tasks]] if args.diagnose else None
    size = model.code.size
    table = F.normalize(model.code.embedding.weight.detach().float(), dim=-1)
    codes, exact, actions = assign(model, table, x[:, 0], y[:, 0], x[:, 1], y[:, 1])
    share = torch.bincount(codes, minlength=size).float() / len(codes)
    rate = torch.zeros(size, device=device).scatter_add_(0, codes, exact.float()) / torch.bincount(codes, minlength=size).clamp(min=1)
    mixed = [int(c) for c in torch.nonzero((share >= args.min_share) & (rate < args.max_rate)).flatten()]
    free = [int(c) for c in share.argsort() if int(c) not in mixed]
    report = {"run": args.run.name, "step": step, "tasks": len(codes), "mixed": {}, "both_pairs_reproduced": float(exact.float().mean())}
    groups = {}
    for code in mixed:
        rows = torch.nonzero(codes == code).flatten()
        group, centers = kmeans(actions[rows], args.children + 1)
        keep = int((centers @ table[code]).argmax())
        mapping = {}
        for j in range(args.children + 1):
            if j == keep:
                mapping[j] = code
            else:
                child = free.pop(0)
                mapping[j] = child
                model.code.embedding.weight.data[child] = centers[j].to(model.code.embedding.weight.dtype) * model.code.embedding.weight.data[code].norm()
        target = torch.tensor([mapping[int(g)] for g in group], device=device)
        groups[code] = (rows, target)
        entry = {"share": float(share[code]), "both_pairs_reproduced": float(rate[code]), "children": [mapping[j] for j in range(args.children + 1) if j != keep],
                 "group_sizes": torch.bincount(group, minlength=args.children + 1).tolist()}
        if names is not None:
            per_group = [Counter(names[int(r)] for r, g in zip(rows, group) if int(g) == j) for j in range(args.children + 1)]
            entry["primitives_by_group"] = [dict(c.most_common(4)) for c in per_group]
            entry["purity"] = sum(max(c.values()) for c in per_group if c) / max(1, len(rows))
            entry["primitives_overall"] = dict(Counter(names[int(r)] for r in rows).most_common(6))
        report["mixed"][code] = entry
        print(f"code {code}: share {share[code]:.3f} reproduced {rate[code]:.2f} children {entry['children']} sizes {entry['group_sizes']}"
              + (f" purity {entry['purity']:.2f} overall {entry['primitives_overall']}" if names is not None else ""), flush=True)
    # sleep training on the reopened groups: executor, codebook and programmer, nothing else
    if groups and args.sleep_steps:
        rows = torch.cat([g[0] for g in groups.values()])
        targets = torch.cat([g[1] for g in groups.values()])
        params = list(model.executor.parameters()) + list(model.code.parameters()) + list(model.programmer.parameters())
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        model.train()
        for s in range(args.sleep_steps):
            pick = torch.randint(len(rows), (128,), device=device)
            r, c = rows[pick], targets[pick]
            vectors = model.code.codebook(c)[:, None]
            loss = 0
            for j in (0, 1):
                logits = model.decoder(model.executor(model.encoder(x[r, j]), vectors))
                loss = loss + F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y[r, j].reshape(-1))
            action = model.programmer(model.encoder(x[r, 0]), model.encoder(y[r, 0])).flatten(1).float()
            similarity = F.normalize(action, dim=-1) @ F.normalize(model.code.embedding.weight.float(), dim=-1).T
            loss = loss + F.cross_entropy(10.0 * similarity, c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        new_codes, new_exact, _ = assign(model, F.normalize(model.code.embedding.weight.detach().float(), dim=-1), x[rows, 0], y[rows, 0], x[rows, 1], y[rows, 1])
        report["after_sleep_reproduced_on_reopened"] = float(new_exact.float().mean())
        report["after_sleep_codes_on_reopened"] = dict(Counter(new_codes.tolist()).most_common(8))
        print(f"after sleep: reopened tasks reproduced on both pairs {report['after_sleep_reproduced_on_reopened']:.2f}, codes {report['after_sleep_codes_on_reopened']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(sorted((args.run / "ckpt").glob("step-*.pt"))[-1], map_location="cpu", weights_only=False)
    checkpoint["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(checkpoint, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
