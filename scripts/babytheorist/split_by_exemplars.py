#!/usr/bin/env python3
"""Sleep step that splits a mixed code by exemplars, with the executor deciding the groups. No label is used, except by
--diagnose.

Two-pair hard EM can end with two to four orientation primitives on one code. The programmer's outputs on that code's
tasks carry nothing that separates them (k-means purity at chance, reopen_mixed_codes.py), so here the executor decides:

1. Every train task is assigned to the code with the smallest summed cross-entropy on its two pairs. A code is mixed when
   it holds at least --min-share of the tasks and reproduces both pairs of fewer than --max-rate of them.
2. For a mixed code, the task it explains worst becomes an exemplar, and one of the least used codes becomes a child,
   started at the parent's vector plus a little noise.
3. Rounds: the executor and the codebook are trained for --steps steps on the mixed code's tasks, each with its current
   code (parent or child), half of every batch being other tasks with their own codes so the shared executor keeps them.
   Then each of the mixed code's tasks goes to whichever of its codes reproduces its two pairs better. A child that the
   executor generalised to the exemplar's transformation draws the tasks of that transformation to itself.
4. Steps 2 and 3 repeat for up to --children children per mixed code, each new exemplar being the task the current codes
   explain worst. At the end the programmer is trained to name each task's code.

The checkpoint is written for waking training (two-pair hard EM from it). With --diagnose the hidden primitive is read,
only to report which primitives each code ended with.

    python scripts/babytheorist/split_by_exemplars.py runs/<run> --out runs/split_init/<name>.pt [--diagnose]
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


@torch.no_grad()
def losses(model, codes: torch.Tensor, x, y, batch: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed cross-entropy over both pairs, and exact reproduction of both, for each task under each given code.
    codes: (N, k) code indices per task. Returns (N, k) losses and (N, k) exact flags."""

    out_loss, out_exact = [], []
    table = model.code.codebook(torch.arange(model.code.size, device=x.device)).float()
    for i in range(0, len(x), batch):
        c = codes[i:i + batch]
        n, k = c.shape
        total = torch.zeros(n, k, device=x.device)
        exact = torch.ones(n, k, dtype=torch.bool, device=x.device)
        for j in (0, 1):
            state = model.encoder(x[i:i + batch, j]).repeat_interleave(k, dim=0)
            logits = model.decoder(model.executor(state, table[c.flatten()][:, None]))
            goal = y[i:i + batch, j].repeat_interleave(k, dim=0)
            total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), goal.reshape(-1), reduction="none").view(n * k, -1).sum(dim=1).view(n, k)
            exact &= logits.argmax(dim=-1).eq(goal).flatten(1).all(dim=1).view(n, k)
        out_loss.append(total)
        out_exact.append(exact)
    return torch.cat(out_loss), torch.cat(out_exact)


def train(model, x, y, rows, targets, others, other_targets, steps: int, lr: float, programmer: bool = False) -> None:
    params = list(model.executor.parameters()) + list(model.code.parameters()) + (list(model.programmer.parameters()) if programmer else [])
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        a = torch.randint(len(rows), (64,), device=x.device)
        b = torch.randint(len(others), (64,), device=x.device)
        r, c = torch.cat([rows[a], others[b]]), torch.cat([targets[a], other_targets[b]])
        vectors = model.code.codebook(c)[:, None]
        loss = 0
        for j in (0, 1):
            logits = model.decoder(model.executor(model.encoder(x[r, j]), vectors))
            loss = loss + F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y[r, j].reshape(-1))
        if programmer:
            action = model.programmer(model.encoder(x[r, 0]), model.encoder(y[r, 0])).flatten(1).float()
            similarity = F.normalize(action, dim=-1) @ F.normalize(model.code.embedding.weight.float(), dim=-1).T
            loss = loss + F.cross_entropy(10.0 * similarity, c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=20000)
    parser.add_argument("--min-share", type=float, default=0.01)
    parser.add_argument("--max-rate", type=float, default=0.8)
    parser.add_argument("--children", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--diagnose", action="store_true", help="report the primitives each code ends with (reads the label)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model, data, config, step = load_run(args.run, None, device)
    model.eval()
    with h5py.File(REPO / config["data"]["train"][0], "r") as f:
        x = torch.from_numpy(f["inputs"][: args.tasks].astype(np.int64)).to(device)
        y = torch.from_numpy(f["outputs"][: args.tasks].astype(np.int64)).to(device)
        names = [json.loads(p)[0]["primitive"] for p in f["transformation_suites"][: args.tasks]] if args.diagnose else None
    size = model.code.size
    every = torch.arange(size, device=device).expand(len(x), size)
    loss_all, exact_all = losses(model, every, x, y)
    assigned = loss_all.argmin(dim=1)
    exact = exact_all.gather(1, assigned[:, None])[:, 0]
    share = torch.bincount(assigned, minlength=size).float() / len(x)
    rate = torch.zeros(size, device=device).scatter_add_(0, assigned, exact.float()) / torch.bincount(assigned, minlength=size).clamp(min=1)
    mixed = [int(c) for c in torch.nonzero((share >= args.min_share) & (rate < args.max_rate)).flatten()]
    free = [int(c) for c in share.argsort() if int(c) not in mixed and share[c] < 0.002]
    report = {"run": args.run.name, "step": step, "tasks": len(x), "both_pairs_reproduced": float(exact.float().mean()), "mixed": {}}
    print(f"tasks reproduced on both pairs {exact.float().mean():.3f}; mixed codes {mixed}; free codes {len(free)}", flush=True)
    for parent in mixed:
        rows = torch.nonzero(assigned == parent).flatten()
        family = [parent]
        target = torch.full((len(rows),), parent, device=device)
        for _ in range(args.children):
            if not free:
                break
            fam = torch.tensor(family, device=device).expand(len(rows), len(family))
            fam_loss, fam_exact = losses(model, fam, x[rows], y[rows])
            best = fam_loss.min(dim=1).values
            exemplar = int(best.argmax())                       # the task the family explains worst
            child = free.pop(0)
            with torch.no_grad():
                vector = model.code.embedding.weight[parent]
                model.code.embedding.weight[child] = vector + args.noise * vector.norm() * F.normalize(torch.randn_like(vector), dim=0)
            family.append(child)
            target[exemplar] = child
            others = torch.nonzero(assigned != parent).flatten()
            for _ in range(args.rounds):
                train(model, x, y, rows, target, others, assigned[others], args.steps, args.lr)
                fam = torch.tensor(family, device=device).expand(len(rows), len(family))
                fam_loss, _ = losses(model, fam, x[rows], y[rows])
                target = torch.tensor(family, device=device)[fam_loss.argmin(dim=1)]
                target[exemplar] = child
        fam = torch.tensor(family, device=device).expand(len(rows), len(family))
        fam_loss, fam_exact = losses(model, fam, x[rows], y[rows])
        target = torch.tensor(family, device=device)[fam_loss.argmin(dim=1)]
        reproduced = fam_exact.gather(1, fam_loss.argmin(dim=1)[:, None])[:, 0]
        entry = {"share": float(share[parent]), "reproduced_before": float(rate[parent]), "reproduced_after": float(reproduced.float().mean()),
                 "codes": {int(c): int((target == c).sum()) for c in family}}
        if names is not None:
            entry["primitives_by_code"] = {int(c): dict(Counter(names[int(r)] for r, t in zip(rows, target) if int(t) == c).most_common(3)) for c in family}
        report["mixed"][parent] = entry
        assigned[rows] = target
        print(f"code {parent}: reproduced {rate[parent]:.2f} -> {reproduced.float().mean():.2f}, codes {entry['codes']}"
              + (f", primitives {entry['primitives_by_code']}" if names is not None else ""), flush=True)
    if mixed:  # the programmer learns to name the new codes, the executor keeps every task's code
        rows = torch.arange(len(x), device=device)
        train(model, x, y, rows, assigned, rows, assigned, args.steps, args.lr, programmer=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(sorted((args.run / "ckpt").glob("step-*.pt"))[-1], map_location="cpu", weights_only=False)
    checkpoint["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(checkpoint, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
