#!/usr/bin/env python3
"""Do the tasks that share a code pull it in different directions before the executor can tell them apart?

For each checkpoint of a two-pair hard EM run, on train tasks of every primitive (two pairs each, from the two-pair train
file; the hidden primitive only sorts the result):
  1. each task goes to the code hard EM would give it: the smallest reconstruction loss summed over cells and over the
     task's two pairs (models/neo_em.py, without the dead-code bonus);
  2. for every task, the gradient of that loss with respect to the code vector it was given (16 numbers, what the split
     would move), and with respect to the conditioning vector the executor makes from it (128 numbers, cond_proj);
  3. for each code holding tasks of at least two primitives, the gradients are centred and three numbers are reported:
       top_share      share of their variance along the first principal direction; uses no labels
       split_purity   the tasks split in two by the sign along that direction, purity of the two halves with respect
                      to the hidden primitive, next to the same purity with the primitives shuffled (a diagnostic)
       kmeans_purity  k-means on unit-length gradients with k the number of primitives holding at least 10% of the
                      code's tasks, purity next to shuffled
A code is a good candidate for a split by gradient if split_purity or kmeans_purity is well above shuffled while the
executor still fails on those tasks.

    python scripts/babytheorist/gradient_split_probe.py --run runs/onestep_s_m24_hardem_pairs2_ckpt_s43 \
        --steps 400 800 1200 --out runs/queue/gradient_probe_s43.json
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

SWAP = ("rot90", "rot270", "transpose", "anti_transpose")


def tasks(path: Path, per_primitive: int, seed: int):
    with h5py.File(path, "r") as f:
        names = [json.loads(t)[0]["primitive"] for t in f["transformation_suites"][:]]
        rng = np.random.default_rng(seed)
        keep = []
        for name in sorted(set(names)):
            rows = [i for i, n in enumerate(names) if n == name]
            keep += sorted(rng.choice(rows, size=min(per_primitive, len(rows)), replace=False).tolist())
        keep = sorted(keep)
        x, y = f["inputs"][keep], f["outputs"][keep]
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), [names[i] for i in keep]


def purity(groups: np.ndarray, labels: list[str]) -> float:
    total = 0
    for g in set(groups.tolist()):
        total += Counter(labels[i] for i in np.nonzero(groups == g)[0]).most_common(1)[0][1]
    return total / len(labels)


def kmeans(points: np.ndarray, k: int, rng: np.random.Generator, restarts: int = 8, iters: int = 50) -> np.ndarray:
    best, best_cost = None, np.inf
    for _ in range(restarts):
        centres = points[rng.choice(len(points), size=k, replace=False)]
        for _ in range(iters):
            groups = ((points[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
            centres = np.stack([points[groups == j].mean(0) if (groups == j).any() else centres[j] for j in range(k)])
        cost = ((points - centres[groups]) ** 2).sum()
        if cost < best_cost:
            best, best_cost = groups, cost
    return best


def analyse(grads: np.ndarray, labels: list[str], rng: np.random.Generator, shuffles: int = 20) -> dict:
    centred = grads - grads.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    top_share = float(s[0] ** 2 / max((s ** 2).sum(), 1e-12))
    split = (centred @ vt[0] > 0).astype(int)
    counts = Counter(labels)
    k = max(2, sum(1 for c in counts.values() if c >= 0.1 * len(labels)))
    unit = grads / np.maximum(np.linalg.norm(grads, axis=1, keepdims=True), 1e-12)
    groups = kmeans(unit, k, rng)
    shuffled_split, shuffled_k = [], []
    for _ in range(shuffles):
        perm = [labels[i] for i in rng.permutation(len(labels))]
        shuffled_split.append(purity(split, perm))
        shuffled_k.append(purity(groups, perm))
    return {"top_share": round(top_share, 3), "split_purity": round(purity(split, labels), 3), "split_shuffled": round(float(np.mean(shuffled_split)), 3),
            "k": k, "kmeans_purity": round(purity(groups, labels), 3), "kmeans_shuffled": round(float(np.mean(shuffled_k)), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--train-file", type=Path, default=REPO / "data/babytheorist/many_p24_pairs2/lv1_train.h5")
    parser.add_argument("--per-primitive", type=int, default=150)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, labels = tasks(args.train_file, args.per_primitive, 0)
    x, y = x.to(device), y.to(device)
    n = len(labels)
    report = []
    for step in args.steps:
        ckpt = args.run / "ckpt" / f"step-{step:06d}.pt"
        if not ckpt.exists():
            print("missing", ckpt, flush=True)
            continue
        model, _, _, _ = load_run(args.run, ckpt, device)
        model.eval()
        size = model.code.size
        table = model.code.codebook(torch.arange(size, device=device)).detach()
        with torch.no_grad():
            states = [model.encoder(x[:, p]) for p in range(2)]
            loss = torch.zeros(n, size, device=device)
            chunk = max(1, 4096 // size)
            for start in range(0, n, chunk):
                rows = torch.arange(start, min(n, start + chunk), device=device)
                for p in range(2):
                    logits = model.decoder(model.executor(states[p][rows].repeat_interleave(size, dim=0), table.repeat(len(rows), 1)[:, None]))
                    part = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y[rows, p].repeat_interleave(size, dim=0).reshape(-1), reduction="none")
                    loss[rows] += part.view(len(rows), size, -1).sum(-1)
        assigned = loss.argmin(1)
        exact = torch.zeros(n, dtype=torch.bool, device=device)
        grad_code = torch.zeros(n, table.shape[1], device=device)
        grad_cond = torch.zeros(n, model.executor.stack.cond_proj.out_features, device=device)
        captured = {}

        def hook(_module, _inputs, output):
            output.retain_grad()
            captured["cond"] = output

        handle = model.executor.stack.cond_proj.register_forward_hook(hook)
        for start in range(0, n, 256):
            rows = torch.arange(start, min(n, start + 256), device=device)
            vectors = table[assigned[rows]].clone().requires_grad_(True)                      # (r, dim), one leaf per task
            source = torch.cat([states[0][rows], states[1][rows]])
            goal = torch.cat([y[rows, 0], y[rows, 1]])
            logits = model.decoder(model.executor(source, torch.cat([vectors, vectors])[:, None]))
            per_cell = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), goal.reshape(-1), reduction="none")
            per_cell.sum().backward()
            grad_code[rows] = vectors.grad.detach()
            cond = captured["cond"].grad.detach()
            grad_cond[rows] = cond[: len(rows)] + cond[len(rows):]
            with torch.no_grad():
                right = logits.argmax(-1).eq(goal).flatten(1).all(dim=1)
                exact[rows] = right[: len(rows)] & right[len(rows):]
            model.zero_grad(set_to_none=True)
        handle.remove()
        assigned_list, exact_list = assigned.cpu().tolist(), exact.cpu().tolist()
        gc, gd = grad_code.cpu().numpy(), grad_cond.cpu().numpy()
        rng = np.random.default_rng(step)
        codes = []
        for c in sorted(set(assigned_list)):
            idx = [i for i in range(n) if assigned_list[i] == c]
            names = [labels[i] for i in idx]
            counts = Counter(names)
            major = [p for p, v in counts.items() if v >= 0.1 * len(idx)]
            if len(idx) < 30 or len(major) < 2:
                continue
            entry = {"code": c, "tasks": len(idx), "composition": dict(counts.most_common()), "exact": round(float(np.mean([exact_list[i] for i in idx])), 3),
                     "swap_share": round(sum(counts[p] for p in SWAP) / len(idx), 3),
                     "code_vector": analyse(gc[idx], names, rng), "conditioning": analyse(gd[idx], names, rng)}
            # the same, on the orientation four alone, when the code holds at least two of them
            four = [i for i in idx if labels[i] in SWAP]
            if len({labels[i] for i in four}) >= 2 and len(four) >= 30:
                entry["four_only"] = {"tasks": len(four), "code_vector": analyse(gc[four], [labels[i] for i in four], rng),
                                      "conditioning": analyse(gd[four], [labels[i] for i in four], rng)}
            codes.append(entry)
        four_codes = Counter(assigned_list[i] for i in range(n) if labels[i] in SWAP)
        record = {"run": args.run.name, "step": step, "four_on_codes": dict(four_codes.most_common()), "mixed_codes": codes}
        report.append(record)
        line = f"{args.run.name} step {step}: the four on codes {dict(four_codes.most_common(4))}"
        for e in codes:
            if e["swap_share"] >= 0.25 and "four_only" in e:
                f = e["four_only"]
                line += (f" | code {e['code']} holds {e['tasks']} tasks, exact {e['exact']}; four only: code-vector top {f['code_vector']['top_share']}"
                         f" split {f['code_vector']['split_purity']}/{f['code_vector']['split_shuffled']} kmeans {f['code_vector']['kmeans_purity']}/{f['code_vector']['kmeans_shuffled']};"
                         f" conditioning split {f['conditioning']['split_purity']}/{f['conditioning']['split_shuffled']} kmeans {f['conditioning']['kmeans_purity']}/{f['conditioning']['kmeans_shuffled']}")
        print(line, flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
