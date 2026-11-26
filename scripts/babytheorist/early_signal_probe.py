#!/usr/bin/env python3
"""Where, early in training, is there any signal that tells the four orientation primitives that swap width and height
apart? Five readings per checkpoint of a two-pair hard EM run, on train tasks of the four (two pairs each, from the two-pair
train file). The hidden primitive is used only to score the readings, never by the model.

  assignment      the code hard EM gives each task (smallest reconstruction loss summed over both pairs, no bonus), and the
                  code it would give from the first pair alone (what a single-pair learner sees); purity of these groups
                  with respect to the primitive, next to the same with the primitives shuffled
  executor        for each of the four, the code that reproduces it most often on its first pairs, how often, and how
                  often that same code reproduces the other three (a code that does one of them and not the others is
                  where a split could start)
  gradient        the gradient of each pair's loss with respect to the code vector the task was given, one per pair:
                  agreement within a task (cosine between the two pairs' gradients, same transformation on two objects)
                  against agreement between tasks on the same code (pairs drawn from different tasks); and, as a
                  diagnostic, agreement between tasks of the same primitive against tasks of different primitives
  programmer      the programmer's output before it is quantized to a code (16 numbers) and its hidden vector before
                  the output layer (256 numbers), from the first pair only: k-means purity with k = 4 next to shuffled, and
                  the accuracy of a linear classifier (5-fold, chance 0.25) as an upper bound on what the vector holds
  nucleus         codes holding at least 20 of the four's tasks with at least 60% from one primitive (by the hard EM
                  assignment)

    python scripts/babytheorist/early_signal_probe.py --run runs/onestep_s_m24_hardem_pairs2_ckpt_s42 --steps 200 400 \
        --out runs/queue/early_signal_s42.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "babytheorist"))

from gradient_split_probe import SWAP, kmeans, purity, tasks  # noqa: E402
from training.evaluate import load_run  # noqa: E402


def shuffled(groups: np.ndarray, labels: list[str], rng: np.random.Generator, times: int = 20) -> float:
    return float(np.mean([purity(groups, [labels[i] for i in rng.permutation(len(labels))]) for _ in range(times)]))


def linear_probe(features: np.ndarray, labels: list[str], rng: np.random.Generator, folds: int = 5) -> float:
    names = sorted(set(labels))
    y = torch.tensor([names.index(n) for n in labels])
    x = torch.tensor(features, dtype=torch.float32)
    x = (x - x.mean(0)) / x.std(0).clamp(min=1e-6)
    order = torch.tensor(rng.permutation(len(labels)))
    right = 0
    for f in range(folds):
        test = order[f::folds]
        train = torch.tensor(sorted(set(order.tolist()) - set(test.tolist())))
        w = torch.zeros(x.shape[1], len(names), requires_grad=True)
        b = torch.zeros(len(names), requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=1e-3)
        for _ in range(300):
            opt.zero_grad()
            F.cross_entropy(x[train] @ w + b, y[train]).backward()
            opt.step()
        right += int(((x[test] @ w + b).argmax(1) == y[test]).sum())
    return right / len(labels)


def mean_cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return float((a * b).sum(1).mean())


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
    keep = [i for i, n in enumerate(labels) if n in SWAP]
    x, y, labels = x[keep].to(device), y[keep].to(device), [labels[i] for i in keep]
    n = len(labels)
    report = json.loads(args.out.read_text()) if args.out.exists() else []
    done = {r["step"] for r in report}
    for step in args.steps:
        ckpt = args.run / "ckpt" / f"step-{step:06d}.pt"
        if step in done or not ckpt.exists():
            continue
        model, _, _, _ = load_run(args.run, ckpt, device)
        model.eval()
        size = model.code.size
        table = model.code.codebook(torch.arange(size, device=device)).detach()
        rng = np.random.default_rng(step)
        with torch.no_grad():
            states = [model.encoder(x[:, p]) for p in range(2)]
            loss = torch.zeros(2, n, size, device=device)
            hits = torch.zeros(2, n, size, dtype=torch.bool, device=device)
            chunk = max(1, 4096 // size)
            for start in range(0, n, chunk):
                rows = torch.arange(start, min(n, start + chunk), device=device)
                for p in range(2):
                    logits = model.decoder(model.executor(states[p][rows].repeat_interleave(size, dim=0), table.repeat(len(rows), 1)[:, None]))
                    goal = y[rows, p].repeat_interleave(size, dim=0)
                    part = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), goal.reshape(-1), reduction="none")
                    loss[p, rows] = part.view(len(rows), size, -1).sum(-1)
                    hits[p, rows] = logits.argmax(-1).eq(goal).flatten(1).all(dim=1).view(len(rows), size)
            hidden = {}
            handle = model.programmer.output.register_forward_hook(lambda m, i, o: hidden.__setitem__("h", i[0].detach()))
            action = model.programmer(states[0], states[1]).flatten(1)
            handle.remove()
        assigned = (loss[0] + loss[1]).argmin(1)
        single = loss[0].argmin(1)
        out = {"run": args.run.name, "step": step}
        a, s = assigned.cpu().numpy(), single.cpu().numpy()
        out["assignment"] = {"two_pair_purity": round(purity(a, labels), 3), "two_pair_shuffled": round(shuffled(a, labels, rng), 3),
                             "one_pair_purity": round(purity(s, labels), 3), "one_pair_shuffled": round(shuffled(s, labels, rng), 3),
                             "codes": dict(Counter(a.tolist()).most_common(6))}
        executor = {}
        h = hits[0].float().cpu().numpy()
        for name in SWAP:
            mine = [i for i in range(n) if labels[i] == name]
            others = [i for i in range(n) if labels[i] != name]
            rate = h[mine].mean(0)
            best = int(rate.argmax())
            executor[name] = {"code": best, "own": round(float(rate[best]), 3), "others": round(float(h[others][:, best].mean()), 3)}
        out["executor"] = executor
        # gradients of each pair's loss with respect to the code vector the task was given
        grads = torch.zeros(2, n, table.shape[1], device=device)
        for start in range(0, n, 256):
            rows = torch.arange(start, min(n, start + 256), device=device)
            for p in range(2):
                vectors = table[assigned[rows]].clone().requires_grad_(True)
                logits = model.decoder(model.executor(states[p][rows], vectors[:, None]))
                F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y[rows, p].reshape(-1), reduction="sum").backward()
                grads[p, rows] = vectors.grad.detach()
                model.zero_grad(set_to_none=True)
        g1, g2 = grads[0].cpu().numpy(), grads[1].cpu().numpy()
        within, between, same_prim, diff_prim = [], [], [], []
        for c in set(a.tolist()):
            idx = np.nonzero(a == c)[0]
            if len(idx) < 10:
                continue
            within.append(mean_cos(g1[idx], g2[idx]) * len(idx))
            partner = idx[rng.permutation(len(idx))]
            between.append(mean_cos(g1[idx], g2[partner]) * len(idx))
            for i in idx[:60]:
                for j in idx[:60]:
                    if i < j:
                        cos = float(g1[i] @ g1[j] / max(np.linalg.norm(g1[i]) * np.linalg.norm(g1[j]), 1e-12))
                        (same_prim if labels[i] == labels[j] else diff_prim).append(cos)
        counted = sum(len(np.nonzero(a == c)[0]) for c in set(a.tolist()) if len(np.nonzero(a == c)[0]) >= 10)
        out["gradient"] = {"within_task": round(sum(within) / max(counted, 1), 3), "between_tasks": round(sum(between) / max(counted, 1), 3),
                           "same_primitive": round(float(np.mean(same_prim)), 3) if same_prim else None,
                           "different_primitive": round(float(np.mean(diff_prim)), 3) if diff_prim else None}
        shared = g1 / np.maximum(np.linalg.norm(g1, axis=1, keepdims=True), 1e-12) + g2 / np.maximum(np.linalg.norm(g2, axis=1, keepdims=True), 1e-12)
        groups = kmeans(shared, 4, rng)
        out["gradient"]["shared_kmeans_purity"] = round(purity(groups, labels), 3)
        out["gradient"]["shared_kmeans_shuffled"] = round(shuffled(groups, labels, rng), 3)
        prog = {}
        for key, feats in (("action", action.float().cpu().numpy()), ("hidden", hidden["h"].float().cpu().numpy())):
            unit = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
            groups = kmeans(unit, 4, rng)
            prog[key] = {"kmeans_purity": round(purity(groups, labels), 3), "kmeans_shuffled": round(shuffled(groups, labels, rng), 3),
                         "linear_probe": round(linear_probe(feats, labels, rng), 3)}
        out["programmer"] = prog
        nucleus = []
        for c, count in Counter(a.tolist()).items():
            if count >= 20:
                top, top_n = Counter(labels[i] for i in range(n) if a[i] == c).most_common(1)[0]
                if top_n / count >= 0.6:
                    nucleus.append({"code": int(c), "primitive": top, "tasks": count, "share": round(top_n / count, 2)})
        out["nucleus"] = nucleus
        report.append(out)
        report.sort(key=lambda r: r["step"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        ex = " ".join(f"{k[:5]} c{v['code']} {v['own']}/{v['others']}" for k, v in executor.items())
        print(f"{args.run.name} {step}: assign 2p {out['assignment']['two_pair_purity']}/{out['assignment']['two_pair_shuffled']} 1p {out['assignment']['one_pair_purity']}/{out['assignment']['one_pair_shuffled']}"
              f" | exec {ex} | grad within {out['gradient']['within_task']} between {out['gradient']['between_tasks']} same {out['gradient']['same_primitive']} diff {out['gradient']['different_primitive']}"
              f" shared {out['gradient']['shared_kmeans_purity']}/{out['gradient']['shared_kmeans_shuffled']}"
              f" | prog action {prog['action']['kmeans_purity']}/{prog['action']['kmeans_shuffled']} probe {prog['action']['linear_probe']}"
              f" hidden {prog['hidden']['kmeans_purity']}/{prog['hidden']['kmeans_shuffled']} probe {prog['hidden']['linear_probe']} | nucleus {nucleus}", flush=True)


if __name__ == "__main__":
    main()
