#!/usr/bin/env python3
"""A frozen model meeting a larger world one task at a time, with a fast memory of new codes and no weight change.

Tasks of the larger environment's eval file (single-primitive tasks: two support pairs and a query) arrive one by one in a
fixed random order. For each task:
  1. verify: every code, the model's own and the new ones made so far, is run on both support pairs; the codes that
     reproduce both are the verified ones. If any, the query is answered with the first of them in the programmer's
     order (own codes) or, among new codes, the one with the most memory; if that code is a new one, the task's support
     pairs join its memory and the code is refitted on its whole memory (warm start, `--refit-steps`).
  2. otherwise allocate: a new code is fitted on the task's two support pairs (from zero and from the own codes that fit
     them best, `--starts`, keeping the fit with the lowest loss) and answers the query; the task is its first memory.
No label is used; the hidden primitive only scores the answers. Reported: accuracy over the stream for primitives the model
already knew and for new ones, in windows of the stream, and how many new codes were made, with which primitives they
ended up serving.

    python scripts/babytheorist/memory_online.py --run runs/onestep_s_m24_hardem_pairs2_price_s43 --small 24 \
        --eval data/babytheorist/many_p52_without4/lv1_eval.h5 --tasks 800 --out runs/queue/memory_online.json
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
sys.path.insert(0, str(REPO / "scripts" / "babytheorist"))

import generate_many_primitives as gen  # noqa: E402
from support_search import code_table  # noqa: E402
from training.evaluate import load_run  # noqa: E402


def fit(model, state, y, starts: list[torch.Tensor], steps: int, lr: float) -> torch.Tensor:
    best, best_loss = None, float("inf")
    for start in starts:
        v = start.clone().requires_grad_(True)
        opt = torch.optim.Adam([v], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            logits = model.decoder(model.executor(state, v.expand(len(state), -1)[:, None]))
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
            loss.backward()
            opt.step()
        if float(loss) < best_loss:
            best, best_loss = v.detach(), float(loss)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--small", default="24")
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=800)
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--refit-steps", type=int, default=100)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _ = load_run(args.run, None, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    own = code_table(model)
    known = set(gen.SETS[args.small])
    with h5py.File(args.eval, "r") as f:
        suites = [json.loads(t) for t in f["transformation_suites"][:]]
        rows = [i for i, s in enumerate(suites) if len(s) == 1]
        order = np.random.default_rng(args.seed).permutation(rows)[: args.tasks]
        order = order.tolist()
        xs = torch.tensor(f["inputs"][:][order], dtype=torch.long, device=device)   # h5py wants increasing indices; read all, then pick
        ys = torch.tensor(f["outputs"][:][order], dtype=torch.long, device=device)
        names = [suites[i][0]["primitive"] for i in order]
    new_codes, memory = [], []   # new code vectors and, for each, its memory (support inputs, outputs) and the primitives it served
    log = []
    for t in range(len(order)):
        sx, sy, qx, qy = xs[t, :2], ys[t, :2], xs[t, 2:3], ys[t, 2:3]
        with torch.no_grad():
            state = model.encoder(sx)
            table = torch.cat([own] + ([torch.stack(new_codes)] if new_codes else []))
            k = table.shape[0]
            out = model.decoder(model.executor(state.repeat(k, 1, 1), table.repeat_interleave(2, dim=0)[:, None])).argmax(dim=-1)
            ok = out.view(k, 2, *sy.shape[1:]).eq(sy[None]).flatten(2).all(dim=2).all(dim=1)
            # the programmer's order among the own codes
            action = model.programmer(state[:1], model.encoder(sy[:1]))
            sims = F.normalize(action.flatten(1), dim=-1) @ F.normalize(own, dim=-1).T
            rank = sims[0].argsort(descending=True).tolist()
        verified_own = [c for c in rank if bool(ok[c])]
        verified_new = [j for j in range(len(new_codes)) if bool(ok[own.shape[0] + j])]
        if verified_own:
            vector, kind, used = own[verified_own[0]], "own", None
        elif verified_new:
            used = max(verified_new, key=lambda j: len(memory[j]["x"]))
            memory[used]["x"].append(sx); memory[used]["y"].append(sy); memory[used]["served"][names[t]] += 1
            mx, my = torch.cat(memory[used]["x"]), torch.cat(memory[used]["y"])
            with torch.no_grad():
                mstate = model.encoder(mx)
            new_codes[used] = fit(model, mstate, my, [new_codes[used]], args.refit_steps, args.lr)
            vector, kind = new_codes[used], "reused new"
        else:
            with torch.no_grad():
                losses = []
                for c in range(own.shape[0]):
                    logits = model.decoder(model.executor(state, own[c].expand(2, -1)[:, None]))
                    losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), sy.reshape(-1))))
            starts = [own[c] for c in sorted(range(len(losses)), key=losses.__getitem__)[: args.starts]] + [torch.zeros_like(own[0])]
            vector = fit(model, state, sy, starts, args.fit_steps, args.lr)
            new_codes.append(vector)
            memory.append({"x": [sx], "y": [sy], "served": Counter({names[t]: 1})})
            kind, used = "allocated", len(new_codes) - 1
        with torch.no_grad():
            pred = model.decoder(model.executor(model.encoder(qx), vector[None, None])).argmax(dim=-1)
            right = bool(pred.eq(qy).all())
        log.append({"t": t, "primitive": names[t], "known": names[t] in known, "kind": kind, "right": right})
        if (t + 1) % 100 == 0:
            recent = log[-100:]
            acc = lambda rows: round(sum(r["right"] for r in rows) / max(len(rows), 1), 3)  # noqa: E731
            print(f"{t + 1} tasks: new codes {len(new_codes)}; last 100: known {acc([r for r in recent if r['known']])}, "
                  f"new {acc([r for r in recent if not r['known']])}; kinds {dict(Counter(r['kind'] for r in recent))}", flush=True)
    windows = []
    for start in range(0, len(log), 100):
        w = log[start: start + 100]
        windows.append({"tasks": f"{start + 1}-{start + len(w)}",
                        "known": round(np.mean([r["right"] for r in w if r["known"]]), 3) if any(r["known"] for r in w) else None,
                        "new": round(np.mean([r["right"] for r in w if not r["known"]]), 3) if any(not r["known"] for r in w) else None})
    served = [dict(m["served"].most_common(3)) for m in memory]
    new_prims = sorted({n for n in names if n not in known})
    report = json.loads(args.out.read_text()) if args.out.exists() else []
    report.append({"run": args.run.name, "tasks": len(log), "new_codes": len(new_codes), "new_primitives_seen": len(new_prims),
                   "windows": windows, "codes_served": served, "log": log})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print("new codes", len(new_codes), "for", len(new_prims), "new primitives; windows", windows, flush=True)


if __name__ == "__main__":
    main()
