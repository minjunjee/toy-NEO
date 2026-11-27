#!/usr/bin/env python3
"""A frozen model meeting a larger world one task at a time, with a fast memory of new codes and no weight change.

Tasks of the larger environment's eval file (single-primitive tasks: two support pairs and a query) arrive one by one in a
fixed random order. For each task:
  1. verify: every code, the model's own and the new ones made so far, is run on both support pairs; the codes that
     reproduce both are the verified ones. If any, the query is answered with the first of them in the programmer's
     order (own codes) or, among new codes, the one with the most memory; if that code is a new one, the task's support
     pairs join its memory and the code is refitted on its whole memory (warm start, `--refit-steps`).
  2. otherwise, with `--reuse-refit M`, extend a new code: the M new codes with the lowest loss on the task's supports are
     refitted in turn on their memory plus the task (warm start, `--refit-steps`); the first that then reproduces the task
     and at least `--accept` of its earlier memory tasks keeps the refit, takes the task into its memory and answers.
     This is the order of the research principles: reuse what exists before allocating.
  3. otherwise allocate: a new code is fitted on the task's two support pairs (from zero and from the own codes that fit
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
import time
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


def fit(model, state, y, starts: list[torch.Tensor], steps: int, lr: float, unit: bool = False) -> torch.Tensor:
    """Fit code vectors from several starting points at once (one batch, one Adam per row, so the rows do not interact);
    return the row with the lowest reproduction loss at the last step. With `unit`, the executor always gets the vector
    scaled to length 1, as the codebook gives its codes, and the returned vector has length 1."""
    v = torch.stack(starts).clone().requires_grad_(True)
    s, n = v.shape[0], state.shape[0]
    big_state, big_y = state.repeat(s, *([1] * (state.dim() - 1))), y.repeat(s, *([1] * (y.dim() - 1)))
    opt = torch.optim.Adam([v], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        code = F.normalize(v, dim=-1) if unit else v
        logits = model.decoder(model.executor(big_state, code.repeat_interleave(n, dim=0)[:, None]))
        per = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), big_y.reshape(-1), reduction="none").view(s, -1).mean(dim=1)
        per.sum().backward()
        opt.step()
    best = v.detach()[int(per.argmin())]
    return F.normalize(best, dim=-1) if unit else best


def reproduced_tasks(model, state, y, vector: torch.Tensor) -> torch.Tensor:
    """For memory pairs stored two per task, which tasks the code reproduces on both pairs."""
    with torch.no_grad():
        out = model.decoder(model.executor(state, vector.expand(len(state), -1)[:, None])).argmax(dim=-1)
    return out.eq(y).flatten(1).all(dim=1).view(-1, 2).all(dim=1)


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
    parser.add_argument("--reuse-refit", type=int, default=0, help="before allocating, refit up to this many new codes (lowest loss on the task first) "
                        "on their memory plus the task, and keep the first that reproduces the task and --accept of its memory")
    parser.add_argument("--accept", type=float, default=0.75)
    parser.add_argument("--skip", type=int, default=0, help="leave out this many tasks at the start of the fixed order (a later part of the same stream)")
    parser.add_argument("--save-memory", type=Path, help="save the new code vectors and, for each, the eval rows of its memory tasks (for a sleep that consolidates them)")
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
        order = np.random.default_rng(args.seed).permutation(rows)[args.skip: args.skip + args.tasks]
        order = order.tolist()
        xs = torch.tensor(f["inputs"][:][order], dtype=torch.long, device=device)   # h5py wants increasing indices; read all, then pick
        ys = torch.tensor(f["outputs"][:][order], dtype=torch.long, device=device)
        names = [suites[i][0]["primitive"] for i in order]
    new_codes, memory = [], []   # new code vectors and, for each, its memory (support inputs, outputs) and the primitives it served
    log = []
    began = time.time()
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
            memory[used]["x"].append(sx); memory[used]["y"].append(sy); memory[used]["served"][names[t]] += 1; memory[used]["t"].append(t)
            mx, my = torch.cat(memory[used]["x"]), torch.cat(memory[used]["y"])
            with torch.no_grad():
                mstate = model.encoder(mx)
            new_codes[used] = fit(model, mstate, my, [new_codes[used]], args.refit_steps, args.lr)
            vector, kind = new_codes[used], "reused new"
        else:
            reused = None
            if args.reuse_refit and new_codes:
                with torch.no_grad():
                    news = torch.stack(new_codes)
                    logits = model.decoder(model.executor(state.repeat(len(news), 1, 1), news.repeat_interleave(2, dim=0)[:, None]))
                    loss_new = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), sy.repeat(len(news), 1, 1).reshape(-1),
                                               reduction="none").view(len(news), -1).mean(dim=1)
                for j in loss_new.argsort()[: args.reuse_refit].tolist():
                    mx, my = torch.cat(memory[j]["x"] + [sx]), torch.cat(memory[j]["y"] + [sy])
                    with torch.no_grad():
                        mstate = model.encoder(mx)
                    candidate = fit(model, mstate, my, [new_codes[j]], args.refit_steps, args.lr)
                    done = reproduced_tasks(model, mstate, my, candidate)
                    if bool(done[-1]) and float(done[:-1].float().mean()) >= args.accept:
                        reused = j
                        new_codes[j] = candidate
                        memory[j]["x"].append(sx); memory[j]["y"].append(sy); memory[j]["served"][names[t]] += 1; memory[j]["t"].append(t)
                        break
            if reused is not None:
                vector, kind, used = new_codes[reused], "refit new", reused
            else:
                with torch.no_grad():
                    logits = model.decoder(model.executor(state.repeat(own.shape[0], 1, 1), own.repeat_interleave(2, dim=0)[:, None]))
                    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), sy.repeat(own.shape[0], 1, 1).reshape(-1),
                                             reduction="none").view(own.shape[0], -1).mean(dim=1)
                starts = [own[c] for c in losses.argsort()[: args.starts].tolist()] + [torch.zeros_like(own[0])]
                vector = fit(model, state, sy, starts, args.fit_steps, args.lr)
                new_codes.append(vector)
                memory.append({"x": [sx], "y": [sy], "served": Counter({names[t]: 1}), "t": [t]})
                kind, used = "allocated", len(new_codes) - 1
        with torch.no_grad():
            pred = model.decoder(model.executor(model.encoder(qx), vector[None, None])).argmax(dim=-1)
            right = bool(pred.eq(qy).all())
        log.append({"t": t, "primitive": names[t], "known": names[t] in known, "kind": kind, "right": right})
        if (t + 1) % 100 == 0:
            recent = log[-100:]
            acc = lambda rows: round(sum(r["right"] for r in rows) / max(len(rows), 1), 3)  # noqa: E731
            print(f"{t + 1} tasks: new codes {len(new_codes)}; last 100: known {acc([r for r in recent if r['known']])}, "
                  f"new {acc([r for r in recent if not r['known']])}; kinds {dict(Counter(r['kind'] for r in recent))}; {time.time() - began:.0f} s", flush=True)
    windows = []
    for start in range(0, len(log), 100):
        w = log[start: start + 100]
        windows.append({"tasks": f"{start + 1}-{start + len(w)}",
                        "known": round(np.mean([r["right"] for r in w if r["known"]]), 3) if any(r["known"] for r in w) else None,
                        "new": round(np.mean([r["right"] for r in w if not r["known"]]), 3) if any(not r["known"] for r in w) else None})
    served = [dict(m["served"].most_common(3)) for m in memory]
    new_prims = sorted({n for n in names if n not in known})
    report = json.loads(args.out.read_text()) if args.out.exists() else []
    report.append({"run": args.run.name, "skip": args.skip, "reuse_refit": args.reuse_refit, "accept": args.accept, "fit_steps": args.fit_steps,
                   "refit_steps": args.refit_steps, "tasks": len(log), "new_codes": len(new_codes), "new_primitives_seen": len(new_prims),
                   "windows": windows, "codes_served": served, "log": log})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    if args.save_memory:
        torch.save({"run": str(args.run), "eval": str(args.eval), "rows": order, "vectors": torch.stack(new_codes).cpu() if new_codes else None,
                    "memory_rows": [[order[t] for t in m["t"]] for m in memory], "log": log}, args.save_memory)
    print("new codes", len(new_codes), "for", len(new_prims), "new primitives; windows", windows, flush=True)


if __name__ == "__main__":
    main()
