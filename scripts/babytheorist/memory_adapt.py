#!/usr/bin/env python3
"""Adapting to a sibling world from a fast memory instead of the weights.

The trained model stays frozen. A memory holds N tasks of the new world (two pairs each, the first N of a train file).
1. Each memory task is filed under the code the programmer names for its first pair (no labels).
2. For every code with memory tasks, a correction is fitted so that the executor, run with that code plus the correction,
   reproduces those tasks' pairs (Adam, `--fit-steps` steps, a little weight decay). Where the correction is added
   (`--where`):
     code   to the code vector itself (16 numbers per code)
     cond   to the conditioning vector the executor makes from the code (cond_proj output, 128 numbers per code)
     mod    to every block's modulation, the scales, shifts and gates the executor makes from the code (6 x 128 numbers
            per block, 3,072 per code for four blocks)
   The executor's weights, the programmer and the code vectors do not change; the corrections are kept next to them.
3. At test time the programmer extracts a program from the first support pair as usual (greedy search over the original
   codes). The program is run twice on the supports, with the original codes and with each code's corrected version
   where one exists; the corrected version answers the query only where it reproduces every support pair and the
   original does not.

Reported on the new world's eval file and on the old world's (to see that nothing is lost there): transfer with the
original codes only, with the corrected codes always, with the verified choice, and the better of the two per task (an
upper bound on the choice), by hidden program length; and how many memory pairs each version reproduces.

    python scripts/babytheorist/memory_adapt.py --run runs/onestep_s_m24_hardem_pairs2_price_s43 --where mod \
        --memory data/babytheorist/many_p24_pairs2_large_n256/lv1_train.h5 --new data/babytheorist/many_p24_large/lv1_eval.h5 \
        --old data/babytheorist/many_p24/lv1_eval.h5 --out runs/queue/memory_adapt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "babytheorist"))

from support_search import beam_search, code_table, pad  # noqa: E402
from training.evaluate import load_run  # noqa: E402


class Corrections:
    """Per-code additive corrections applied inside the executor through forward hooks; `current` holds one row per sample."""

    def __init__(self, model, where: str, size: int, device) -> None:
        self.model, self.where = model, where
        stack = model.executor.stack
        if where == "cond":
            self.sites = [stack.cond_proj]
        elif where == "mod":
            self.sites = [block.adaLN_modulation for block in stack.blocks]
        else:
            self.sites = []
        self.widths = [site[-1].out_features if isinstance(site, torch.nn.Sequential) else site.out_features for site in self.sites]
        self.bank = [torch.zeros(size, w, device=device) for w in self.widths]   # one row per code, zero where no correction
        self.current = None
        self.handles = [site.register_forward_hook(self._hook(k)) for k, site in enumerate(self.sites)]

    def _hook(self, k):
        def add(_module, _inputs, output):
            return output if self.current is None else output + self.current[k]
        return add

    def set(self, rows: list[torch.Tensor] | None) -> None:
        self.current = rows


@torch.no_grad()
def name_codes(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    action = model.programmer(model.encoder(x), model.encoder(y))
    return model.code(action, False).indices.view(x.shape[0], -1)[:, 0]


def execute(model, table: torch.Tensor, corr: Corrections, state: torch.Tensor, codes: torch.Tensor, corrected: torch.Tensor) -> torch.Tensor:
    """Executor logits for each state with its code; `corrected` marks rows that use their code's correction."""

    vectors = table[codes].clone()
    if corr.where == "code":
        vectors = vectors + corr.code_bank[codes] * corrected[:, None]
        corr.set(None)
    else:
        corr.set([bank[codes] * corrected[:, None] for bank in corr.bank])
    try:
        return model.decoder(model.executor(state, vectors[:, None]))
    finally:
        corr.set(None)


def fit_shared(model, table: torch.Tensor, corr: Corrections, mx: torch.Tensor, my: torch.Tensor, codes: torch.Tensor, steps: int, lr: float, decay: float,
               batch: int = 256) -> None:
    """One correction for the whole new world, fitted on every memory task with its own code, then given to every code."""

    x = torch.cat([mx[:, p] for p in range(mx.shape[1])])
    y = torch.cat([my[:, p] for p in range(my.shape[1])])
    c = torch.cat([codes] * mx.shape[1])
    with torch.no_grad():
        state = model.encoder(x)
    widths = [table.shape[1]] if corr.where == "code" else corr.widths
    deltas = [torch.zeros(w, device=x.device, requires_grad=True) for w in widths]
    opt = torch.optim.Adam(deltas, lr=lr, weight_decay=decay)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(steps):
        rows = torch.randperm(len(x), generator=gen)[:batch].to(x.device)
        opt.zero_grad()
        if corr.where == "code":
            corr.set(None)
            logits = model.decoder(model.executor(state[rows], (table[c[rows]] + deltas[0])[:, None]))
        else:
            corr.set([d.expand(len(rows), -1) for d in deltas])
            logits = model.decoder(model.executor(state[rows], table[c[rows]][:, None]))
            corr.set(None)
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y[rows].reshape(-1)).backward()
        opt.step()
    if corr.where == "code":
        corr.code_bank[:] = deltas[0].detach()
    else:
        for bank, d in zip(corr.bank, deltas):
            bank[:] = d.detach()


def fit(model, table: torch.Tensor, corr: Corrections, mx: torch.Tensor, my: torch.Tensor, codes: torch.Tensor, steps: int, lr: float, decay: float) -> None:
    for c in sorted(set(codes.tolist())):
        rows = torch.nonzero(codes == c).flatten()
        x = torch.cat([mx[rows, p] for p in range(mx.shape[1])])
        y = torch.cat([my[rows, p] for p in range(my.shape[1])])
        with torch.no_grad():
            state = model.encoder(x)
        widths = [table.shape[1]] if corr.where == "code" else corr.widths
        deltas = [torch.zeros(w, device=x.device, requires_grad=True) for w in widths]
        opt = torch.optim.Adam(deltas, lr=lr, weight_decay=decay)
        for _ in range(steps):
            opt.zero_grad()
            if corr.where == "code":
                corr.set(None)
                logits = model.decoder(model.executor(state, (table[c] + deltas[0]).expand(len(x), -1)[:, None]))
            else:
                corr.set([d.expand(len(x), -1) for d in deltas])
                logits = model.decoder(model.executor(state, table[c].expand(len(x), -1)[:, None]))
                corr.set(None)
            F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1)).backward()
            opt.step()
        if corr.where == "code":
            corr.code_bank[c] = deltas[0].detach()
        else:
            for bank, d in zip(corr.bank, deltas):
                bank[c] = d.detach()


@torch.no_grad()
def memory_fit(model, table, corr, mx, my, codes) -> tuple[float, float]:
    right, total = [0, 0], 0
    for c in sorted(set(codes.tolist())):
        rows = torch.nonzero(codes == c).flatten()
        x = torch.cat([mx[rows, p] for p in range(mx.shape[1])])
        y = torch.cat([my[rows, p] for p in range(my.shape[1])])
        state = model.encoder(x)
        idx = torch.full((len(x),), c, device=x.device)
        for k in range(2):
            pred = execute(model, table, corr, state, idx, torch.full((len(x),), float(k), device=x.device)).argmax(dim=-1)
            right[k] += int(pred.eq(y).flatten(1).all(dim=1).sum())
        total += len(x)
    return right[0] / total, right[1] / total


@torch.no_grad()
def run_program(model, table, corr, grids: torch.Tensor, programs: torch.Tensor, use: torch.Tensor) -> torch.Tensor:
    grids = grids.clone()
    for t in range(programs.shape[1]):
        codes = programs[:, t]
        active = codes >= 0
        if not active.any():
            break
        rows = torch.nonzero(active).flatten()
        logits = execute(model, table, corr, model.encoder(grids[rows]), codes[rows], use[rows].float())
        grids[rows] = logits.argmax(dim=-1).to(grids.dtype)
    return grids


@torch.no_grad()
def evaluate(model, table, corr, has: torch.Tensor, path: Path, max_samples: int, max_steps: int, device) -> dict:
    with h5py.File(path, "r") as f:
        n = min(max_samples, len(f["depths"]))
        xs, ys = torch.tensor(f["inputs"][:n], dtype=torch.long), torch.tensor(f["outputs"][:n], dtype=torch.long)
        depth = torch.tensor(f["depths"][:n])
    counts = {}
    for start in range(0, n, 256):
        rows = slice(start, min(n, start + 256))
        sx, sy = xs[rows, :-1].to(device), ys[rows, :-1].to(device)
        qx, qy = xs[rows, -1].to(device), ys[rows, -1].to(device)
        found, fallback = beam_search(model, table, sx[:, 0], sy[:, 0], 1, max_steps)
        programs = torch.tensor(pad([list(p[0]) if p else [c for c in fallback[i].tolist() if c >= 0] for i, p in enumerate(found)], max_steps), device=device)
        m = programs.shape[0]
        off, on = torch.zeros(m, dtype=torch.bool, device=device), torch.ones(m, dtype=torch.bool, device=device)

        def passes(use):
            ok = torch.ones(m, dtype=torch.bool, device=device)
            for j in range(sx.shape[1]):
                ok &= run_program(model, table, corr, sx[:, j], programs, use).eq(sy[:, j]).flatten(1).all(dim=1)
            return ok

        base_ok, fixed_ok = passes(off), passes(on)
        base_q = run_program(model, table, corr, qx, programs, off).eq(qy).flatten(1).all(dim=1)
        fixed_q = run_program(model, table, corr, qx, programs, on).eq(qy).flatten(1).all(dim=1)
        chosen = torch.where(fixed_ok & ~base_ok, fixed_q, base_q)
        for i in range(m):
            d = int(depth[start + i])
            e = counts.setdefault(d, {"tasks": 0, "original": 0, "corrected": 0, "verified": 0, "better_of_two": 0})
            e["tasks"] += 1
            e["original"] += int(base_q[i]); e["corrected"] += int(fixed_q[i])  # noqa: E702
            e["verified"] += int(chosen[i]); e["better_of_two"] += int(base_q[i] | fixed_q[i])  # noqa: E702
    total = {k: sum(e[k] for e in counts.values()) for k in ("tasks", "original", "corrected", "verified", "better_of_two")}
    share = lambda e: {k: (round(v / e["tasks"], 3) if k != "tasks" else v) for k, v in e.items()}  # noqa: E731
    return {"by_length": {str(d): share(e) for d, e in sorted(counts.items())}, "all": share(total)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--where", choices=("code", "cond", "mod"), default="mod")
    parser.add_argument("--shared", action="store_true", help="one correction for the whole new world instead of one per code")
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--decay", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, step = load_run(args.run, None, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    table = code_table(model)
    corr = Corrections(model, args.where, table.shape[0], device)
    corr.code_bank = torch.zeros_like(table)
    with h5py.File(args.memory, "r") as f:
        mx, my = torch.tensor(f["inputs"][:], dtype=torch.long, device=device), torch.tensor(f["outputs"][:], dtype=torch.long, device=device)
    codes = name_codes(model, mx[:, 0], my[:, 0])
    (fit_shared if args.shared else fit)(model, table, corr, mx, my, codes, args.fit_steps, args.lr, args.decay)
    has = torch.zeros(table.shape[0], dtype=torch.bool, device=device)
    has[codes.unique()] = True
    if args.shared:
        has[:] = True
    before, after = memory_fit(model, table, corr, mx, my, codes)
    entry = {"run": args.run.name, "where": args.where, "shared": args.shared, "memory": str(args.memory), "memory_tasks": len(mx), "codes_with_memory": int(has.sum()),
             "fit_steps": args.fit_steps, "lr": args.lr, "decay": args.decay, "memory_reproduced_original": round(before, 3), "memory_reproduced_corrected": round(after, 3),
             "new_world": evaluate(model, table, corr, has, args.new, args.max_samples, args.max_steps, device),
             "old_world": evaluate(model, table, corr, has, args.old, args.max_samples, args.max_steps, device)}
    report = json.loads(args.out.read_text()) if args.out.exists() else []
    report.append(entry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(args.run.name, args.where, "shared" if args.shared else "per code", f"memory {len(mx)} tasks, reproduced {before:.2f} -> {after:.2f}", "new", entry["new_world"]["all"], "old", entry["old_world"]["all"], flush=True)


if __name__ == "__main__":
    main()
