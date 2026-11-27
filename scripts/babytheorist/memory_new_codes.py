#!/usr/bin/env python3
"""Can a frozen model take on a new primitive from a few examples by a new code alone? A diagnostic that reads labels.

The model of a smaller environment stays frozen. For each primitive of the larger set that the smaller one lacks, k train
tasks of that primitive (two pairs each, from the larger set's two-pair train file; the hidden primitive only groups them)
fit a new code vector (from zero, or from the existing code that best reproduces the fitting pairs) by Adam on the executor's reproduction loss; nothing else changes. The new code is then
run on that primitive's held-out eval tasks (the query of single-primitive tasks of the larger set's eval file).

Two references per primitive: the best of the frozen model's existing codes on the same held-out queries (what reuse alone
gives), and the share of the k fitting tasks the new code reproduces (does the code fit at all).

    python scripts/babytheorist/memory_new_codes.py --run runs/onestep_s_m24_hardem_pairs2_price_s43 --small 24 --large 52 \
        --train data/babytheorist/many_p52_pairs2/lv1_train.h5 --eval data/babytheorist/many_p52_without4/lv1_eval.h5 \
        --shots 4 16 64 --out runs/queue/memory_new_codes.json
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

import generate_many_primitives as gen  # noqa: E402
from support_search import code_table  # noqa: E402
from training.evaluate import load_run  # noqa: E402


def by_primitive(path: Path, names: set[str], depth_one: bool) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    with h5py.File(path, "r") as f:
        prims = [json.loads(t)[0]["primitive"] if len(json.loads(t)) == 1 else None for t in f["transformation_suites"][:]]
        depths = f["depths"][:]
        out = {}
        for name in names:
            rows = [i for i, p in enumerate(prims) if p == name and (not depth_one or depths[i] == 1)]
            if rows:
                out[name] = (torch.tensor(f["inputs"][rows], dtype=torch.long), torch.tensor(f["outputs"][rows], dtype=torch.long))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--small", default="24")
    parser.add_argument("--large", default="52")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--shots", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--init", choices=("zero", "nearest"), default="zero", help="start the new code from zero or from the existing codes that fit best")
    parser.add_argument("--starts", type=int, default=1, help="with --init nearest: how many of the best-fitting existing codes to start from (zero is always one more)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _ = load_run(args.run, None, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    table = code_table(model)
    new = sorted(set(gen.SETS[args.large]) - set(gen.SETS[args.small]))
    train = by_primitive(args.train, set(new), False)
    test = by_primitive(args.eval, set(new), True)
    results = {}
    for name in new:
        if name not in train or name not in test:
            continue
        ex, ey = test[name]
        qx, qy = ex[:, -1].to(device), ey[:, -1].to(device)
        with torch.no_grad():
            qstate = model.encoder(qx)
            best_old = 0.0
            for c in range(table.shape[0]):
                pred = model.decoder(model.executor(qstate, table[c].expand(len(qx), -1)[:, None])).argmax(dim=-1)
                best_old = max(best_old, float(pred.eq(qy).flatten(1).all(dim=1).float().mean()))
        entry = {"held_out_tasks": len(qx), "best_existing_code": round(best_old, 3)}
        tx, ty = train[name]
        for k in args.shots:
            x = torch.cat([tx[:k, p] for p in range(tx.shape[1])]).to(device)
            y = torch.cat([ty[:k, p] for p in range(ty.shape[1])]).to(device)
            with torch.no_grad():
                state = model.encoder(x)
            # starting points: zero, or the existing codes that best reproduce the fitting pairs; each start is fitted and
            # the result with the lowest loss on the fitting pairs is kept (chosen without the held-out tasks)
            with torch.no_grad():
                losses = []
                for c in range(table.shape[0]):
                    logits = model.decoder(model.executor(state, table[c].expand(len(x), -1)[:, None]))
                    losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))))
            order = sorted(range(len(losses)), key=losses.__getitem__)
            starts = [table[c].clone() for c in order[:args.starts]] if args.init == "nearest" else []
            starts.append(torch.zeros(table.shape[1], device=device))
            best_loss, vector = float("inf"), None
            for start_vector in starts:
                candidate = start_vector.clone().requires_grad_(True)
                opt = torch.optim.Adam([candidate], lr=args.lr)
                for _ in range(args.fit_steps):
                    opt.zero_grad()
                    logits = model.decoder(model.executor(state, candidate.expand(len(x), -1)[:, None]))
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
                    loss.backward()
                    opt.step()
                if float(loss) < best_loss:
                    best_loss, vector = float(loss), candidate.detach()
            with torch.no_grad():
                fit = float(model.decoder(model.executor(state, vector.expand(len(x), -1)[:, None])).argmax(-1).eq(y).flatten(1).all(dim=1).float().mean())
                held = float(model.decoder(model.executor(qstate, vector.expand(len(qx), -1)[:, None])).argmax(-1).eq(qy).flatten(1).all(dim=1).float().mean())
            entry[f"k{k}"] = {"fit": round(fit, 3), "held_out": round(held, 3)}
        results[name] = entry
        print(name.ljust(16), entry, flush=True)
    summary = {f"k{k}": round(sum(e[f"k{k}"]["held_out"] for e in results.values()) / len(results), 3) for k in args.shots}
    summary["best_existing_code"] = round(sum(e["best_existing_code"] for e in results.values()) / len(results), 3)
    report = json.loads(args.out.read_text()) if args.out.exists() else []
    report.append({"run": args.run.name, "small": args.small, "large": args.large, "fit_steps": args.fit_steps, "lr": args.lr, "init": args.init, "summary": summary, "primitives": results})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print("mean over new primitives:", summary, flush=True)


if __name__ == "__main__":
    main()
