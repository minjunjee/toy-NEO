#!/usr/bin/env python3
"""After a sleep that left part of the memory out (memory_sleep_data.py --hold-out): which codes learned a transformation
and which learned their memory tasks by heart?

For every held-out memory task the programmer names a code from the first pair; the task counts as reproduced when that
code reproduces both pairs. Per code: how many held-out tasks it was named for and the share it reproduced. The same is
done on the memory tasks the sleep trained on. A code that reproduces the tasks it trained on and not the held-out ones
has memorised. No label is read to flag a code; the hidden primitives of a flagged code's tasks are printed only to
check the flag against the truth.

    python scripts/babytheorist/memory_held_out_fit.py --run runs/sleep4_holdout_memory_m24_to_m52_s42 \
        --held data/babytheorist/memory_round4_hold_s42_rep/lv1_train.h5.held.json \
        --memory runs/queue/memory_online_refit_s42.pt ... --out runs/queue/memory_held_out_fit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "babytheorist"))

from support_search import code_table  # noqa: E402
from training.evaluate import load_run  # noqa: E402


@torch.no_grad()
def named_and_reproduced(model, table, x, y):
    action = model.programmer(model.encoder(x[:, 0]), model.encoder(y[:, 0]))
    code = (F.normalize(action.flatten(1), dim=-1) @ table.T).argmax(dim=1)
    ok = torch.ones(len(x), dtype=torch.bool, device=x.device)
    for pair in (0, 1):
        pred = model.decoder(model.executor(model.encoder(x[:, pair]), table[code][:, None])).argmax(dim=-1)
        ok &= pred.eq(y[:, pair]).flatten(1).all(dim=1)
    return code.tolist(), ok.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--held", type=Path, required=True)
    parser.add_argument("--memory", type=Path, nargs="+", required=True, help="the saved memories the train file was made from")
    parser.add_argument("--min-tasks", type=int, default=3, help="flag only codes named for at least this many held-out tasks")
    parser.add_argument("--below", type=float, default=0.5, help="flag a code whose held-out share is below this")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _ = load_run(args.run, None, device)
    model.eval()
    table = code_table(model)
    held = json.loads(args.held.read_text())
    memory = sorted({r for path in args.memory for code in torch.load(path, map_location="cpu", weights_only=False)["memory_rows"] for r in code})
    trained = sorted(set(memory) - set(held["rows"]))
    with h5py.File(held["eval"], "r") as f:
        inputs, outputs = torch.tensor(f["inputs"][:], dtype=torch.long), torch.tensor(f["outputs"][:], dtype=torch.long)
        names = [json.loads(t)[0]["primitive"] if len(json.loads(t)) == 1 else "?" for t in f["transformation_suites"][:]]
    stats = defaultdict(lambda: {"held": 0, "held_ok": 0, "trained": 0, "trained_ok": 0, "primitives": Counter()})
    for label, rows in (("held", held["rows"]), ("trained", trained)):
        codes, oks = named_and_reproduced(model, table, inputs[rows][:, :2].to(device), outputs[rows][:, :2].to(device))
        for row, code, ok in zip(rows, codes, oks):
            stats[code][label] += 1
            stats[code][label + "_ok"] += int(ok)
            if label == "held":
                stats[code]["primitives"][names[row]] += 1
    report = []
    for code, s in sorted(stats.items(), key=lambda kv: -kv[1]["held"]):
        if s["held"] < args.min_tasks:
            continue
        entry = {"code": code, "held_tasks": s["held"], "held_share": round(s["held_ok"] / s["held"], 2),
                 "trained_tasks": s["trained"], "trained_share": round(s["trained_ok"] / max(s["trained"], 1), 2),
                 "flagged": s["held_ok"] / s["held"] < args.below, "primitives": dict(s["primitives"].most_common(3))}
        report.append(entry)
    flagged = [e for e in report if e["flagged"]]
    clean = [e for e in report if not e["flagged"]]
    print(f"{args.run.name}: {len(report)} codes named for {args.min_tasks}+ held-out tasks; {len(flagged)} flagged")
    for e in flagged:
        print("  flagged", e)
    print("  not flagged:", [(e["code"], e["held_share"], list(e["primitives"])[:2]) for e in clean])
    saved = json.loads(args.out.read_text()) if args.out.exists() else []
    saved.append({"run": args.run.name, "min_tasks": args.min_tasks, "below": args.below, "codes": report})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(saved, indent=1) + "\n")


if __name__ == "__main__":
    main()
