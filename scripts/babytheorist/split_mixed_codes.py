#!/usr/bin/env python3
"""Give the codes that explain their pairs badly some fresh codes next to them, as the first half of a sleep step.

The diagnosis (docs/reports/2026-11-22-scaling.html): in the 24-primitive environment the programmer sends four
orientation primitives to the same two codes, so each of those codes stands for a mixture and the executor can fit
none of them; with one code per primitive given as a label the same executor does all seven at 83-98%. Choosing the
code by executing the nearest candidates did not help, because every candidate near those pairs was itself a mixture.

This script uses no label. On `--pairs` training pairs it takes the code the programmer names for each pair and checks
whether that code, applied once, reproduces the pair exactly. A code is called mixed when it is named for at least
`--min-share` of the pairs and reproduces fewer than `--max-rate` of them. Each mixed code gets `--children` codes that
were not in use (named for fewer than `--unused` of the pairs): every child is the mixed code's vector turned by a small
random amount (`--noise`, then normalised), so the children start right next to the code they are to split. Mixed codes
with more badly explained pairs are served first while unused codes last. Writes a checkpoint and a report.

The second half is training from that checkpoint with the code chosen by executing the nearest candidates
(models/neo_compete.py) while the code vectors stay fixed, so the children cannot be pulled back onto their parent; the
executor learns what each child does from the pairs it wins, and the programmer learns to name the winner.

Children placed next to the parent (the default) inherited the parent's mixture: the executor could not make two nearly
equal code vectors do different things. With `--random` the children are fresh random vectors far from everything, and
the competition is told the family (the report's `children`, passed to neo_compete as `families`) so that a pair whose
nearest code is a parent also tries that parent's children.

    python scripts/babytheorist/split_mixed_codes.py runs/<run> <out.pt> [--children 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from training.evaluate import latest_checkpoint, load_run  # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--pairs", type=int, default=40000)
    parser.add_argument("--min-share", type=float, default=0.01)
    parser.add_argument("--max-rate", type=float, default=0.6)
    parser.add_argument("--unused", type=float, default=0.001)
    parser.add_argument("--children", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--random", action="store_true", help="children are random unit vectors (times the parent's norm) instead of turned copies of the parent")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data, config, step = load_run(args.run, None, device)
    size = model.code.size
    named = torch.zeros(size, device=device)
    reproduced = torch.zeros(size, device=device)
    seen = 0
    loader = data.train_loader(500, args.pairs // 500 + 1, 0, args.seed, 0, 1, 0, False)
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            state, target = model.encoder(x), model.encoder(y)
            code = model.code(model.programmer(state, target), False)
            exact = model.decoder(model.executor(state, code.values)).argmax(dim=-1).eq(y).flatten(1).all(dim=1)
        index = code.indices.view(len(x), -1)[:, 0]
        named += torch.bincount(index, minlength=size).float()
        reproduced += torch.bincount(index, weights=exact.float(), minlength=size)
        seen += len(x)
        if seen >= args.pairs:
            break
    share, rate = named / seen, reproduced / named.clamp(min=1)
    mixed = [c for c in range(size) if share[c] >= args.min_share and rate[c] < args.max_rate]
    mixed.sort(key=lambda c: -float(named[c] - reproduced[c]))
    free = [c for c in range(size) if share[c] < args.unused]
    payload = torch.load(latest_checkpoint(args.run), map_location="cpu", weights_only=False)
    weights = payload["model"]["code.embedding.weight"].clone().float()
    generator = torch.Generator().manual_seed(args.seed)
    plan = {}
    for parent in mixed:
        children = []
        for _ in range(args.children):
            if not free:
                break
            child = free.pop(0)
            direction = torch.randn(weights.shape[1], generator=generator)
            vector = F.normalize(direction, dim=0) if args.random else F.normalize(weights[parent], dim=0) + args.noise * F.normalize(direction, dim=0)
            weights[child] = F.normalize(vector, dim=0) * weights[parent].norm()
            children.append(child)
        plan[parent] = children
    payload["model"]["code.embedding.weight"] = weights.to(payload["model"]["code.embedding.weight"].dtype)
    payload["split_mixed_codes"] = {"from": str(latest_checkpoint(args.run)), "plan": plan}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    report = {"pairs": seen, "step": step, "codes": {c: {"share": round(float(share[c]), 4), "reproduced": round(float(rate[c]), 3)} for c in range(size) if share[c] > 0},
              "mixed": mixed, "children": plan, "unused_left": free}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(f"{seen} pairs; mixed codes (share, reproduced): " + ", ".join(f"{c} ({float(share[c]):.3f}, {float(rate[c]):.2f})" for c in mixed))
    print("children:", plan, "| unused codes left:", len(free))


if __name__ == "__main__":
    main()
