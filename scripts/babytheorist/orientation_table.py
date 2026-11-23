#!/usr/bin/env python3
"""Orientation primitives in the saved evaluation tables of 24-primitive runs: transfer of the seven, of the four that
swap the bbox's width and height, of the other seventeen, and the code NMI over the seven, at every saved evaluation.

    python scripts/babytheorist/orientation_table.py [--last] runs/<run> [...]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from evals.babytheorist import normalized_mutual_information as nmi  # noqa: E402

ORIENT = ["rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "anti_transpose"]
SWAP = ["rot90", "rot270", "transpose", "anti_transpose"]


def rows(run: Path):
    out = []
    for path in glob.glob(str(run / "eval" / "*_step*.json")):
        table = json.loads(Path(path).read_text()).get("one_step_tasks")
        if not table:
            continue
        step = int(re.search(r"step(\d+)", path).group(1))
        rate = {name: e["transfer_strict"] / max(e["tasks"], 1) for name, e in table.items()}
        mean = lambda names: sum(rate[n] for n in names if n in rate) / max(1, sum(n in rate for n in names))  # noqa: E731
        others = [n for n in rate if n not in ORIENT]
        out.append({"step": step, "full": "full" in Path(path).name, "orientation": mean(ORIENT), "swap": mean(SWAP), "others": mean(others),
                    "nmi_orientation": nmi(torch.tensor([table[n]["code"] for n in ORIENT if n in table])),
                    "nmi_all": nmi(torch.tensor([e["code"] for e in table.values()])),
                    "per_primitive": {n: round(rate[n], 3) for n in ORIENT if n in rate}})
    return sorted(out, key=lambda r: (r["step"], r["full"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--last", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {}
    for run in args.runs:
        table = rows(run)
        report[run.name] = table
        for r in (table[-1:] if args.last else table):
            print(f"{run.name:32s} step {r['step']:6d}{' full' if r['full'] else '     '} orientation {r['orientation']:.2f} swap four {r['swap']:.2f} others {r['others']:.2f} "
                  f"NMI orientation {r['nmi_orientation']:.2f} all {r['nmi_all']:.2f}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
