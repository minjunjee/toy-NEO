#!/usr/bin/env python3
"""Seed means of the final observation of finished runs, grouped by config name without the seed.

    python scripts/babytheorist/summarize.py [--runs runs] [--out docs/reports/<name>/summary.json] [name prefix ...]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KEYS = ("explain", "transfer_lab", "transfer_strict", "transfer_consistent")
DEPTHS = (1, 2, 3, 4, 5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefixes", nargs="*")
    parser.add_argument("--runs", type=Path, default=REPO / "runs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    groups: dict[str, dict[int, dict]] = {}
    for directory in sorted(args.runs.glob("neo_*_s[0-9]*")):
        match = re.fullmatch(r"(.+)_s(\d+)", directory.name)
        if not match or (args.prefixes and not any(match.group(1).startswith(p) for p in args.prefixes)):
            continue
        files = sorted((directory / "eval").glob("observe_full_*_step*.json"))
        if not files or not (directory / "DONE").exists():
            continue
        rows = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        minutes = max((r.get("elapsed", 0.0) for r in rows), default=0.0) / 60
        groups.setdefault(match.group(1), {})[int(match.group(2))] = {**json.loads(files[-1].read_text()), "minutes": minutes}
    report = {}
    print(f"{'runs':46s} seeds  " + "  ".join(f"{k:>19s}" for k in KEYS) + "   strict by depth 1 2 3 4 5+        explain by depth                 minutes")
    for name, seeds in groups.items():
        mean = lambda key: sum(s[key] for s in seeds.values()) / len(seeds)  # noqa: E731
        entry = {"seeds": sorted(seeds), **{k: mean(k) for k in KEYS}, **{f"{k}_per_seed": [seeds[s][k] for s in sorted(seeds)] for k in KEYS},
                 **{f"{k}_depth{d}": mean(f"{k}_depth{d}") for k in KEYS for d in DEPTHS}, "tasks_by_depth": [next(iter(seeds.values()))[f"tasks_depth{d}"] for d in DEPTHS],
                 "minutes": mean("minutes")}
        report[name] = entry
        print(f"{name:46s} {len(seeds):3d}    " + "  ".join(f"{entry[k]:19.4f}" for k in KEYS) + "   " + " ".join(f"{entry[f'transfer_strict_depth{d}']:.3f}" for d in DEPTHS)
              + "     " + " ".join(f"{entry[f'explain_depth{d}']:.3f}" for d in DEPTHS) + f"   {entry['minutes']:6.0f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
