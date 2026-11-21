#!/usr/bin/env python3
"""One table for a search over the standard parts on the one-step task.

For every run whose name starts with the given prefix: the training compute, self-explanation and transfer on the
single-primitive eval tasks at each light evaluation, and from the last full evaluation how many primitives have a code
that carries to a new input (transfer of at least --solved on that primitive's tasks) and how many codes each primitive
is spread over. The primitive's name only labels the columns.

    python scripts/babytheorist/onestep_table.py [--prefix onestep_] [--split lv1] [--out file.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHORT = {"next_color": "col", "translate_up": "up", "translate_down": "down", "translate_left": "left", "translate_right": "right", "flip_h": "flipUD", "flip_v": "flipLR", "rot90": "rot"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default="onestep_")
    parser.add_argument("--split", default="lv1")
    parser.add_argument("--solved", type=float, default=0.9)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    flops = json.loads((REPO / "runs" / "flops.json").read_text())
    report = []
    for run in sorted((REPO / "runs").glob(f"{args.prefix}*")):
        if not (run / "train.jsonl").exists():
            continue
        light = {}
        for file in sorted((run / "eval").glob(f"observe_{args.split}_step*.json")):
            d = json.loads(file.read_text())
            light[int(file.stem.split("step")[1])] = (d["explain_depth1"], d["transfer_strict_depth1"])
        state = "done" if (run / "DONE").exists() else "stopped" if (run / "STOPPED.txt").exists() else "running" if light or (run / "train.jsonl").stat().st_size else "queued"
        entry = {"run": run.name, "done": (run / "DONE").exists(), "state": state, "tflops_per_step": flops.get(run.name, {}).get("train_flops_per_step", 0) / 1e12, "light": light}
        full = sorted((run / "eval").glob(f"observe_full_{args.split}_step*.json"))
        if full:
            d = json.loads(full[-1].read_text())
            entry["full"] = {"step": d.get("step"), "explain_depth1": d["explain_depth1"], "transfer_depth1": d["transfer_strict_depth1"]}
            table = d.get("one_step_tasks", {})
            entry["primitives"] = {name: {"transfer": v["transfer_strict"] / v["tasks"], "explained": v["explained"] / v["tasks"],
                                          "codes": sum(1 for c in v["exact_code"] if c >= 0.05 * v["tasks"])} for name, v in table.items()}
            entry["solved"] = sum(1 for v in entry["primitives"].values() if v["transfer"] >= args.solved)
            entry["codes_used"] = sum(1 for c in d.get("code_usage", []) if c > 0)
        report.append(entry)
    for entry in report:
        steps = sorted(entry["light"])
        last = steps[-1] if steps else None
        head = f"{entry['run']:34s} {entry['tflops_per_step']:5.2f} TFLOPs/step  {entry['state']:7s}"
        if "full" in entry:
            per = "  ".join(f"{SHORT.get(n, n)} {v['transfer']:.2f}/{v['codes']}" for n, v in entry["primitives"].items())
            print(f"{head} depth-1 explain {entry['full']['explain_depth1']:.3f} transfer {entry['full']['transfer_depth1']:.3f}  primitives carried {entry['solved']}/{len(entry['primitives'])}  codes used {entry['codes_used']}")
            print(f"{'':36s}transfer/codes per primitive: {per}")
        elif last is not None:
            curve = "  ".join(f"{s}: {entry['light'][s][0]:.2f}/{entry['light'][s][1]:.2f}" for s in steps)
            print(f"{head} depth-1 explain/transfer by step  {curve}")
        else:
            print(head)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
