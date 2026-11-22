#!/usr/bin/env python3
"""Over training: transfer on the primitives that were left out of training and on the ones that were trained, at every
saved checkpoint of each run. The program is the model's own form (one code; one code and a count; a kind and a count).
Writes one json that scripts/babytheorist/scaling_figures.py draws.

    python scripts/babytheorist/heldout_curve.py runs/counted_circle_s42 [...] --out docs/reports/2026-11-22-scaling/heldout_curve.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from training.evaluate import load_run  # noqa: E402


@torch.no_grad()
def at(run: Path, checkpoint: Path, device: torch.device, held_out: tuple[str, ...], per_group: int) -> dict:
    model, data, config, step = load_run(run, checkpoint, device)
    counted = config["model"]["type"] in ("neo_repeat", "neo_counted")
    own = {"max_steps": model.max_length, "rollout": "hard"} if counted else {"max_steps": 1, "rollout": "carry"}
    picked = {"held out": {k: [] for k in ("support_x", "support_y", "query_x", "query_y")}, "trained": {k: [] for k in ("support_x", "support_y", "query_x", "query_y")}}
    for batch in data.eval_loader("lv1", 1000):
        for i in torch.nonzero(batch["depth"] == 1).flatten().tolist():
            name = json.loads(batch["transitions"][i])[0]["primitive"]
            group = picked["held out" if name in held_out else "trained"]
            if len(group["support_x"]) < per_group:
                for key in group:
                    group[key].append(batch[key][i])
    out = {"step": step}
    for group, tensors in picked.items():
        sx, sy, qx, qy = (torch.stack(tensors[k]).to(device) for k in ("support_x", "support_y", "query_x", "query_y"))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out[f"{group} explain"] = float(model.self_explain(sx, sy, **own)["pred"].eq(sy).flatten(1).all(dim=1).float().mean())
            out[f"{group} transfer"] = float(model.infer(sx, sy, qx, **own)["pred"].eq(qy).flatten(1).all(dim=1).float().mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--held-out", nargs="*", default=["shift_up_4", "shift_down_4", "shift_left_4", "shift_right_4"])
    parser.add_argument("--per-group", type=int, default=800)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = json.loads(args.out.read_text()) if args.out.exists() else {}
    for run in args.runs:
        done = {p["step"] for p in report.get(run.name, [])}
        for checkpoint in sorted((run / "ckpt").glob("step-*.pt")):
            if int(checkpoint.stem.split("-")[1]) in done:
                continue
            point = at(run, checkpoint, device, tuple(args.held_out), args.per_group)
            report.setdefault(run.name, []).append(point)
            report[run.name].sort(key=lambda p: p["step"])
            print(run.name, point, flush=True)
            args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
