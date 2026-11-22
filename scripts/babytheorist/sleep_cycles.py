#!/usr/bin/env python3
"""Alternate sleep and wake: split the mixed codes of a run, train, split again, train again.

Each cycle: scripts/babytheorist/split_mixed_codes.py on the previous run (codes named often that reproduce fewer than 60%
of their pairs get fresh random children; children that never won are unused again and can be handed out again), then
`--steps` steps of training that choose each pair's code by executing candidates with the code vectors fixed
(models/neo_compete.py, the family tried whenever its parent is nearest). The config of a cycle is the given template
config with the starting checkpoint and the families replaced; configs and runs are written under the output root.

    python scripts/babytheorist/sleep_cycles.py --template configs/babytheorist/onestep_m24_split_random_s42.yaml \
        --start runs/onestep_s_m24_wide_s42 --cycles 4 --name onestep_m24_sleep
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--name", required=True)
    parser.add_argument("--children", type=int, default=3)
    args = parser.parse_args()
    template = yaml.safe_load(args.template.read_text())
    previous = args.start
    folder = REPO / "runs" / "sleep_cycles"
    folder.mkdir(parents=True, exist_ok=True)
    for cycle in range(1, args.cycles + 1):
        name = f"{args.name}{cycle}_s{template['train']['seed']}"
        checkpoint = folder / f"{name}_start.pt"
        subprocess.run([sys.executable, str(REPO / "scripts" / "babytheorist" / "split_mixed_codes.py"), str(previous), str(checkpoint), "--children", str(args.children),
                        "--random", "--seed", str(cycle)], check=True, cwd=REPO)
        report = json.loads(checkpoint.with_suffix(".json").read_text())
        config = json.loads(json.dumps(template))
        config["model"]["families"] = {int(k): v for k, v in report["children"].items()}
        config["train"]["init_from"] = str(checkpoint.relative_to(REPO))
        config["train"]["steps"] = config["method"]["total_steps"] = args.steps
        path = folder / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        with open(folder / f"{name}.log", "w") as log:
            subprocess.run([sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(path), "--output-root", str(REPO / "runs"), "--wandb-mode", "disabled"],
                           check=True, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=os.environ)
        previous = REPO / "runs" / name
        print(f"cycle {cycle}: {name} done, mixed codes split: {report['mixed']}", flush=True)


if __name__ == "__main__":
    main()
