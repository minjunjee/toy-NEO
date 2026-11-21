#!/usr/bin/env python3
"""Train a list of configs on the given GPUs, a fixed number at a time per GPU.

The jobs file has one job per line: a config path, optionally followed by `after=<run name>[,<run name>...]` and
`resume=<checkpoint>` or `sleep_every=<steps>`. A job starts only when every run it names has finished without error; with
resume= it continues from that checkpoint; with sleep_every= it trains in segments with an offline phase between them
(scripts/babytheorist/train_with_sleep.py). The file is read again on every pass, so lines can
be appended while the runner is alive. The runner ends when every line has run and nothing is running. A run whose
directory already holds a DONE file is not started again.

    python scripts/run_queue.py --jobs runs/queue/jobs.txt --gpus GPU-<uuid> GPU-<uuid> --per-gpu 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]


def say(text: str) -> None:
    print(f"{datetime.now(ZoneInfo('Asia/Seoul')):%F %T} KST {text}", flush=True)


def read_jobs(path: Path) -> list[tuple[str, list[str], str | None]]:
    jobs = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        after = [name for p in parts[1:] if p.startswith("after=") for name in p[6:].split(",") if name]
        resume = next((p[7:] for p in parts[1:] if p.startswith("resume=")), None)
        sleep = next((p[12:] for p in parts[1:] if p.startswith("sleep_every=")), None)
        if sleep:
            extra = next((p[11:].split(",") for p in parts[1:] if p.startswith("sleep_args=")), [])
            sleep = (sleep, next((p[12:] for p in parts[1:] if p.startswith("first_sleep=")), str(2 * int(sleep))), extra)
        jobs.append((parts[0], after, (resume, sleep)))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--per-gpu", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=REPO / "runs")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    running: dict[str, tuple[subprocess.Popen, str]] = {}
    finished: dict[str, int] = {}
    while True:
        for name, (process, gpu) in list(running.items()):
            code = process.poll()
            if code is not None:
                finished[name] = code
                del running[name]
                if code == 0:
                    (args.output_root / name / "DONE").write_text("0\n")
                say(f"{'done' if code == 0 else f'FAILED ({code})'} {name}")
        waiting = 0
        for config, after, (resume, sleep) in read_jobs(args.jobs):
            name = Path(config).stem
            if name in running or name in finished:
                continue
            if (args.output_root / name / "DONE").exists():
                finished[name] = 0
                say(f"already done {name}")
                continue
            if any(finished.get(a) not in (0,) and not (args.output_root / a / "DONE").exists() for a in after):
                if any(finished.get(a, 0) != 0 for a in after):
                    finished[name] = 1
                    say(f"skipped {name}: a run it waits for failed")
                else:
                    waiting += 1
                continue
            load = {gpu: sum(1 for _, g in running.values() if g == gpu) for gpu in args.gpus}
            gpu = min(load, key=load.get)
            if load[gpu] >= args.per_gpu:
                waiting += 1
                continue
            directory = args.output_root / name
            directory.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads)}
            log = open(directory / "stdout.log", "w")
            command = [sys.executable, str(REPO / "scripts" / "train.py"), "--config", config, "--output-root", str(args.output_root), "--wandb-mode", "disabled"]
            if resume:
                command += ["--resume", resume]
            if sleep:  # wake and sleep in turns: first phase after two periods unless first_sleep= says otherwise, eval split named lv1
                command = [sys.executable, str(REPO / "scripts" / "babytheorist" / "train_with_sleep.py"), "--config", config, "--output-root", str(args.output_root),
                           "--split", "lv1", "--sleep-every", sleep[0], "--first-sleep", sleep[1], "--candidates", "3", *sleep[2]]
            running[name] = (subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO), gpu)
            say(f"start {name} on {gpu}")
        if not running and not waiting:
            say(f"all done, failed: {[n for n, c in finished.items() if c != 0]}")
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
