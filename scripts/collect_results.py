#!/usr/bin/env python3
"""Copy the result files that the write-ups cite into the repository, so the numbers stay with the text.

runs/ is not in git. This copies every JSON result in runs/queue/ to docs/results/raw/, and writes
docs/results/code_formation_by_run.json: for every one-step run of the many-primitive environments (names starting with
onestep_s_m), the orientation table of every saved evaluation (scripts/babytheorist/orientation_table.py): the four
orientation primitives that swap width and height, the other primitives, and code NMI. Run it again after new results.

    python scripts/collect_results.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "babytheorist"))

from orientation_table import rows  # noqa: E402


def main() -> None:
    raw = REPO / "docs" / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted((REPO / "runs" / "queue").glob("*.json")):
        shutil.copy2(path, raw / path.name)
        copied += 1
    table = {}
    for run in sorted((REPO / "runs").glob("onestep_s_m*")):
        if not run.is_dir() or not (run / "eval").exists():
            continue
        try:
            found = rows(run)
        except Exception as error:  # a run without the per-primitive tables
            table[run.name] = {"error": str(error)[:200]}
            continue
        if found:
            table[run.name] = {"done": (run / "DONE").exists(), "evaluations": found}
    (REPO / "docs" / "results" / "code_formation_by_run.json").write_text(json.dumps(table, indent=1) + "\n")
    print(f"copied {copied} result files; code formation tables for {len(table)} runs")


if __name__ == "__main__":
    main()
