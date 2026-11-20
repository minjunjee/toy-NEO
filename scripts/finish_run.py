#!/usr/bin/env python3
"""Pick a finished run's best checkpoint and evaluate it: python scripts/finish_run.py --run runs/<run>"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from training.best import finish_best  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    finish_best(parser.parse_args().run)
