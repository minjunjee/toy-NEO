#!/usr/bin/env python3
"""Generate a dataset from a profile: python scripts/generate.py --spec configs/data/<name>.yaml"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from data.gridworld.hdf5 import write_profile  # noqa: E402
from data.gridworld.profiles import get_profile  # noqa: E402
from data.gridworld.shape_pool import load_shape_pool  # noqa: E402
from training.trainer import git_state  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    unknown = set(spec) - {"profile", "shape_pool", "output"}
    if unknown:
        raise KeyError(f"spec: unknown keys {sorted(unknown)}")
    output = Path(spec["output"])
    if (output / "manifest.json").exists():
        raise FileExistsError(f"{output} already has a manifest; datasets are immutable")
    profile = get_profile(spec["profile"])
    shape_pool = load_shape_pool(spec["shape_pool"])
    paths = write_profile(profile, shape_pool, output)
    manifest = {
        "spec": spec,
        "profile": profile.name,
        "generator_git": git_state(REPO),
        "files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in paths},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
