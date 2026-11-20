"""Best checkpoint: highest mean self-explanation exact on the validation (else ID) splits, earliest on ties.

OOD splits never take part in the selection. The chosen checkpoint gets every configured eval and a summary file.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from training.evaluate import load_run
from training.trainer import run_evals


def selection_blocks(config: dict) -> list[dict]:
    """Unnamed self_explain_exact blocks on val* splits, or on id* splits when there is no validation split."""

    candidates = [b for b in config["evals"] if b["type"] == "self_explain_exact" and "name" not in b]
    validation = [b for b in candidates if b["split"].startswith("val")]
    selected = validation or [b for b in candidates if b["split"].startswith("id")]
    if not selected:
        raise ValueError("no validation or ID self_explain_exact block to select on")
    return selected


def eval_path(run: Path, block: dict, step: int) -> Path:
    return run / "eval" / f"{block.get('name', block['type'])}_{block['split']}_step{step:06d}.json"


def select_best(run: Path, config: dict) -> tuple[int, float, list[str]]:
    blocks = selection_blocks(config)
    candidates = []
    for checkpoint in sorted((run / "ckpt").glob("step-*.pt")):
        step = int(checkpoint.stem.split("-")[1])
        paths = [eval_path(run, b, step) for b in blocks]
        if all(p.exists() for p in paths):
            scores = [json.loads(p.read_text())["exact"] for p in paths]
            candidates.append((sum(scores) / len(scores), -step))
    if not candidates:
        raise RuntimeError("no checkpoint has every selection eval")
    score, negative_step = max(candidates)
    return -negative_step, score, [b["split"] for b in blocks]


@torch.no_grad()
def finish_best(run: str | Path, device: torch.device | None = None) -> dict:
    run = Path(run)
    config = json.loads((run / "resolved.json").read_text())
    best, score, splits = select_best(run, config)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data, saved_config, _ = load_run(run, run / "ckpt" / f"step-{best:06d}.pt", device)
    missing = [b for b in saved_config["evals"] if not eval_path(run, b, best).exists()]
    if missing:
        run_evals(model, data, missing, best, device, run / "eval")
    summary = {
        "selection": "mean validation/ID self-explanation exact; earliest step on ties; OOD splits not used",
        "selection_splits": splits, "best_step": best, "best_score": score,
        "best_checkpoint": str(run / "ckpt" / f"step-{best:06d}.pt"),
        "results": {p.stem: json.loads(p.read_text()) for p in sorted((run / "eval").glob(f"*step{best:06d}.json"))},
    }
    (run / "analysis").mkdir(exist_ok=True)
    (run / "analysis" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"best_step": best, "best_score": score, "selection_splits": splits}), flush=True)
    return summary
