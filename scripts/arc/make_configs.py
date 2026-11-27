#!/usr/bin/env python3
"""Write the run configs under configs/arc/. One yaml per run.

    python scripts/arc/make_configs.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "configs" / "arc"
DATA = "data/arc"
TASKS = {16: 203, 30: 400}          # tasks of the re-ARC train file at each canvas size (scripts/arc/prepare_arc.py)


class Flow(yaml.SafeDumper):
    pass


def run(size: int, steps: int, seed: int = 0, batch: int = 64, pairs: int = 4, code_dim: int = 64, table: bool = False, loops: int = 1,
        executor: tuple[int, int] = (256, 6), programmer: tuple[int, int] = (256, 4), color_perms: int = 8, symmetries: int = 8,
        lr: float = 3.0e-4, tie_weight: float = 1.0, table_share: float = 0.5, table_lr_scale: float = 1.0, pad_weight: float = 1.0,
        tie_programmer: float | None = None, evals_every: int | None = None, final_fit: int = 100, final_votes: int = 8, workers: int = 4,
        init_from: str | None = None, ckpt_every: int | None = None, plain_share: float = 0.0) -> dict:
    """NEO on re-ARC episodes at one canvas size. `batch` counts episodes; an episode is `pairs` pairs of one task variant."""

    every = evals_every or max(steps // 10, 1)
    files = {"heldout": f"{DATA}/rearc_heldout_{size}.npz", "evaluation": f"{DATA}/arc_evaluation_{size}.npz", "training": f"{DATA}/arc_training_{size}.npz"}
    width = lambda d, layers: {"d_model": d, "num_heads": 8, "d_ff": 4 * d, "num_layers": layers, "dropout": 0.0}  # noqa: E731
    evals = [{"type": "arc_tasks", "name": "during", "every": every, "split": split, "batch_size": 256, "table": table} for split in files]
    evals += [{"type": "arc_tasks", "name": "final", "every": steps, "split": split, "batch_size": 256, "table": table, "fit_steps": final_fit,
               "fit_lr": 0.05, "fit_report": [10, 30], "votes": final_votes} for split in files]
    return {
        "data": {"type": "arc_tasks", "train": f"{DATA}/rearc_train_{size}.npz", "evals": files, "pairs": pairs, "color_perms": color_perms, "symmetries": symmetries, "seed": 0,
                 **({"plain_share": plain_share} if plain_share else {})},
        "model": {"type": "neo_arc", "encoder": {"type": "token_embed", "vocab": 11, "d_model": 128},
                  "decoder": {"type": "linear_head", "d_model": 128, "vocab": 11, "layer_norm_eps": 1.0e-6, "spatial": [size, size]},
                  "programmer": {"type": "token_pair_transformer", "state_dim": 128, "action_dim": code_dim, "grid_dim": size, "layer_norm_eps": 1.0e-6, **width(*programmer)},
                  "executor": {"type": "adaln_transformer", "state_dim": 128, "cond_dim": code_dim, "grid_dim": size, **width(*executor)},
                  "code_dim": code_dim, "tasks": TASKS[size] if table else 0, "variants": color_perms * symmetries, "table_share": table_share, "loops": loops,
                  **({} if tie_programmer is None else {"tie_programmer": tie_programmer})},
        "method": {"type": "neo_arc", "lr": lr, "weight_decay": 0.01, "warmup_ratio": 0.02, "min_lr_ratio": 0.1, "grad_clip": 1.0, "total_steps": steps,
                   "tie_weight": tie_weight, "table_lr_scale": table_lr_scale, "pad_weight": pad_weight},
        "evals": evals,
        "train": {"steps": steps, "batch_size": batch, "seed": seed, "precision": "bf16", "ckpt_every": ckpt_every or steps, "log_every": 50, "num_workers": workers,
                  "init_from": init_from, "wandb_project": "toy-neo", "wandb_mode": "disabled", "compile": "parts"},
    }


def sleep(base: dict, start: str, new: str, steps: int, lr: float = 1.0e-4, new_share: float = 0.5, every: int | None = None, fit: int = 30, votes: int = 8) -> dict:
    """A sleep of a trained run: all parts train on the demonstration pairs of new tasks (`new`, written by
    scripts/arc/sleep_data.py) mixed with replayed episodes of the run's own train file. New tasks have no name: their
    code comes from their pairs alone."""

    config = {key: dict(value) if isinstance(value, dict) else value for key, value in base.items()}
    old = base["data"]["train"]
    config["data"] = {**base["data"], "train": [new, old] if new_share < 1 else [new], "shares": [new_share, 1 - new_share] if new_share < 1 else [1.0],
                      "named": [False, True] if new_share < 1 else [False]}
    config["method"] = {**base["method"], "lr": lr, "total_steps": steps, "warmup_ratio": 0.05}
    every = every or max(steps // 8, 1)
    splits = base["data"]["evals"]
    table = bool(base["model"]["tasks"])
    config["evals"] = [{"type": "arc_tasks", "name": "during", "every": every, "split": split, "batch_size": 256, "table": table,
                        **({"fit_steps": fit, "fit_lr": 0.05} if split == "evaluation" and fit else {})} for split in splits]
    config["evals"] += [{"type": "arc_tasks", "name": "final", "every": steps, "split": split, "batch_size": 256, "table": table, "fit_steps": 100,
                         "fit_lr": 0.05, "fit_report": [10, 30], "votes": votes} for split in splits]
    config["train"] = {**base["train"], "steps": steps, "ckpt_every": steps, "init_from": start}
    return config


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configs: dict[str, dict] = {}
    # the cheap proxy: 16x16 canvas, the 203 tasks with enough examples that fit it
    proxy = dict(size=16, steps=20000)
    configs["arc16_programmer_s0"] = run(**proxy)
    configs["arc16_table_s0"] = run(**proxy, table=True)
    configs["arc16_programmer_loops3_s0"] = run(**proxy, loops=3)
    configs["arc16_programmer_code256_s0"] = run(**proxy, code_dim=256)
    configs["arc16_programmer_plain_views_s0"] = run(**proxy, color_perms=1, symmetries=1)
    configs["arc16_programmer_deep_s0"] = run(**proxy, executor=(384, 8))
    # second batch: the table only follows the programmer's codes; three times the steps; sleeps on the evaluation tasks' pairs
    configs["arc16_table_follows_s0"] = run(**proxy, table=True, tie_programmer=0.0)
    configs["arc16_programmer_60k_s0"] = run(size=16, steps=60000)
    configs["arc16_programmer_lr1e3_s0"] = run(**proxy, lr=1.0e-3)
    configs["arc16_programmer_colors512_s0"] = run(**proxy, color_perms=512)
    configs["arc16_programmer_half_plain_s0"] = run(**proxy, plain_share=0.5)
    start, new = "runs/arc16_programmer_s0/ckpt/step-020000.pt", f"{DATA}/sleep_evaluation_16.npz"
    configs["arc16_sleep_replay_s0"] = sleep(configs["arc16_programmer_s0"], start, new, steps=4000)
    configs["arc16_sleep_alone_s0"] = sleep(configs["arc16_programmer_s0"], start, new, steps=4000, new_share=1.0)
    configs["arc16_sleep_replay_lr3_s0"] = sleep(configs["arc16_programmer_s0"], start, new, steps=4000, lr=3.0e-4)
    # third batch, all with the code of 256 dims: the 30x30 run's model at proxy size, a longer code, a larger programmer, more pairs
    wide = dict(size=16, steps=20000, code_dim=256)
    configs["arc16_code256_deep_s0"] = run(**wide, executor=(384, 8))
    configs["arc16_code1024_s0"] = run(size=16, steps=20000, code_dim=1024)
    configs["arc16_code256_reader_s0"] = run(**wide, programmer=(384, 6))
    configs["arc16_code256_pairs6_s0"] = run(**wide, pairs=6)
    configs["arc16_code256_half_plain_s0"] = run(**wide, plain_share=0.5)
    # the full canvas: all 400 tasks at 30x30, the best of the first proxy batch (wider and deeper executor, code of 256 dims).
    # The heavy frozen evaluation is run apart (scripts/arc/evaluate.py); the run itself ends with a light one.
    configs["arc30_programmer_s0"] = run(size=30, steps=200000, code_dim=256, executor=(384, 8), evals_every=10000, ckpt_every=25000, final_fit=30, final_votes=1, workers=8)
    for name, config in configs.items():
        (OUT / f"{name}.yaml").write_text(yaml.dump(config, Dumper=Flow, sort_keys=False, default_flow_style=None, width=200))
    print(f"{len(configs)} configs in {OUT}")


if __name__ == "__main__":
    main()
