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
        init_from: str | None = None, ckpt_every: int | None = None, plain_share: float = 0.0, code_tokens: int = 1, map_dim: int = 0,
        program_length: int = 1, tie_table: float | None = None, loop_detach: bool | None = None) -> dict:
    """NEO on re-ARC episodes at one canvas size. `batch` counts episodes; an episode is `pairs` pairs of one task variant."""

    every = evals_every or max(steps // 10, 1)
    files = {"heldout": f"{DATA}/rearc_heldout_{size}.npz", "evaluation": f"{DATA}/arc_evaluation_{size}.npz", "training": f"{DATA}/arc_training_{size}.npz"}
    width = lambda d, layers: {"d_model": d, "num_heads": 8, "d_ff": 4 * d, "num_layers": layers, "dropout": 0.0}  # noqa: E731
    evals = [{"type": "arc_tasks", "name": "during", "every": every, "split": split, "batch_size": 256, "table": table} for split in files]
    reader = {"type": "token_pair_transformer", "state_dim": 128, "action_dim": code_dim * program_length, "grid_dim": size, "layer_norm_eps": 1.0e-6, **width(*programmer)}
    runner = {"type": "adaln_transformer", "state_dim": 128, "cond_dim": code_dim, "grid_dim": size, **width(*executor)}
    if code_tokens > 1:                # the code is several vectors that the executor reads as tokens
        reader = {**reader, "type": "token_pair_queries", "action_dim": code_dim // code_tokens, "num_queries": code_tokens}
        runner = {**runner, "type": "token_context_transformer", "cond_dim": code_dim // code_tokens, "num_code_tokens": code_tokens}
    if map_dim:                        # the code is the vector plus a map of one small vector per cell, read by attention
        reader = {**reader, "type": "token_pair_map", "map_dim": map_dim}
        runner = {**runner, "type": "adaln_map_transformer", "map_dim": map_dim}
    evals += [{"type": "arc_tasks", "name": "final", "every": steps, "split": split, "batch_size": 256, "table": table, "fit_steps": final_fit,
               "fit_lr": 0.05, "fit_report": [10, 30], "votes": final_votes} for split in files]
    return {
        "data": {"type": "arc_tasks", "train": f"{DATA}/rearc_train_{size}.npz", "evals": files, "pairs": pairs, "color_perms": color_perms, "symmetries": symmetries, "seed": 0,
                 **({"plain_share": plain_share} if plain_share else {})},
        "model": {"type": "neo_arc", "encoder": {"type": "token_embed", "vocab": 11, "d_model": 128},
                  "decoder": {"type": "linear_head", "d_model": 128, "vocab": 11, "layer_norm_eps": 1.0e-6, "spatial": [size, size]},
                  "programmer": reader, "executor": runner,
                  "code_dim": code_dim * program_length + size * size * map_dim, "tasks": TASKS[size] if table else 0, "variants": color_perms * symmetries, "table_share": table_share, "loops": loops,
                  **({} if tie_programmer is None else {"tie_programmer": tie_programmer}), **({"code_tokens": code_tokens} if code_tokens > 1 else {}),
                  **({"program_length": program_length} if program_length > 1 else {}), **({} if tie_table is None else {"tie_table": tie_table}),
                  **({} if loop_detach is None else {"loop_detach": loop_detach})},
        "method": {"type": "neo_arc", "lr": lr, "weight_decay": 0.01, "warmup_ratio": 0.02, "min_lr_ratio": 0.1, "grad_clip": 1.0, "total_steps": steps,
                   "tie_weight": tie_weight, "table_lr_scale": table_lr_scale, "pad_weight": pad_weight},
        "evals": evals,
        "train": {"steps": steps, "batch_size": batch, "seed": seed, "precision": "bf16", "ckpt_every": ckpt_every or steps, "log_every": 50, "num_workers": workers,
                  "init_from": init_from, "wandb_project": "toy-neo", "wandb_mode": "disabled", "compile": "parts"},
    }


def sleep(base: dict, start: str, new: str, steps: int, lr: float = 1.0e-4, new_share: float = 0.5, every: int | None = None, fit: int = 30, votes: int = 8,
          given: int = 0, table_share: float = 0.5, tie_weight: float = 0.0, table_lr_scale: float = 10.0, final_fit: int = 100) -> dict:
    """A sleep of a trained run: all parts train on the demonstration pairs of new tasks (`new`, written by
    scripts/arc/sleep_data.py) mixed with replayed episodes of the run's own train file. By default new tasks have no
    name: their code comes from their pairs alone. With `given` (the number of new tasks) every new task and view owns a
    code in the table, started from what the frozen model reads (`start` is then a checkpoint from scripts/arc/grow_table.py),
    and `table_share` of the new episodes are executed with it; the replayed tasks are then the ones without a name."""

    config = {key: dict(value) if isinstance(value, dict) else value for key, value in base.items()}
    old = base["data"]["train"]
    config["data"] = {**base["data"], "train": [new, old] if new_share < 1 else [new], "shares": [new_share, 1 - new_share] if new_share < 1 else [1.0],
                      "named": [bool(given), not given] if new_share < 1 else [bool(given)]}
    config["method"] = {**base["method"], "lr": lr, "total_steps": steps, "warmup_ratio": 0.05}
    if given:
        config["model"] = {**base["model"], "tasks": given, "table_share": table_share}
        config["method"] = {**config["method"], "tie_weight": tie_weight, "table_lr_scale": table_lr_scale}
    every = every or max(steps // 8, 1)
    splits = base["data"]["evals"]
    table = lambda split: {"table": True, "table_tasks": "same"} if given and split == "evaluation" else {"table": bool(base["model"]["tasks"]) and not given}  # noqa: E731
    config["evals"] = [{"type": "arc_tasks", "name": "during", "every": every, "split": split, "batch_size": 256, **table(split),
                        **({"fit_steps": fit, "fit_lr": 0.05} if split == "evaluation" and fit else {})} for split in splits]
    many = lambda split: bool(given) and split == "evaluation"  # noqa: E731  every view of the table is voted, with a shorter fit
    config["evals"] += [{"type": "arc_tasks", "name": "final", "every": steps, "split": split, "batch_size": 256, **table(split), "fit_steps": 30 if many(split) else final_fit,
                         "fit_lr": 0.05, "fit_report": [10] if many(split) or final_fit <= 30 else [10, 30], "votes": base["model"]["variants"] if many(split) else votes}
                        for split in splits]
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
    # the code as 16 vectors of 64 numbers that the executor reads by attention; against one vector of 1024 (arc16_code1024_s0)
    configs["arc16_tokens16_s0"] = run(size=16, steps=20000, code_dim=1024, code_tokens=16)
    configs["arc16_tokens16_deep_s0"] = run(size=16, steps=20000, code_dim=1024, code_tokens=16, executor=(384, 8))
    # the code as a 256-dim vector plus a map of 32 numbers per cell that the executor reads by attention; against arc16_programmer_code256_s0
    configs["arc16_map_s0"] = run(size=16, steps=20000, code_dim=256, map_dim=32)
    configs["arc16_map_deep_s0"] = run(size=16, steps=20000, code_dim=256, map_dim=32, executor=(384, 8))
    # a program of three codes: the executor is applied three times, once per code, each on the state the last step left; all else as arc16_programmer_code256_s0
    configs["arc16_program3_s0"] = run(size=16, steps=20000, code_dim=256, program_length=3)
    # given codes done right: every training task owns a free code in the table and the executor learns from those alone, on the tasks as given
    # (no symmetries or colour permutations, as VARC trains offline); the programmer only imitates the table. Against arc16_programmer_plain_views_s0.
    configs["arc16_given_plain_s0"] = run(size=16, steps=20000, code_dim=256, table=True, table_share=1.0, tie_table=0.0, tie_weight=0.02, table_lr_scale=10.0, color_perms=1, symmetries=1)
    configs["arc16_given_plain_deep_s0"] = run(size=16, steps=20000, code_dim=256, table=True, table_share=1.0, tie_table=0.0, tie_weight=0.02, table_lr_scale=10.0, color_perms=1, symmetries=1,
                                               executor=(384, 8))
    # the same given codes with the 64 views: every task and view owns a code
    configs["arc16_given_views_s0"] = run(size=16, steps=20000, code_dim=256, table=True, table_share=1.0, tie_table=0.0, tie_weight=0.02, table_lr_scale=10.0)
    # the executor refining its answer three times with the same code, the gradient flowing through all passes (arc16_programmer_loops3_s0 cut it between passes)
    configs["arc16_refine3_s0"] = run(size=16, steps=20000, code_dim=256, loops=3, loop_detach=False)
    # sleeps in which every evaluation task and view owns a code in the table (210 evaluation tasks fit the 16x16 canvas)
    grown = "runs/arc16_programmer_s0/ckpt/grown_evaluation.pt"
    configs["arc16_sleep_given_s0"] = sleep(configs["arc16_programmer_s0"], grown, new, steps=4000, given=210)
    configs["arc16_sleep_given_all_s0"] = sleep(configs["arc16_programmer_s0"], grown, new, steps=4000, given=210, table_share=1.0)
    # joint sleeps of the larger proxy model, the first stage before per-task sleeps (a joint sleep first lifted 33.3% to 37.6% on the small model)
    deep = "runs/arc16_programmer_deep_s0/ckpt/step-020000.pt"
    configs["arc16_deep_sleep_s0"] = sleep(configs["arc16_programmer_deep_s0"], deep, new, steps=4000, lr=3.0e-4)
    configs["arc16_deep_sleep_long_s0"] = sleep(configs["arc16_programmer_deep_s0"], deep, new, steps=12000, lr=3.0e-4)
    # the full canvas: all 400 tasks at 30x30, the best of the first proxy batch (wider and deeper executor, code of 256 dims).
    # The heavy frozen evaluation is run apart (scripts/arc/evaluate.py); the run itself ends with a light one.
    configs["arc30_programmer_s0"] = run(size=30, steps=200000, code_dim=256, executor=(384, 8), evals_every=10000, ckpt_every=25000, final_fit=30, final_votes=1, workers=8)
    # the same with 512 colour permutations instead of 8: on the proxy this lifted the frozen score on unseen tasks from 7.4% to 12.9%
    # (two voted guesses) at the same score on the training tasks. arc30_programmer_s0 was stopped at 4,500 steps for it.
    configs["arc30_colors512_s0"] = run(size=30, steps=200000, code_dim=256, executor=(384, 8), color_perms=512, evals_every=10000, ckpt_every=25000,
                                        final_fit=30, final_votes=1, workers=8)
    # the 30x30 run grown with a map: its 50,000-step checkpoint loads into the parts with a map (the map's reading starts at zero, so the
    # model starts as that checkpoint) and trains on. On the proxy the map lifted unseen examples of training tasks from 43.8% to 50.3% and
    # the per-task sleep on evaluation tasks from 29.5% to 36.2%.
    configs["arc30_map_from50k_s0"] = run(size=30, steps=120000, code_dim=256, map_dim=32, executor=(384, 8), color_perms=512, evals_every=10000, ckpt_every=20000,
                                          final_fit=30, final_votes=1, workers=8, init_from="runs/arc30_colors512_s0/ckpt/step-050000.pt")
    # the 30x30 run turned into a given-code run from its 50,000-step checkpoint: every training task and each of 64 views owns a code in the
    # table, filled with what that checkpoint's programmer reads (scripts/arc/grow_table.py); from then on the executor learns from the table
    # codes alone and the programmer imitates them. On the proxy given codes with views lifted unseen examples of training tasks from 43.8% to
    # 56.8% and the per-task sleep on evaluation tasks from 29.5% to 35.2%.
    configs["arc30_given_from50k_s0"] = run(size=30, steps=120000, code_dim=256, executor=(384, 8), table=True, table_share=1.0, tie_table=0.0, tie_weight=0.02,
                                            table_lr_scale=10.0, color_perms=8, symmetries=8, evals_every=10000, ckpt_every=20000, final_fit=30, final_votes=1, workers=8,
                                            init_from="runs/arc30_colors512_s0/ckpt/grown_given_50k.pt")
    # a joint sleep of the 30x30 run on all 400 evaluation tasks, from its checkpoints: the first stage before per-task sleeps.
    # Light evaluations inside the run; the answers come from scripts/arc/sleep_per_task.py afterwards.
    for at in (25, 50, 100, 200):
        configs[f"arc30_sleep_{at}k_s0"] = sleep(configs["arc30_colors512_s0"], f"runs/arc30_colors512_s0/ckpt/step-{at * 1000:06d}.pt", f"{DATA}/sleep_evaluation_30.npz",
                                               steps=4000, lr=3.0e-4, every=1000, fit=0, votes=1, final_fit=30)
    for name, config in configs.items():
        (OUT / f"{name}.yaml").write_text(yaml.dump(config, Dumper=Flow, sort_keys=False, default_flow_style=None, width=200))
    print(f"{len(configs)} configs in {OUT}")


if __name__ == "__main__":
    main()
