#!/usr/bin/env python3
"""Write the run configs under configs/babytheorist/. One yaml per run, as the lab code expects.

    python scripts/babytheorist/make_configs.py
"""

from __future__ import annotations

import copy
import json

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "configs" / "babytheorist"
DATA = "data/babytheorist"
GRID, COLORS = 15, 10
OBJECT_EVALS = {"objects1": f"{DATA}/single_a/lv1_eval.h5", "objects2": f"{DATA}/single_a_2obj/lv1_eval.h5", "objects3": f"{DATA}/single_a_3obj/lv1_eval.h5"}
OBJECT_TRAIN = [f"{DATA}/single_a/lv1_train.h5", f"{DATA}/single_a_2obj/lv1_train.h5", f"{DATA}/single_a_3obj/lv1_train.h5"]


class Flow(yaml.SafeDumper):
    pass


def train_block(steps: int, batch: int, seed: int, init_from: str | None, ckpt_every: int, **extra) -> dict:
    return {"steps": steps, "batch_size": batch, "seed": seed, "precision": "bf16", "ckpt_every": ckpt_every, "log_every": 50, "num_workers": 0,
            "init_from": init_from, "wandb_project": "toy-neo", "wandb_mode": "disabled", **extra}


def codec_cnn(state_dim: int, steps: int = 40000) -> dict:
    """The lab's CNN state encoder and decoder at 15x15, one latent of state_dim, trained on grids with one, two and three objects."""

    part = {"type": "cnn_vae_any", "grid_size": GRID, "num_colors": COLORS, "state_dim": state_dim, "num_state_tokens": 1, "dropout": 0.1}
    return {"data": {"type": "babytheorist_pairs", "train": OBJECT_TRAIN, "evals": OBJECT_EVALS},
            "model": {"type": "vae", "encoder": {**part, "variational": True, "deterministic": False}, "decoder": part},
            "method": {"type": "vae_pretrain", "lr": 5.0e-3, "weight_decay": 0.01, "warmup_ratio": 0.1, "min_lr_ratio": 0.005, "grad_clip": 1.0, "kl_weight": 0.0, "total_steps": steps},
            "evals": [{"type": "reconstruct_exact", "every": steps // 10, "split": split, "batch_size": 1024} for split in OBJECT_EVALS],
            "train": train_block(steps, 2048, 42, None, steps // 5)}


def codec_token(steps: int = 3000) -> dict:
    """The transformer NEO's state: one 128-d token per cell."""

    return {"data": {"type": "babytheorist_pairs", "train": OBJECT_TRAIN, "evals": OBJECT_EVALS},
            "model": {"type": "vae", "encoder": {"type": "token_embed", "vocab": COLORS, "d_model": 128},
                      "decoder": {"type": "linear_head", "d_model": 128, "vocab": COLORS, "layer_norm_eps": 1.0e-6, "spatial": [GRID, GRID]}},
            "method": {"type": "vae_pretrain", "lr": 1.0e-3, "weight_decay": 0.01, "warmup_ratio": 0.1, "min_lr_ratio": 0.1, "grad_clip": 1.0, "kl_weight": 0.0, "total_steps": steps},
            "evals": [{"type": "reconstruct_exact", "every": steps, "split": split, "batch_size": 1024} for split in OBJECT_EVALS],
            "train": train_block(steps, 512, 42, None, steps)}


def observe_blocks(split: str, steps: int, infer: dict) -> list[dict]:
    """A light look ten times during training and the full file with its tables at the end."""

    return [{"type": "babytheorist_observe", "name": "observe", "every": steps // 10, "split": split, "batch_size": 250, "max_samples": 2000, "tables": False, "infer": infer},
            {"type": "babytheorist_observe", "name": "observe_full", "every": steps, "split": split, "batch_size": 250, "infer": infer}]


def neo_transformer(train: list[str], evals: dict[str, str], seed: int, steps: int = 78200, codes: int = 16, max_length: int = 5) -> dict:
    """The lab's transformer NEO (neo_transformer_gw033_s42.yaml) with the grid size and the number of colours changed."""

    infer = {"max_steps": max_length, "rollout": "carry"}
    return {"data": {"type": "babytheorist_pairs", "train": train, "evals": evals},
            "model": {"type": "neo",
                      "encoder": {"type": "token_embed", "vocab": COLORS, "d_model": 128},
                      "decoder": {"type": "linear_head", "d_model": 128, "vocab": COLORS, "layer_norm_eps": 1.0e-6, "spatial": [GRID, GRID]},
                      "programmer": {"type": "token_query_transformer", "d_model": 128, "action_dim": 16, "num_heads": 8, "d_ff": 1024, "num_layers": 6, "grid_dim": GRID, "dropout": 0.0, "layer_norm_eps": 1.0e-6},
                      "code": {"type": "vq", "size": codes, "dim": 16, "num_tokens": 1, "commitment": 0.25, "stochastic": False, "commitment_grad": True},
                      "executor": {"type": "adaln_transformer", "state_dim": 128, "cond_dim": 16, "d_model": 128, "num_layers": 4, "num_heads": 8, "d_ff": 512, "dropout": 0.0, "grid_dim": GRID},
                      "max_length": max_length, "length_coefficient": 0.95, "state_update": "reencode_soft"},
            "method": {"type": "neo", "lr": 3.0e-4, "programmer_lr_scale": 0.25, "executor_lr_scale": 1.0, "weight_decay": 0.01, "warmup_ratio": 0.05, "min_lr_ratio": 0.1, "grad_clip": 1.0,
                       "grounding_weight": 0.1, "code_weight": 1.0, "auto_weight": 0.0, "total_steps": steps, "freeze": ["encoder", "decoder"], "code_loss_scope": "prefix"},
            "evals": [block for split in evals for block in observe_blocks(split, steps, infer)],
            "train": train_block(steps, 512, seed, "runs/codec_token/ckpt/step-003000.pt", steps // 10, ema=0.999)}


def set_path(config: dict, path: str, value) -> None:
    """config["a"]["b"]["c"] = value for path "a.b.c"."""

    *parents, last = path.split(".")
    for key in parents:
        config = config[key]
    config[last] = value


def one_step(seed: int, steps: int = 6000, codes: int = 16, changes: dict | None = None, data: tuple | None = None) -> dict:
    """The transformer NEO cut down to one program step, trained on single-primitive tasks only.

    The programmer picks one code from a pair and the executor applies it once, so a run costs a fifth of the full one.
    It asks one thing of the standard parts: can they learn every primitive as one code that works on any input? The full
    file is still evaluated, but only the single-primitive tasks (depth 1) can be solved. `changes` overrides settings by
    dotted path, for searching over the parts."""

    config = neo_transformer(*(data or SINGLE_A_LV1), seed, steps=steps, codes=codes, max_length=1)
    config["data"]["train_max_depth"] = 1
    config["train"]["compile"] = "parts"
    config["train"]["ckpt_every"] = steps  # one checkpoint per run: a search starts dozens of these and the disks are small
    # (the averaged weights lag the trained ones by about a thousand steps, which hides most of a run this short; the
    # screens from the fourth on switch the average off through `changes`)
    for path, value in (changes or {}).items():
        set_path(config, path, copy.deepcopy(value))  # a shared dict such as PAIR_PROGRAMMER must not be changed by a later path
    return config


PAIR_PROGRAMMER = {"type": "token_pair_transformer", "state_dim": 128, "d_model": 128, "action_dim": 16, "num_heads": 8, "d_ff": 512, "num_layers": 6, "grid_dim": GRID,
                   "dropout": 0.0, "layer_norm_eps": 1.0e-6}

# first screen of the standard parts on the one-step task: name -> (codes, changes)
ONE_STEP_SCREEN_1 = {
    "base": (16, {}),
    "codes8": (8, {}),
    "exec8l": (16, {"model.executor.num_layers": 8}),
    "exec256": (16, {"model.executor.d_model": 256, "model.executor.d_ff": 1024}),
    "exec256x8l": (16, {"model.executor.d_model": 256, "model.executor.d_ff": 1024, "model.executor.num_layers": 8}),
    "lr1e3": (16, {"method.lr": 1.0e-3}),
    "proglr1": (16, {"method.programmer_lr_scale": 1.0}),
    "vqsample": (16, {"model.code.stochastic": True, "model.code.tau_start": 0.3, "model.code.tau_end": 0.1, "model.code.tau_steps": 3000}),
}

# second screen: the cheaper programmer that joins the current and the target state cell by cell
ONE_STEP_SCREEN_2 = {
    "pair": (16, {"model.programmer": PAIR_PROGRAMMER}),
    "pair_ff1024": (16, {"model.programmer": {**PAIR_PROGRAMMER, "d_ff": 1024}}),
    "pair256": (16, {"model.programmer": {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}}),
    "pair256_exec256": (16, {"model.programmer": {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}, "model.executor.d_model": 256, "model.executor.d_ff": 1024}),
}


def one_step_screen_3() -> dict:
    """Third screen, all on the cheaper programmer: how the codebook is found. The cheaper programmer learns from step 300
    but stays on three codes; these vary what decides whether more codes come into use."""

    pair = {"model.programmer": PAIR_PROGRAMMER}
    restart = {"type": "vq_restart", "restart_every": 100, "restart_below": 0.1, "usage_decay": 0.99}
    sample = {"model.code.stochastic": True, "model.code.tau_start": 0.3, "model.code.tau_end": 0.1, "model.code.tau_steps": 3000}
    screen = {
        "p_codes8": (8, pair),
        "p_codes32": (32, pair),
        "p_commit005": (16, {**pair, "model.code.commitment": 0.05}),
        "p_commit1": (16, {**pair, "model.code.commitment": 1.0}),
        "p_nocommitgrad": (16, {**pair, "model.code.commitment_grad": False}),
        "p_dim4": (16, {**pair, "model.code.dim": 4, "model.programmer.action_dim": 4, "model.executor.cond_dim": 4}),
        "p_dim64": (16, {**pair, "model.code.dim": 64, "model.programmer.action_dim": 64, "model.executor.cond_dim": 64}),
        "p_sample": (16, {**pair, **sample}),
        "p_sample_hot": (16, {**pair, **sample, "model.code.tau_start": 1.0}),
        "p_proglr1": (16, {**pair, "method.programmer_lr_scale": 1.0}),
        "p_lr1e3": (16, {**pair, "method.lr": 1.0e-3}),
        "p_lr1e3_proglr1": (16, {**pair, "method.lr": 1.0e-3, "method.programmer_lr_scale": 1.0}),
        "p_restart": (16, {**pair, "model.code.type": "vq_restart", "model.code.restart_every": 100, "model.code.restart_below": 0.1, "model.code.usage_decay": 0.99}),
        "p_restart_fast": (16, {**pair, "model.code.type": "vq_restart", "model.code.restart_every": 25, "model.code.restart_below": 0.3, "model.code.usage_decay": 0.95}),
        "p_restart_codes8": (8, {**pair, "model.code.type": "vq_restart", "model.code.restart_every": 100, "model.code.restart_below": 0.1, "model.code.usage_decay": 0.99}),
        "p_restart_codes32": (32, {**pair, "model.code.type": "vq_restart", "model.code.restart_every": 100, "model.code.restart_below": 0.1, "model.code.usage_decay": 0.99}),
        "p_restart_proglr1": (16, {**pair, "method.programmer_lr_scale": 1.0, "model.code.type": "vq_restart", "model.code.restart_every": 100, "model.code.restart_below": 0.1, "model.code.usage_decay": 0.99}),
        "p_restart_sample": (16, {**pair, **sample, "model.code.type": "vq_restart", "model.code.restart_every": 100, "model.code.restart_below": 0.1, "model.code.usage_decay": 0.99}),
        "p_noema": (16, {**pair, "train.ema": 0.0}),
        "p_exec8l": (16, {**pair, "model.executor.num_layers": 8}),
        "p_exec256": (16, {**pair, "model.executor.d_model": 256, "model.executor.d_ff": 1024}),
        "p_prog3l": (16, {"model.programmer": {**PAIR_PROGRAMMER, "num_layers": 3}}),
    }
    del restart
    return screen


def one_step_screen_4() -> dict:
    """Fourth screen: combinations of what helped early in the third (restarts of unused codes, a wider or shallower
    programmer), more and smaller updates, and longer runs. The weight average is off in all of them, so the evaluations
    during training show the trained weights. name -> (codes, steps, changes)"""

    restart = {"model.code.type": "vq_restart", "model.code.restart_every": 100, "model.code.restart_below": 0.1, "model.code.usage_decay": 0.99}
    wide = {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}
    base = {"model.programmer": PAIR_PROGRAMMER, "train.ema": 0.0, **restart}
    exec_wide = {"model.executor.d_model": 256, "model.executor.d_ff": 1024}
    small_batch = {"train.batch_size": 128}
    return {
        "q_r": (16, 6000, base),
        "q_r_w": (16, 6000, {**base, "model.programmer": wide}),
        "q_r_3l": (16, 6000, {**base, "model.programmer": {**PAIR_PROGRAMMER, "num_layers": 3}}),
        "q_r_w_3l": (16, 6000, {**base, "model.programmer": {**wide, "num_layers": 3}}),
        "q_r_w_x": (16, 6000, {**base, "model.programmer": wide, **exec_wide}),
        "q_r_w_3l_x": (16, 6000, {**base, "model.programmer": {**wide, "num_layers": 3}, **exec_wide}),
        "q_r_b128": (16, 24000, {**base, **small_batch}),
        "q_r_w_b128": (16, 24000, {**base, "model.programmer": wide, **small_batch}),
        "q_r_c8": (8, 6000, base),
        "q_r_w_c8": (8, 6000, {**base, "model.programmer": wide}),
        "q_r_w_c32": (32, 6000, {**base, "model.programmer": wide}),
        "q_r_w_until3k": (16, 6000, {**base, "model.programmer": wide, "model.code.restart_until": 3000}),
        "q_r_w_commit005": (16, 6000, {**base, "model.programmer": wide, "model.code.commitment": 0.05}),
        "q_r_w_nocommitgrad": (16, 6000, {**base, "model.programmer": wide, "model.code.commitment_grad": False}),
        "q_r_w_12k": (16, 12000, {**base, "model.programmer": wide}),
        "q_r_w_3l_12k": (16, 12000, {**base, "model.programmer": {**wide, "num_layers": 3}}),
        "q_w_noema": (16, 6000, {"model.programmer": wide, "train.ema": 0.0}),
        # added after the third screen's first 3,000 steps: a small code dimension and no commitment gradient both sped up
        # the discovery of codes about as much as restarts did
        "q_r_w_dim4": (16, 6000, {**base, "model.programmer": {**wide, "action_dim": 4}, "model.code.dim": 4, "model.executor.cond_dim": 4}),
        "q_r_w_dim4_nocommitgrad": (16, 6000, {**base, "model.programmer": {**wide, "action_dim": 4}, "model.code.dim": 4, "model.executor.cond_dim": 4, "model.code.commitment_grad": False}),
        "q_r_w_3l_dim4_nocommitgrad": (16, 6000, {**base, "model.programmer": {**wide, "action_dim": 4, "num_layers": 3}, "model.code.dim": 4, "model.executor.cond_dim": 4, "model.code.commitment_grad": False}),
        "q_r_dim4_nocommitgrad": (16, 6000, {**base, "model.programmer": {**PAIR_PROGRAMMER, "action_dim": 4}, "model.code.dim": 4, "model.executor.cond_dim": 4, "model.code.commitment_grad": False}),
        "q_r_w_dim4_nocommitgrad_c32": (32, 6000, {**base, "model.programmer": {**wide, "action_dim": 4}, "model.code.dim": 4, "model.executor.cond_dim": 4, "model.code.commitment_grad": False}),
    }


def one_step_screen_5() -> dict:
    """Fifth screen: transfer instead of explanation. The fourth screen brought explanation to 99% but restarts put every
    code to use and primitives got split over several codes; the best transfer came without restarts (wide programmer, no
    weight average) and with four times as many, four times smaller updates. These vary those two things and the number
    of codes. name -> (codes, steps, changes); the weight average is off in all."""

    wide = {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}
    base = {"model.programmer": wide, "train.ema": 0.0}
    exec_wide = {"model.executor.d_model": 256, "model.executor.d_ff": 1024}
    restart_early = {"model.code.type": "vq_restart", "model.code.restart_every": 100, "model.code.restart_below": 0.1, "model.code.usage_decay": 0.99, "model.code.restart_until": 500}
    b128, b64 = {"train.batch_size": 128}, {"train.batch_size": 64}
    return {
        "t_w_b128": (16, 24000, {**base, **b128}),
        "t_w_x_b128": (16, 24000, {**base, **exec_wide, **b128}),
        "t_w_b64": (16, 48000, {**base, **b64}),
        "t_n_b128": (16, 24000, {"model.programmer": PAIR_PROGRAMMER, "train.ema": 0.0, **b128}),
        "t_w_early": (16, 6000, {**base, **restart_early}),
        "t_w_early_b128": (16, 24000, {**base, **restart_early, **b128}),
        "t_w_b128_c8": (8, 24000, {**base, **b128}),
        "t_w_b128_c10": (10, 24000, {**base, **b128}),
        "t_w_b128_c12": (12, 24000, {**base, **b128}),
        "t_w_b128_c32": (32, 24000, {**base, **b128}),
        "t_w_x": (16, 6000, {**base, **exec_wide}),
        "t_w_12k": (16, 12000, base),
        "t_w_b128_nocommitgrad": (16, 24000, {**base, **b128, "model.code.commitment_grad": False}),
        "t_w_b128_lr1e4": (16, 24000, {**base, **b128, "method.lr": 1.0e-4}),
        "t_w_b128_lr6e4": (16, 24000, {**base, **b128, "method.lr": 6.0e-4}),
        "t_w_b128_48k": (16, 48000, {**base, **b128}),
    }


def suite(name: str) -> tuple:
    return [f"data/babytheorist/{name}/lv1_train.h5"], {"lv1": f"data/babytheorist/{name}/lv1_eval.h5"}


def one_step_scale_screen() -> dict:
    """More primitives, one program step: how many primitives get a code that carries to a new input as their number
    grows. `single_a` (8) and `single_11prim` (11) are the benchmark's generator; `many_p*` are from
    scripts/babytheorist/generate_many_primitives.py (8 is the same set as single_a, to compare the two generators).
    Two recipes from the earlier screens, both with batch 128 and 24,000 steps and no weight average; the codebook has
    twice as many codes as there are primitives. name -> (data, codes, steps, changes)"""

    wide = {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}
    no_restart = {"model.programmer": wide, "train.ema": 0.0, "train.batch_size": 128}
    restart = {"model.programmer": PAIR_PROGRAMMER, "train.ema": 0.0, "train.batch_size": 128, "model.code.type": "vq_restart", "model.code.restart_every": 100,
               "model.code.restart_below": 0.1, "model.code.usage_decay": 0.99}
    screen = {}
    for tag, data, primitives in (("a8", "single_a", 8), ("m8", "many_p8", 8), ("b11", "single_11prim", 11), ("m16", "many_p16", 16), ("m24", "many_p24", 24), ("m32", "many_p32", 32)):
        screen[f"s_{tag}_wide"] = (data, 2 * primitives, 24000, no_restart)
        # the fifth screen: with 8 primitives, 10 codes carried 97%, 16 codes 90%, 32 codes 84%, 8 codes found too few
        screen[f"s_{tag}_tight"] = (data, primitives + max(2, primitives // 4), 24000, no_restart)
        screen[f"s_{tag}_restart"] = (data, 2 * primitives, 24000, restart)
    screen["s_m32_wide_c32"] = ("many_p32", 32, 24000, no_restart)
    screen["s_m32_wide_48k"] = ("many_p32", 64, 48000, no_restart)
    screen["s_m32_wide_x"] = ("many_p32", 64, 24000, {**no_restart, "model.executor.d_model": 256, "model.executor.d_ff": 1024})
    return screen


def one_step_amount_screen() -> dict:
    """Does a codebook indexed by kind and integer amount give an amount that was never trained?

    Data: 24 primitives, a move by one to six cells in each of four directions (generate_many_primitives.py --primitives
    shifts). `shifts24_without4` has no four-cell move in its train file; its eval file has all 24. One program step,
    48 codes everywhere, the recipe of the scale screen (wide programmer, batch 128, 24,000 steps, no weight average).

      flat          the lab VQ (control)
      flat_alldata  the lab VQ trained with the four-cell moves (what is reachable when nothing is left out)
      rot           codes c(g, t) = exp(t A_g) b_g, 6 kinds x amounts 1..8, nothing asked of the executor
      free_add      a free vector per (g, t), and the executor is asked that amounts add (methods/neo_compose.py)
      rot_add       both
      line_add      codes on a straight line per kind instead of a rotation, amounts add
      rot_signed_add  3 kinds x amounts -8..8 without 0, amounts add: can one kind hold a direction and its opposite

    name -> (data, changes)"""

    wide = {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}
    base = {"model.programmer": wide, "train.ema": 0.0, "train.batch_size": 128}
    positive = {"kinds": 6, "amounts": list(range(1, 9))}
    signed = {"kinds": 3, "amounts": [a for a in range(-8, 9) if a]}

    def code(family: str, layout: dict) -> dict:
        return {"model.code.type": "vq_orbit", "model.code.family": family, **{f"model.code.{k}": v for k, v in layout.items()}}

    add = {"method.type": "neo_compose", "method.compose_weight": 1.0}
    held_out = "shifts24_without4"
    return {
        "k_flat": (held_out, base),
        "k_flat_alldata": ("shifts24", base),
        "k_rot": (held_out, {**base, **code("rotation", positive)}),
        "k_free_add": (held_out, {**base, **code("free", positive), **add}),
        "k_rot_add": (held_out, {**base, **code("rotation", positive), **add}),
        "k_line_add": (held_out, {**base, **code("line", positive), **add}),
        "k_rot_signed_add": (held_out, {**base, **code("rotation", signed), **add}),
        # added while the first runs trained: with every pair of codes composed from step 0 the executor agreed with itself
        # at once (96% at step 2,000) while explaining nothing, which is what an executor that ignores the code does.
        # `known`: targets only from codes that training pairs pick. `late`: the term starts after the codes are found.
        "k_rot_add_known": (held_out, {**base, **code("rotation", positive), **add, "method.compose_known": 0.5}),
        "k_free_add_known": (held_out, {**base, **code("free", positive), **add, "method.compose_known": 0.5}),
        "k_rot_add_late": (held_out, {**base, **code("rotation", positive), **add, "method.compose_from": 3000}),
    }


def repeat_program(seed: int, steps: int = 24000, codes: int = 8, repeats: int = 6, changes: dict | None = None, data: tuple | None = None) -> dict:
    """A program that is one code and a count (models/neo_repeat.py), on single-primitive tasks, with the recipe of the
    one-step screens (wide programmer, batch 128, no weight average, compiled parts). The count goes up to `repeats`."""

    config = neo_transformer(*(data or suite("shifts24_without4")), seed, steps=steps, codes=codes, max_length=repeats)
    config["data"]["train_max_depth"] = 1
    config["model"]["type"] = "neo_repeat"
    config["model"].pop("state_update", None)
    config["model"]["programmer"] = {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}
    for block in config["evals"]:
        block["infer"] = {"max_steps": repeats, "rollout": "hard"}
    config["train"].update({"compile": "parts", "ckpt_every": steps, "ema": 0.0, "batch_size": 128})
    for path, value in (changes or {}).items():
        set_path(config, path, copy.deepcopy(value))
    return config


def repeat_screen() -> dict:
    """One code applied a counted number of times, on the shifts with the four-cell ones left out of training. The number
    of codes is what is varied: with 4 codes for 20 trained primitives a code has to be a unit that is repeated, with 24
    every trained primitive can keep a code of its own. `rot` uses the codebook tied by rotations (it found the codes
    faster in the one-step runs). name -> (codes, changes)"""

    def rot(kinds: int, amounts: int) -> dict:
        return {"model.code.type": "vq_orbit", "model.code.family": "rotation", "model.code.kinds": kinds, "model.code.amounts": list(range(1, amounts + 1))}

    return {
        "c4": (4, {}), "c6": (6, {}), "c8": (8, {}), "c12": (12, {}), "c24": (24, {}),
        "c4_rot": (4, rot(1, 4)), "c8_rot": (8, rot(2, 4)), "c24_rot": (24, rot(3, 8)),
        # added after the first runs: 6 and 12 codes reached 100% explained and transferred on all 24 primitives by step
        # 7,200, the four-cell moves included, and 6 codes then fell to 2% on the training batch by step 10,750; 8 codes
        # went up and down. Multi-step training wrecked a working model in the earlier report too, and there a lower
        # learning rate with the grounding term off slowed it. These keep a checkpoint every 2,400 steps.
        "c12_g0": (12, {"method.grounding_weight": 0.0, "train.ckpt_every": 2400}),
        "c12_lr1e4": (12, {"method.lr": 1.0e-4, "train.ckpt_every": 2400}),
        "c12_g0_lr1e4": (12, {"method.grounding_weight": 0.0, "method.lr": 1.0e-4, "train.ckpt_every": 2400}),
        "c6_g0_lr1e4": (6, {"method.grounding_weight": 0.0, "method.lr": 1.0e-4, "train.ckpt_every": 2400}),
        "c12_cut": (12, {"model.gradient_between": False, "train.ckpt_every": 2400}),
        "c6_cut": (6, {"model.gradient_between": False, "train.ckpt_every": 2400}),
    }


def counted_screen() -> dict:
    """The codebook indexed by kind and amount, with the amount executed as that many applications of the kind's unit code
    (models/neo_counted.py), on the shifts with the four-cell ones left out of training. What is varied is how the entries
    of one kind are tied: by a rotation (the Lie algebra form), on a line, or not at all. With `free` the entry for an
    amount that training never reached stays where it was initialised. 12 kinds x counts 1..6. name -> changes"""

    def entries(family: str, kinds: int = 12) -> dict:
        return {"model.type": "neo_counted", "model.code.type": "vq_orbit", "model.code.family": family, "model.code.kinds": kinds, "model.code.amounts": list(range(1, 7)),
                "model.code.size": kinds * 6, "train.ckpt_every": 2400}

    # `circle`: a rotation in one learned plane by a fixed angle per unit of amount. Added after `rot` put the six amounts
    # of a kind within 20 degrees of each other by step 7,200 (the learned rotation shrank), so the programmer could not
    # name a count by pointing at an entry.
    return {"rot": entries("rotation"), "free": entries("free"), "line": entries("line"), "rot_k6": entries("rotation", 6),
            "circle": entries("circle"), "circle_wide": {**entries("circle"), "model.code.angle": 0.6}}


GROW_STAGES = (("p8", "many_p8"), ("p16", "many_p16"), ("p24", "many_p24"), ("p32", "many_p32"), ("p52", "many_p52_without4"), ("p64", "many_p64_without4"))


def growing_chain(model: str, seed: int = 42, steps: int = 6000, stages: tuple = GROW_STAGES) -> dict:
    """An environment that grows while one model keeps learning: 8 primitives, then 16, 24, 32, 52, 64, each set
    containing the one before it. Every stage trains `steps` steps on its whole set (old and new primitives mixed, single-
    primitive pairs only) and starts from every part of the model at the end of the stage before. The codebook has its
    final size from the first stage on, so nothing about the model changes between stages; only the data grows. The
    evaluation every 1,000 steps keeps the per-primitive tables, so it shows how fast the new primitives are learned and
    whether the old ones are kept. name -> config"""

    wide = {"model.programmer": {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}, "train.ema": 0.0, "train.batch_size": 128}
    configs, previous = {}, None
    for tag, data in stages:
        if model in ("lab", "labsame"):
            config = one_step(seed, steps=steps, codes=128, changes=wide, data=suite(data))
        elif model == "repeat":
            config = repeat_program(seed, steps=steps, codes=64, repeats=6, data=suite(data))
        elif model == "repeatcut":
            config = repeat_program(seed, steps=steps, codes=64, repeats=6, data=suite(data), changes={"model.gradient_between": False})
        elif model == "expert":   # one network per code, 128 codes as the lab chain
            config = one_step(seed, steps=steps, codes=128, changes={**wide, "train.compile": "none", "model.type": "neo_expert", "model.executor": {
                "type": "per_code", "state_dim": 128, "cond_dim": 16, "size": 128, "d_model": 64, "num_layers": 2, "num_heads": 4, "d_ff": 256, "dropout": 0.0, "grid_dim": GRID}}, data=suite(data))
        elif model == "lora":   # the lab executor plus a per-code low-rank change, shared up-projection moving ten times slower
            config = one_step(seed, steps=steps, codes=128, changes={**wide, "train.compile": "none", "model.type": "neo_expert", "model.executor": {
                "type": "lora_code", "state_dim": 128, "cond_dim": 16, "size": 128, "d_model": 128, "num_layers": 4, "num_heads": 8, "d_ff": 512, "dropout": 0.0, "grid_dim": GRID,
                "rank": 8, "shared_speed": 0.1, "slow_base": True}}, data=suite(data))
        elif model == "counted":   # the fixed-angle kind-and-count codebook: 64 kinds x counts 1..6
            config = repeat_program(seed, steps=steps, codes=384, repeats=6, data=suite(data), changes={
                "model.type": "neo_counted", "model.code.type": "vq_orbit", "model.code.family": "circle", "model.code.kinds": 64, "model.code.amounts": list(range(1, 7))})
        else:
            raise ValueError(model)
        light = {**config["evals"][0], "every": 1000, "max_samples": 3000, "tables": True}
        config["evals"] = [light, {**config["evals"][-1], "every": steps}]
        config["train"]["ckpt_every"] = steps
        if previous:
            config["method"]["type"] = "neo_warm"
            config["train"]["init_from"] = f"runs/{previous}/ckpt/step-{steps:06d}.pt"
        name = f"grow_{model}_{tag}_s{seed}"
        configs[name] = config
        previous = name
    return configs


def fast_neo(train: list[str], evals: dict[str, str], seed: int, steps: int = 12000, codes: int = 16, max_length: int = 5, changes: dict | None = None) -> dict:
    """The full multi-step transformer NEO with the standard parts that the one-step search settled on.

    Against the lab config (neo_transformer): the programmer joins the current and the target state cell by cell and is
    wider and shallower (256 wide, 3 layers), the executor is 256 wide, unused codes are restarted, and the programmer's
    and the executor's forward are compiled. The rollout, the length rule, the losses and the state handed between steps
    are the lab's. `changes` overrides settings by dotted path."""

    config = neo_transformer(train, evals, seed, steps=steps, codes=codes, max_length=max_length)
    config["model"]["programmer"] = {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024, "num_layers": 3}
    config["model"]["executor"] = {**config["model"]["executor"], "d_model": 256, "d_ff": 1024}
    config["model"]["code"] = {**config["model"]["code"], "type": "vq_restart", "restart_every": 100, "restart_below": 0.1, "usage_decay": 0.99}
    config["train"]["compile"] = "parts"
    config["train"]["ckpt_every"] = steps // 4
    for path, value in (changes or {}).items():
        set_path(config, path, copy.deepcopy(value))
    return config


def fast_neo_screen_1() -> dict:
    """First multi-step runs with the searched parts on the benchmark's level 1. name -> (codes, changes)"""

    return {
        "c16": (16, {}),
        "c8": (8, {}),
        "c32": (32, {}),
        "c16_norestart": (16, {"model.code.restart_every": 0}),
        "c16_dim4_nocommitgrad": (16, {"model.code.dim": 4, "model.programmer.action_dim": 4, "model.executor.cond_dim": 4, "model.code.commitment_grad": False}),
        "c16_until6k": (16, {"model.code.restart_until": 6000}),
    }


def neo_mlp(train: list[str], evals: dict[str, str], seed: int, state_dim: int, steps: int = 39100, codes: int = 12, max_length: int = 4, codec_steps: int = 40000, hidden: int = 128) -> dict:
    """The lab's official NEO (neo_gw033_k4_cb6_s42.yaml) with the 15x15 CNN state encoder and decoder."""

    part = {"type": "cnn_vae_any", "grid_size": GRID, "num_colors": COLORS, "state_dim": state_dim, "num_state_tokens": 1, "dropout": 0.0}
    mlp = {"type": "film_mlp", "state_dim": state_dim, "action_dim": 16, "num_state_tokens": 1, "num_action_tokens": 1, "hidden": hidden, "dropout": 0.0}
    infer = {"max_steps": max_length}
    return {"data": {"type": "babytheorist_pairs", "train": train, "evals": evals},
            "model": {"type": "neo", "encoder": {**part, "variational": True, "deterministic": True}, "decoder": part,
                      "programmer": {**mlp, "max_length": max_length},
                      "code": {"type": "vq", "size": codes, "dim": 16, "num_tokens": 1, "commitment": 0.25, "stochastic": True, "tau_start": 0.3, "tau_end": 0.1, "tau_steps": steps // 2},
                      "executor": {**mlp, "residual": True},
                      "max_length": max_length, "length_coefficient": 0.95, "state_update": "carry_latent"},
            "method": {"type": "neo", "lr": 5.0e-4, "programmer_lr_scale": 0.25, "executor_lr_scale": 1.0, "weight_decay": 0.01, "warmup_ratio": 0.10, "min_lr_ratio": 0.1, "grad_clip": 1.0,
                       "grounding_weight": 0.1, "code_weight": 1.0, "auto_weight": 0.0, "total_steps": steps, "freeze": ["encoder", "decoder"]},
            "evals": [block for split in evals for block in observe_blocks(split, steps, infer)],
            "train": train_block(steps, 256, seed, f"runs/codec_cnn_d{state_dim}/ckpt/step-{codec_steps:06d}.pt", steps // 10)}


SINGLE_A_LV1 = ([f"{DATA}/single_a/lv1_train.h5"], {"lv1": f"{DATA}/single_a/lv1_eval.h5"})
# one object and the four translations only: the lab GridWorld's kind of task on the benchmark's grids
TRANSLATE_LV1 = ([f"{DATA}/single_translate/lv1_train.h5"], {"lv1": f"{DATA}/single_translate/lv1_eval.h5"})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configs = {f"codec_cnn_d{d}": codec_cnn(d) for d in (32, 64, 128, 256, 512)}
    configs["codec_token"] = codec_token()
    # the 12,000-step run was still improving when its schedule ended; three times as long, one seed
    configs["neo_transformer_36k_single_a_lv1_s42"] = neo_transformer(*SINGLE_A_LV1, 42, steps=36000)
    # the long run split primitives across its 16 codes; the same run with as many codes as there are primitives, and with ten
    configs["neo_transformer_36k_codes8_single_a_lv1_s42"] = neo_transformer(*SINGLE_A_LV1, 42, steps=36000, codes=8)
    configs["neo_transformer_36k_codes10_single_a_lv1_s42"] = neo_transformer(*SINGLE_A_LV1, 42, steps=36000, codes=10)
    # other seeds of the eight-code run, each once as it is and once trained in segments with an offline phase between them
    # (same config under another name; the phases come from scripts/babytheorist/train_with_sleep.py)
    # a codebook that grows by need: eight codes at the start, one more switched on per offline phase while training pairs
    # stay unexplained, sixteen at most (trained with scripts/babytheorist/train_with_sleep.py)
    for seed in (42, 43, 44):
        grow = neo_transformer(*SINGLE_A_LV1, seed, steps=36000, codes=16)
        grow["model"]["code"] = {**grow["model"]["code"], "type": "vq_growable", "initial": 8}
        configs[f"neo_transformer_36k_grow8to16_single_a_lv1_s{seed}"] = grow
        configs[f"neo_transformer_36k_grow8to9_single_a_lv1_s{seed}"] = grow  # the same model; the offline phase is told --max-active 9
    # what a program step may hand to the next one. The lab config hands over the decoded cell probabilities (reencode_soft),
    # and trained models pass information there that is not visible on the grid. Two ways to close that path with eight
    # codes: the lab's reencode_hard (no gradient between steps) and the straight-through model neo_visible_state
    for seed in (42, 43):
        hard = neo_transformer(*SINGLE_A_LV1, seed, steps=36000, codes=8)
        hard["model"]["state_update"] = "reencode_hard"
        configs[f"neo_transformer_36k_codes8_hard_single_a_lv1_s{seed}"] = hard
        visible = neo_transformer(*SINGLE_A_LV1, seed, steps=36000, codes=8)
        visible["model"] = {**{k: v for k, v in visible["model"].items() if k != "state_update"}, "type": "neo_visible_state"}
        configs[f"neo_transformer_36k_codes8_visible_single_a_lv1_s{seed}"] = visible
    for seed in (43, 44):
        configs[f"neo_transformer_36k_codes8_single_a_lv1_s{seed}"] = neo_transformer(*SINGLE_A_LV1, seed, steps=36000, codes=8)
        configs[f"neo_transformer_36k_codes8_sleeps_single_a_lv1_s{seed}"] = neo_transformer(*SINGLE_A_LV1, seed, steps=36000, codes=8)
    # more seeds of the same run: the first two differed threefold, so the spread over seeds comes before any comparison
    # the search over the standard parts on the one-step task (the base twice, to see the spread over seeds)
    for name, (codes, changes) in ONE_STEP_SCREEN_1.items():
        for seed in ((42, 43) if name == "base" else (42,)):
            configs[f"onestep_{name}_s{seed}"] = one_step(seed, codes=codes, changes=changes)
    for name, (codes, changes) in one_step_screen_3().items():
        configs[f"onestep_{name}_s42"] = one_step(42, codes=codes, changes=changes)
    for name, (codes, steps, changes) in one_step_screen_5().items():
        for seed in ((42, 43, 44) if name == "t_w_b128" else (42,)):
            configs[f"onestep_{name}_s{seed}"] = one_step(seed, steps=steps, codes=codes, changes=changes)
    # the full model with what the fifth screen is testing: no restarts, no weight average, smaller and more updates
    for name, changes in {"c16_norestart_b128": {"model.code.restart_every": 0, "train.batch_size": 128, "train.ema": 0.0},
                          "c16_b128": {"train.batch_size": 128, "train.ema": 0.0},
                          "c16_norestart_noema": {"model.code.restart_every": 0, "train.ema": 0.0}}.items():
        steps = 48000 if "b128" in name else 12000
        configs[f"fastneo_{steps // 1000}k_{name}_single_a_lv1_s42"] = fast_neo(*SINGLE_A_LV1, 42, steps=steps, changes=changes)
    for name, (data, codes, steps, changes) in one_step_scale_screen().items():
        configs[f"onestep_{name}_s42"] = one_step(42, steps=steps, codes=codes, changes=changes, data=suite(data))
    for name, (codes, changes) in repeat_screen().items():
        for seed in (42, 43, 44) if name in ("c4", "c8", "c8_rot") else (42,):
            configs[f"repeat_{name}_s{seed}"] = repeat_program(seed, codes=codes, changes=changes)
    for name, changes in counted_screen().items():
        for seed in (42, 43, 44) if name in ("rot", "free", "circle") else (42,):
            configs[f"counted_{name}_s{seed}"] = repeat_program(seed, codes=72, changes=changes)
    # the size ladder: nested primitive sets of 8, 16, 24, 32, 52 and 64 (the last two without the four-cell moves in the
    # train file), single-primitive training pairs only, the same budget at every size. What is read is how each model's
    # numbers move from one size to the next, on the primitives it trained on, on programs of several primitives (never
    # trained) and on the primitives left out.
    #   lab       the lab NEO, one program step, twice as many codes as primitives (8..32 are the scale screen's runs)
    #   repeat    one code applied a counted number of times, as many codes as primitives, count up to 6
    #   counted   the kind-and-amount codebook tied by rotations, the amount executed as that many applications
    for tag, data, primitives in (("p8", "many_p8", 8), ("p16", "many_p16", 16), ("p24", "many_p24", 24), ("p32", "many_p32", 32), ("p52", "many_p52_without4", 52),
                                  ("p64", "many_p64_without4", 64)):
        wide = {"model.programmer": {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}, "train.ema": 0.0, "train.batch_size": 128}
        if primitives > 32:
            configs[f"ladder_lab_{tag}_s42"] = one_step(42, steps=24000, codes=2 * primitives, changes=wide, data=suite(data))
        configs[f"ladder_repeat_{tag}_s42"] = repeat_program(42, codes=primitives, repeats=6, data=suite(data))
        # the same with no gradient between applications: on the small shift family it reached the answer sooner and kept it
        configs[f"ladder_repeatcut_{tag}_s42"] = repeat_program(42, codes=primitives, repeats=6, data=suite(data), changes={"model.gradient_between": False})
        configs[f"ladder_counted_{tag}_s42"] = repeat_program(42, codes=6 * primitives, repeats=6, data=suite(data), changes={
            "model.type": "neo_counted", "model.code.type": "vq_orbit", "model.code.family": "rotation", "model.code.kinds": primitives, "model.code.amounts": list(range(1, 7))})
    for model in ("lab", "repeat", "repeatcut", "counted", "expert", "lora"):
        configs.update(growing_chain(model))
    # control: the same stages, the same restart of the learning-rate schedule at every stage, but the environment stays
    # at 8 primitives; the dips right after each growth are read against the dips of this chain
    configs.update(growing_chain("labsame", stages=tuple((tag, "many_p8") for tag, _ in GROW_STAGES)))
    # the counted program on the mixed primitive sets of the scale screen (as many codes as primitives, count up to 4): there
    # a move by the object's own size or to the border has a count that differs from input to input
    for tag, data, primitives in (("m16", "many_p16", 16), ("m24", "many_p24", 24)):
        configs[f"repeat_{tag}_c{primitives}_s42"] = repeat_program(42, codes=primitives, repeats=4, data=suite(data))
    # one network per code (components/executors_expert.py), asked by the user on 2026-11-22: does an executor whose codes
    # cannot touch each other's weights keep the seven orientation primitives apart, and keep old primitives as the
    # environment grows? The one-step recipe of the scale screen on 24 primitives, 48 codes; compiled parts are off
    # because the executor picks a network per pair.
    wide = {"model.programmer": {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}, "train.ema": 0.0, "train.batch_size": 128, "train.compile": "none", "model.type": "neo_expert"}

    def per_code(codes: int, d_model: int = 64, layers: int = 2, d_ff: int = 256, shared: int = 0) -> dict:
        return {"model.executor": {"type": "per_code", "state_dim": 128, "cond_dim": 16, "size": codes, "d_model": d_model, "num_layers": layers, "num_heads": 4,
                                   "d_ff": d_ff, "dropout": 0.0, "grid_dim": GRID, "shared_layers": shared}}
    for name, changes in (("expert_m24_d64", per_code(48)), ("expert_m24_d128", per_code(48, 128, 2, 512)), ("expert_m24_shared", per_code(48, 64, 1, 256, shared=2))):
        configs[f"onestep_{name}_s42"] = one_step(42, steps=24000, codes=48, changes={**wide, **changes}, data=suite("many_p24"))

    # where to split shared and per-code layers (asked by the user on 2026-11-22): four layers in all, the first k shared
    # (128 wide, as the lab executor), the other 4 - k one per code (64 wide); k = 4 is the lab executor
    # (onestep_s_m24_wide_s42). `sandwich`: one shared, two per code, one shared.
    def split(codes: int, shared: int, after: int = 0) -> dict:
        own = 4 - shared - after
        return {"model.executor": {"type": "per_code", "state_dim": 128, "cond_dim": 16, "size": codes, "d_model": 64, "num_layers": own, "num_heads": 4, "d_ff": 256,
                                   "dropout": 0.0, "grid_dim": GRID, "shared_layers": shared, "shared_d_model": 128, "shared_d_ff": 512, "shared_after_layers": after}}
    for name, changes in (("split0_m24", split(48, 0)), ("split1_m24", split(48, 1)), ("split2_m24", split(48, 2)), ("split3_m24", split(48, 3)),
                          ("sandwich_m24", split(48, 1, 1))):
        configs[f"onestep_{name}_s42"] = one_step(42, steps=24000, codes=48, changes={**wide, **changes}, data=suite("many_p24"))
    # share first, split later: one network per code learns too little too late when it starts from nothing (the sweep
    # above). Here every code's network starts as a copy of the trained shared executor of onestep_s_m24_wide_s42
    # (scripts/babytheorist/split_executor_checkpoint.py), then 12,000 more steps. Control: the same 12,000 steps with the
    # shared executor kept shared. Both at learning rate 1e-4, from the same checkpoint.
    lab_end = "runs/onestep_s_m24_wide_s42/ckpt/step-024000.pt"
    more = {**wide, "method.type": "neo_warm", "method.lr": 1.0e-4, "train.ckpt_every": 12000}
    configs["onestep_m24_more_shared_s42"] = one_step(42, steps=12000, codes=48, changes={**more, "model.type": "neo", "train.init_from": lab_end, "train.compile": "parts"},
                                                       data=suite("many_p24"))
    configs["onestep_m24_more_split_s42"] = one_step(42, steps=12000, codes=48, changes={**more, "train.init_from": "runs/split_init/onestep_s_m24_wide_s42_split48.pt",
        "model.executor": {"type": "per_code", "state_dim": 128, "cond_dim": 16, "size": 48, "d_model": 128, "num_layers": 4, "num_heads": 8, "d_ff": 512, "dropout": 0.0,
                           "grid_dim": GRID}}, data=suite("many_p24"))
    # the code of a training pair chosen by executing candidates (models/neo_compete.py): the probe showed the programmer
    # sending four orientation primitives to the same codes. From the lab run's end (control: onestep_m24_more_shared_s42)
    # with 4 and 8 candidates, and from the start (control: onestep_s_m24_wide_s42). The executor is the lab's, shared.
    compete = {**wide, "model.type": "neo_compete", "train.compile": "parts"}
    configs["onestep_m24_more_compete4_s42"] = one_step(42, steps=12000, codes=48, changes={**more, **compete, "train.init_from": lab_end, "model.candidates": 4},
                                                         data=suite("many_p24"))
    configs["onestep_m24_more_compete8_s42"] = one_step(42, steps=12000, codes=48, changes={**more, **compete, "train.init_from": lab_end, "model.candidates": 8},
                                                         data=suite("many_p24"))
    configs["onestep_compete4_m24_s42"] = one_step(42, steps=24000, codes=48, changes={**compete, "model.candidates": 4}, data=suite("many_p24"))
    # diagnostic with the hidden primitive as the code (models/neo_oracle.py): can the shared executor do the seven
    # orientation primitives at all when each has a code of its own? Codebook of 24, one per primitive.
    oracle = one_step(42, steps=24000, codes=24, changes={**wide, "train.compile": "parts", "model.type": "neo_oracle"}, data=suite("many_p24"))
    oracle["data"]["type"] = "babytheorist_pairs_labeled"
    configs["onestep_s_m24_oracle_s42"] = oracle
    # a sleep step that splits mixed codes (scripts/babytheorist/split_mixed_codes.py, no labels): codes that are named
    # often but reproduce fewer than 60% of their pairs get three fresh codes next to them; then the code of each pair is
    # chosen by executing the 6 nearest candidates while the code vectors stay fixed. Control: the same training from the
    # lab run's end without the children.
    frozen = {**more, **compete, "model.candidates": 6, "method.freeze": ["encoder", "decoder", "code"]}
    configs["onestep_m24_split_children_s42"] = one_step(42, steps=12000, codes=48, changes={**frozen, "train.init_from": "runs/split_init/onestep_s_m24_wide_s42_children3.pt"},
                                                          data=suite("many_p24"))
    configs["onestep_m24_compete6_frozen_s42"] = one_step(42, steps=12000, codes=48, changes={**frozen, "train.init_from": lab_end}, data=suite("many_p24"))
    # the children next to the parent inherited its mixture. Fresh random children, tried whenever their parent is the
    # nearest code (the family comes from the split report, written by split_mixed_codes.py --random)
    report = Path("runs/split_init/onestep_s_m24_wide_s42_children3_random.json")
    if report.exists():
        families = {int(k): v for k, v in json.loads(report.read_text())["children"].items()}
        configs["onestep_m24_split_random_s42"] = one_step(42, steps=12000, codes=48, changes={**frozen, "model.candidates": 4, "model.families": families,
            "train.init_from": "runs/split_init/onestep_s_m24_wide_s42_children3_random.pt"}, data=suite("many_p24"))
    # does the orientation family come with much longer training? (asked by the user on 2026-11-23: pretraining may take as
    # long as it takes.) 24k and 48k steps did not move it; 200k steps here, with the lab executor and with one twice as
    # wide (which moved it at 32 primitives). Per-primitive tables every 10k steps.
    for name, changes in (("long200k", {}), ("long200k_x", {"model.executor.d_model": 256, "model.executor.d_ff": 1024})):
        config = one_step(42, steps=200000, codes=48, changes={**wide, "model.type": "neo", "train.compile": "parts", "train.ckpt_every": 50000, **changes},
                          data=suite("many_p24"))
        config["evals"][0].update({"every": 10000, "max_samples": 3000, "tables": True})
        config["evals"][-1]["every"] = 200000
        configs[f"onestep_s_m24_{name}_s42"] = config
    # soft EM over the codes with an annealed temperature (models/neo_em.py) and turns between the programmer and the
    # executor (methods/neo_em.py, neo_alternate), both from the 24-primitive lab run's end, 12,000 steps, learning rate
    # 1e-4; control: onestep_m24_more_shared_s42. Agreed with the user on 2026-11-23.
    em = {**wide, "train.compile": "parts", "model.type": "neo_em", "method.type": "neo_em", "method.lr": 1.0e-4, "train.init_from": lab_end, "train.ckpt_every": 12000,
          "model.candidates": 0, "model.train_top": 4, "model.temperature_start": 10.0, "model.temperature_end": 0.1, "model.anneal_steps": 10000}
    configs["onestep_m24_em_s42"] = one_step(42, steps=12000, codes=48, changes=em, data=suite("many_p24"))
    configs["onestep_m24_em_hot_s42"] = one_step(42, steps=12000, codes=48, changes={**em, "model.temperature_start": 100.0, "model.temperature_end": 1.0}, data=suite("many_p24"))
    configs["onestep_m24_alternate_s42"] = one_step(42, steps=12000, codes=48, changes={**more, "model.type": "neo", "train.compile": "parts", "train.init_from": lab_end,
                                                                                       "method.type": "neo_alternate", "method.period": 1000}, data=suite("many_p24"))
    # soft EM and hard EM from the start (agreed with the user on 2026-11-23). The soft EM above began from a run whose
    # orientation codes had already merged, where every code is about equally wrong on the merged pairs. Hard EM: a
    # temperature near zero and the executor trained on the first code only; a code nothing chooses gets a bonus that
    # grows by 0.05 nats a step up to 20 and returns to zero once it is used (models/neo_em.py). Soft EM: the annealing
    # above over 20,000 steps, no rescue. Three seeds each, the lab's 24,000-step recipe otherwise; the lab runs of the same
    # recipe for seeds 43 and 44 complete the control (seed 42 is onestep_s_m24_wide_s42). The light evaluation keeps the
    # per-primitive tables, to follow the orientation primitives and the code NMI during training.
    plain = {"model.programmer": {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}, "train.ema": 0.0, "train.batch_size": 128}
    scratch = {**plain, "model.type": "neo_em", "method.type": "neo_em", "model.candidates": 0}
    hard_em = {**scratch, "model.train_top": 1, "model.temperature_start": 0.01, "model.temperature_end": 0.01, "model.anneal_steps": 1,
               "model.dead_share": 0.1, "model.usage_decay": 0.99, "model.bonus_step": 0.05, "model.bonus_cap": 20.0}
    soft_em = {**scratch, "model.train_top": 4, "model.temperature_start": 10.0, "model.temperature_end": 0.1, "model.anneal_steps": 20000}
    for seed in (42, 43, 44):
        for name, changes in (("hardem", hard_em), ("softem", soft_em)):
            config = one_step(seed, steps=24000, codes=48, changes=changes, data=suite("many_p24"))
            config["evals"][0]["tables"] = True
            configs[f"onestep_s_m24_{name}_s{seed}"] = config
        if seed != 42:
            configs[f"onestep_s_m24_wide_s{seed}"] = one_step(seed, steps=24000, codes=48, changes=plain, data=suite("many_p24"))
    # train tasks of two pairs that share a program (generate_many_primitives.py --train-pairs 2, 40,000 single-primitive
    # tasks, 80,000 pairs; the single-pair runs above keep about 75,000 single-primitive pairs). Hard EM from the start made
    # codes that reproduce their support pair and fail the query (98% to 100% against 69% on the seventeen non-orientation
    # primitives): nothing asks a code to do the same on another input. Here a code is chosen, and trained, for what it does
    # on both pairs of a task (models/neo_em.py); the lab model trains the code named from the first pair on both
    # (models/neo_episode.py). The grouping of pairs is used, never the program. Same eval file as above.
    for seed in (42, 43, 44):
        for name, changes in (("lab_pairs2", {**plain, "model.type": "neo_episode"}), ("hardem_pairs2", hard_em), ("softem_pairs2", soft_em)):
            if name == "softem_pairs2" and seed != 42:
                continue
            config = one_step(seed, steps=24000, codes=48, changes=changes, data=suite("many_p24"))
            config["data"]["type"] = "babytheorist_episodes"
            config["data"]["train"] = [f"{DATA}/many_p24_pairs2/lv1_train.h5"]
            config["evals"][0]["tables"] = True
            configs[f"onestep_s_m24_{name}_s{seed}"] = config
    # the same at 32 primitives (64 codes), seed 42: does the split hold as the environment grows? 53,000 two-pair tasks,
    # in proportion to the 40,000 at 24 primitives. Single-pair reference: onestep_s_m32_wide_s42.
    for name, changes in (("lab_pairs2", {**plain, "model.type": "neo_episode"}), ("hardem_pairs2", hard_em)):
        config = one_step(42, steps=24000, codes=64, changes=changes, data=suite("many_p32"))
        config["data"]["type"] = "babytheorist_episodes"
        config["data"]["train"] = [f"{DATA}/many_p32_pairs2/lv1_train.h5"]
        config["evals"][0]["tables"] = True
        configs[f"onestep_s_m32_{name}_s42"] = config
    # is it the programmer we chose? The searched programmer joins the state and the target cell by cell before attention;
    # the lab programmer keeps them as two tokens per cell. The same 24-primitive run with the lab programmer, and with the
    # searched programmer twice as deep.
    configs["onestep_s_m24_labprog_s42"] = one_step(42, steps=24000, codes=48, changes={"train.ema": 0.0, "train.batch_size": 128}, data=suite("many_p24"))
    configs["onestep_s_m24_deepprog_s42"] = one_step(42, steps=24000, codes=48, changes={**wide, "train.compile": "parts", "model.type": "neo",
        "model.programmer": {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024, "num_layers": 12}}, data=suite("many_p24"))
    # the lab executor plus a low-rank change per code with a shared up-projection (components/executors_lora.py), asked
    # by the user on 2026-11-22: B shared, A per code. `slow`: B and the base weights move ten times slower.
    def lora(codes: int, rank: int = 8, speed: float = 1.0) -> dict:
        return {"model.executor": {"type": "lora_code", "state_dim": 128, "cond_dim": 16, "size": codes, "d_model": 128, "num_layers": 4, "num_heads": 8, "d_ff": 512,
                                   "dropout": 0.0, "grid_dim": GRID, "rank": rank, "shared_speed": speed, "slow_base": speed != 1.0}}
    for name, changes in (("lora_m24", lora(48)), ("lora_m24_slow", lora(48, speed=0.1))):
        configs[f"onestep_{name}_s42"] = one_step(42, steps=24000, codes=48, changes={**wide, **changes}, data=suite("many_p24"))
    # only the generators of the orientation group as atoms: rot180, rot270, transpose and anti_transpose are left out of
    # the train file of the 24-primitive set; every one of them is an exact composition of rot90 and a flip. Asked: with
    # 7 members of the group in training every orientation primitive fell to 12-24% transfer; are 3 members learnable,
    # and do the left-out ones come out of the stepwise rollout?
    wide = {"model.programmer": {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}, "train.ema": 0.0, "train.batch_size": 128}
    configs["onestep_s_m24gen_wide_s42"] = one_step(42, steps=24000, codes=40, changes=wide, data=suite("many_p24_generators"))
    for name, (data, changes) in one_step_amount_screen().items():
        for seed in (42, 43, 44) if name in ("k_flat", "k_rot", "k_free_add", "k_rot_add") else (42,):
            configs[f"onestep_{name}_s{seed}"] = one_step(seed, steps=24000, codes=48, changes=changes, data=suite(data))
    # the fast parts reach 76% self-explanation in 6,000 steps, but through two-code programs that only work when a step
    # hands the next one the cell probabilities (2% with grid-only handover). The same model trained with grid-only
    # handover: the lab's reencode_hard (no gradient between steps) and the straight-through model neo_visible_state
    for name, changes in {"hard_c16": {"model.state_update": "reencode_hard"},
                          "hard_c16_norestart": {"model.state_update": "reencode_hard", "model.code.restart_every": 0},
                          "visible_c16": {"model.type": "neo_visible_state", "model.state_update": None},
                          "visible_c16_norestart": {"model.type": "neo_visible_state", "model.state_update": None, "model.code.restart_every": 0}}.items():
        config = fast_neo(*SINGLE_A_LV1, 42, steps=12000, changes=changes)
        if config["model"].get("state_update", "") is None:
            del config["model"]["state_update"]
        configs[f"fastneo_12k_{name}_single_a_lv1_s42"] = config
    # grid-only handover learns with the fast parts (76% self-explanation and 33% transfer at 7,200 steps, both measured
    # with grid-only handover). The one-step search says a codebook with little slack and smaller, more frequent updates
    # carry over best; the same here
    hard = {"model.state_update": "reencode_hard", "model.code.restart_every": 0}
    for name, codes, steps, changes in (("hard_c10_norestart", 10, 12000, hard), ("hard_c12_norestart", 12, 12000, hard),
                                        ("hard_c10_norestart_b128", 10, 48000, {**hard, "train.batch_size": 128}),
                                        ("hard_c16_norestart_b128", 16, 48000, {**hard, "train.batch_size": 128}),
                                        ("hard_c12_norestart_b128", 12, 48000, {**hard, "train.batch_size": 128})):
        configs[f"fastneo_{steps // 1000}k_{name}_single_a_lv1_s42"] = fast_neo(*SINGLE_A_LV1, 42, steps=steps, codes=codes, changes=changes)
    # a model trained with one program step on single-primitive pairs composes at inference (83% transfer on level 1 with
    # five steps and grid-only handover, against 40% for every model trained with several steps). Picking single-primitive
    # pairs reads the hidden depth, which the benchmark does not allow. The same one-step training on ALL pairs, where the
    # pairs that need more than one primitive cannot be reproduced and only add loss:
    wide_all = {"model.programmer": {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}, "train.ema": 0.0}
    for name, codes, steps, changes in (("u_w_c16", 16, 12000, wide_all), ("u_w_b128_c10", 10, 24000, {**wide_all, "train.batch_size": 128}),
                                        ("u_w_b128_c16", 16, 24000, {**wide_all, "train.batch_size": 128}), ("u_w_b128_c32", 32, 24000, {**wide_all, "train.batch_size": 128}),
                                        ("u_w_c16_36k", 16, 36000, wide_all)):
        config = one_step(42, steps=steps, codes=codes, changes=changes)
        del config["data"]["train_max_depth"]
        configs[f"onestep_{name}_s42"] = config
    # more seeds of the one-step run whose model composes at inference (83% transfer on level 1)
    wide_512 = {"model.programmer": {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}, "train.ema": 0.0}
    for seed in (43, 44):
        configs[f"onestep_t_w_12k_s{seed}"] = one_step(seed, steps=12000, codes=16, changes=wide_512)
    # the programmer learns to plan toward far targets from programs it dreams with its own frozen executor
    # (methods/neo_dream.py), started 2026-11-23 in the autonomous session: written in each primitive's best code, the
    # hidden program reaches 92% to 97% of eight-step queries with the executor of seeds 42 and 45, the programmer's own
    # choices 11% to 42%. From the end of each one-step run, 6,000 steps, the input grids of all training pairs (no depth
    # filter, no label); only the programmer learns. The evaluation rolls out up to five steps, only the grid handed over.
    for seed in (42, 43, 44, 45, 46, 47):
        config = one_step(seed, steps=6000, codes=16, changes={**wide_512, "method.type": "neo_dream", "method.freeze": ["encoder", "decoder", "executor", "code"],
                                                                "train.init_from": f"runs/onestep_t_w_12k_s{seed}/ckpt/step-012000.pt",
                                                                "train.batch_size": 128})  # a pair gives four to five dreamed positions
        del config["data"]["train_max_depth"]
        for block in config["evals"]:
            block["infer"] = {"max_steps": 5, "rollout": "hard"}
        configs[f"dream_onestep_s{seed}"] = config
    # does training on several steps damage codes that start out clean? The multi-step model with grid-only handover,
    # started from that one-step model (all parts), with everything trained and with only the programmer trained
    start = "runs/onestep_t_w_12k_s42/ckpt/step-012000.pt"
    same_parts = {"model.programmer": {**PAIR_PROGRAMMER, "d_model": 256, "d_ff": 1024}, "model.executor.d_model": 128, "model.executor.d_ff": 512,
                  "model.code.type": "vq", "model.state_update": "reencode_hard", "method.type": "neo_warm", "train.init_from": start, "train.ema": 0.0}
    for name, changes in {"warm_all": same_parts, "warm_programmer_only": {**same_parts, "method.freeze": ["encoder", "decoder", "code", "executor"]},
                          "cold_same_parts": {**same_parts, "method.type": "neo", "train.init_from": "runs/codec_token/ckpt/step-003000.pt"}}.items():
        config = fast_neo(*SINGLE_A_LV1, 42, steps=6000, codes=16, changes=changes)
        for key in ("restart_every", "restart_below", "usage_decay"):
            config["model"]["code"].pop(key, None)
        configs[f"fastneo_6k_{name}_single_a_lv1_s42"] = config
    # the same warm start with a tenth of the learning rate: is the damage only a learning rate that is too high for a
    # model that already works?
    for name, changes in {"warm_all_lr3e5": {**same_parts, "method.lr": 3.0e-5}, "warm_programmer_only_lr3e5": {**same_parts, "method.lr": 3.0e-5, "method.freeze": ["encoder", "decoder", "code", "executor"]}}.items():
        config = fast_neo(*SINGLE_A_LV1, 42, steps=3000, codes=16, changes=changes)
        for key in ("restart_every", "restart_below", "usage_decay"):
            config["model"]["code"].pop(key, None)
        configs[f"fastneo_3k_{name}_single_a_lv1_s42"] = config
    # the one-step model was never trained with the grounding term (it is skipped when a program has one step), and at
    # the warm start that term is as large as the reconstruction loss. The warm start without it:
    for name, changes in {"warm_all_g0": {**same_parts, "method.grounding_weight": 0.0},
                          "warm_programmer_only_g0": {**same_parts, "method.grounding_weight": 0.0, "method.freeze": ["encoder", "decoder", "code", "executor"]},
                          "warm_all_g0_lr3e5": {**same_parts, "method.grounding_weight": 0.0, "method.lr": 3.0e-5}}.items():
        config = fast_neo(*SINGLE_A_LV1, 42, steps=3000, codes=16, changes=changes)
        for key in ("restart_every", "restart_below", "usage_decay"):
            config["model"]["code"].pop(key, None)
        configs[f"fastneo_3k_{name}_single_a_lv1_s42"] = config
    # programs that grow longer step by step (asked by the user on 2026-11-22): from the one-step model, first programs of up
    # to two steps on tasks of depth at most two, then up to four steps on depth at most four. The jump from one step to
    # five destroyed the model within a few hundred steps; is a gradual schedule different? Two learning rates; the
    # evaluation always rolls out five steps with only the grid handed over, so the stages are comparable.
    for tag, lr_changes in (("", {}), ("_lr3e5_g0", {"method.lr": 3.0e-5, "method.grounding_weight": 0.0})):
        previous = start
        for length in (2, 4):
            name = f"fastneo_3k_len{length}{tag}_single_a_lv1_s42"
            config = fast_neo(*SINGLE_A_LV1, 42, steps=3000, codes=16, max_length=length, changes={**same_parts, **lr_changes, "train.init_from": previous, "train.ckpt_every": 3000})
            for key in ("restart_every", "restart_below", "usage_decay"):
                config["model"]["code"].pop(key, None)
            config["data"]["train_max_depth"] = length
            for block in config["evals"]:
                block["infer"] = {"max_steps": 5, "rollout": "hard"}
                block["every"] = 300 if block.get("max_samples") else block["every"]
            configs[name] = config
            previous = f"runs/{name}/ckpt/step-003000.pt"
    # the gradual program length on the two other seeds of the one-step model (learning rate 3e-5, no grounding term)
    for seed in (43, 44):
        previous = f"runs/onestep_t_w_12k_s{seed}/ckpt/step-012000.pt"
        for length in (2, 4):
            name = f"fastneo_3k_len{length}_lr3e5_g0_single_a_lv1_s{seed}"
            config = fast_neo(*SINGLE_A_LV1, seed, steps=3000, codes=16, max_length=length, changes={**same_parts, "method.lr": 3.0e-5, "method.grounding_weight": 0.0,
                                                                                                  "train.init_from": previous, "train.ckpt_every": 3000})
            for key in ("restart_every", "restart_below", "usage_decay"):
                config["model"]["code"].pop(key, None)
            config["data"]["train_max_depth"] = length
            for block in config["evals"]:
                block["infer"] = {"max_steps": 5, "rollout": "hard"}
                block["every"] = 300 if block.get("max_samples") else block["every"]
            configs[name] = config
            previous = f"runs/{name}/ckpt/step-003000.pt"
    # what made the length-2 stage keep the model (84% at 3,000 steps, where the jump to five steps fell to 16%): the
    # shorter rollout or the pairs of depth at most two? One changed at a time.
    for name, length, depth in (("len2_alldata", 2, None), ("len5_depth2", 5, 2)):
        config = fast_neo(*SINGLE_A_LV1, 42, steps=3000, codes=16, max_length=length, changes={**same_parts, "train.ckpt_every": 3000})
        for key in ("restart_every", "restart_below", "usage_decay"):
            config["model"]["code"].pop(key, None)
        if depth:
            config["data"]["train_max_depth"] = depth
        for block in config["evals"]:
            block["infer"] = {"max_steps": 5, "rollout": "hard"}
            block["every"] = 300 if block.get("max_samples") else block["every"]
        configs[f"fastneo_3k_{name}_single_a_lv1_s42"] = config
    # seeds: the one-step result moved from 83% to 43% over three seeds, and the multi-step runs with the searched parts
    # have one seed each. Two more seeds of the multi-step model with grid-only handover (12 and 16 codes), and three more
    # of the one-step run
    for seed in (43, 44):
        for codes in (12, 16):
            configs[f"fastneo_12k_hard_c{codes}_norestart_single_a_lv1_s{seed}"] = fast_neo(*SINGLE_A_LV1, seed, steps=12000, codes=codes, changes=hard)
    for seed in (45, 46, 47):
        configs[f"onestep_t_w_12k_s{seed}"] = one_step(seed, steps=12000, codes=16, changes=wide_512)
    for name, (codes, changes) in fast_neo_screen_1().items():
        for seed in ((42, 43) if name == "c16" else (42,)):
            configs[f"fastneo_12k_{name}_single_a_lv1_s{seed}"] = fast_neo(*SINGLE_A_LV1, seed, codes=codes, changes=changes)
    for name, (codes, steps, changes) in one_step_screen_4().items():
        for seed in ((42, 43) if name in ("q_r", "q_r_w") else (42,)):
            configs[f"onestep_{name}_s{seed}"] = one_step(seed, steps=steps, codes=codes, changes=changes)
    for name, (codes, changes) in ONE_STEP_SCREEN_2.items():
        for seed in ((42, 43) if name == "pair" else (42,)):
            configs[f"onestep_{name}_s{seed}"] = one_step(seed, codes=codes, changes=changes)
    # the full five-step model with the cheaper programmer, for timing and later for the long runs
    fast = neo_transformer(*SINGLE_A_LV1, 42, steps=36000)
    fast["model"]["programmer"] = dict(PAIR_PROGRAMMER)
    fast["train"]["compile"] = "parts"
    configs["neo_pair_36k_single_a_lv1_s42"] = fast
    for seed in (45, 46, 47, 48):
        configs[f"neo_transformer_12k_single_a_lv1_s{seed}"] = neo_transformer(*SINGLE_A_LV1, seed, steps=12000)
    for seed in (42, 43, 44):
        # the lab schedule is 78,200 steps; at 15x15 that is two days per run, so the first look uses 12,000 (about 25 passes over the data)
        configs[f"neo_transformer_12k_single_a_lv1_s{seed}"] = neo_transformer(*SINGLE_A_LV1, seed, steps=12000)
        configs[f"neo_transformer_12k_translate_lv1_s{seed}"] = neo_transformer(*TRANSLATE_LV1, seed, steps=12000)
        depth_one = neo_transformer(*SINGLE_A_LV1, seed, steps=12000)  # diagnostic: trained on single-primitive tasks only
        depth_one["data"]["train_max_depth"] = 1
        configs[f"neo_transformer_12k_depth1_single_a_lv1_s{seed}"] = depth_one
        for state_dim in (32, 256):
            configs[f"neo_mlp_d{state_dim}_single_a_lv1_s{seed}"] = neo_mlp(*SINGLE_A_LV1, seed, state_dim)
        # the lab's MLP width of 128 is narrower than a 128-d or 256-d state; these widen the programmer and executor only
        configs[f"neo_mlp_d128_h512_single_a_lv1_s{seed}"] = neo_mlp(*SINGLE_A_LV1, seed, 128, hidden=512)
        configs[f"neo_mlp_d256_h1024_single_a_lv1_s{seed}"] = neo_mlp(*SINGLE_A_LV1, seed, 256, hidden=1024)
        configs[f"neo_mlp_d32_translate_lv1_s{seed}"] = neo_mlp(*TRANSLATE_LV1, seed, 32, codes=6)
        configs[f"neo_mlp_d256_h1024_translate_lv1_s{seed}"] = neo_mlp(*TRANSLATE_LV1, seed, 256, hidden=1024, codes=6)
    for name, config in configs.items():
        (OUT / f"{name}.yaml").write_text(yaml.dump(config, Dumper=Flow, sort_keys=False, default_flow_style=None, width=200))
    print(sorted(configs))


if __name__ == "__main__":
    main()
