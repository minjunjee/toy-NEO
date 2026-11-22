#!/usr/bin/env python3
"""Write the run configs under configs/babytheorist/. One yaml per run, as the lab code expects.

    python scripts/babytheorist/make_configs.py
"""

from __future__ import annotations

import copy

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
