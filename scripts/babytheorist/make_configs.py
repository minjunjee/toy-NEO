#!/usr/bin/env python3
"""Write the run configs under configs/babytheorist/. One yaml per run, as the lab code expects.

    python scripts/babytheorist/make_configs.py
"""

from __future__ import annotations

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


def one_step(seed: int, steps: int = 6000, codes: int = 16, changes: dict | None = None) -> dict:
    """The transformer NEO cut down to one program step, trained on single-primitive tasks only.

    The programmer picks one code from a pair and the executor applies it once, so a run costs a fifth of the full one.
    It asks one thing of the standard parts: can they learn every primitive as one code that works on any input? The full
    file is still evaluated, but only the single-primitive tasks (depth 1) can be solved. `changes` overrides settings by
    dotted path, for searching over the parts."""

    config = neo_transformer(*SINGLE_A_LV1, seed, steps=steps, codes=codes, max_length=1)
    config["data"]["train_max_depth"] = 1
    config["train"]["compile"] = "parts"
    for path, value in (changes or {}).items():
        set_path(config, path, value)
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
