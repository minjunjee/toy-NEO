"""Run config: one yaml with exactly five top-level blocks, all fields explicit."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

TOP_LEVEL_KEYS = ("data", "model", "method", "evals", "train")


@dataclass(frozen=True)
class TrainConfig:
    steps: int
    batch_size: int
    seed: int
    precision: str
    ckpt_every: int
    log_every: int
    num_workers: int
    init_from: str | None
    wandb_project: str
    wandb_mode: str = "online"
    ema: float = 0.0  # decay of an exponential moving average of the weights used for eval and checkpoints; 0 = off

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        if self.ckpt_every < 1 or self.log_every < 1:
            raise ValueError("ckpt_every and log_every must be positive")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if not 0.0 <= self.ema < 1.0:
            raise ValueError("ema must be in [0, 1)")


def load_config(path: str | Path) -> dict:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"{path}: config must be a yaml mapping")
    keys = set(values)
    expected = set(TOP_LEVEL_KEYS)
    if keys != expected:
        raise ValueError(f"{path}: top-level keys must be exactly {sorted(expected)}, got {sorted(keys)}")
    if not isinstance(values["evals"], list):
        raise ValueError(f"{path}: evals must be a list of blocks")
    for block in values["evals"]:
        for key in ("type", "every", "split"):
            if key not in block:
                raise ValueError(f"{path}: every eval block needs '{key}'")
    train_config(values["train"])
    return values


def train_config(block: dict) -> TrainConfig:
    names = {f.name for f in fields(TrainConfig)}
    unknown = set(block) - names
    if unknown:
        raise KeyError(f"train: unknown keys {sorted(unknown)}")
    return TrainConfig(**block)
