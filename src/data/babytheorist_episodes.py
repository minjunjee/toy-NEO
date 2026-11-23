"""BabyTheorist train tasks of several pairs that share one hidden program. Registered as DATASETS["babytheorist_episodes"].

A train file written by generate_many_primitives.py --train-pairs 2 holds, for every task, pairs from the same program on
different objects. A batch carries the first pair as x, y (as babytheorist_pairs does) and the second as x2, y2. The
learner is told which pairs go together, as in an ARC task or the lab's GridWorld episodes; it is never told the program.
Evaluation is the plain BabyTheorist evaluation.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.registry import DATASETS
from data.babytheorist import BabyTheoristPairs, RepeatedDistributedBatchSampler, passthrough


class TrainEpisodes(Dataset):
    def __init__(self, paths: list[str], max_depth: int | None = None) -> None:
        xs, ys = [], []
        for path in paths:
            with h5py.File(path, "r") as file:
                if file["inputs"].shape[1] < 2:
                    raise ValueError(f"{path}: train tasks need two pairs or more")
                keep = slice(None) if max_depth is None else np.nonzero(file["depths"][:] <= max_depth)[0]
                xs.append(torch.from_numpy(file["inputs"][:, :2][keep].astype(np.uint8)))
                ys.append(torch.from_numpy(file["outputs"][:, :2][keep].astype(np.uint8)))
        self.x, self.y = torch.cat(xs), torch.cat(ys)

    def __len__(self) -> int:
        return len(self.x)

    def __getitems__(self, indices: list[int]) -> dict:
        index = torch.as_tensor(indices)
        x, y = self.x[index].long(), self.y[index].long()
        return {"x": x[:, 0], "y": y[:, 0], "x2": x[:, 1], "y2": y[:, 1]}


class BabyTheoristEpisodes(BabyTheoristPairs):
    def train_loader(self, batch_size: int, steps: int, start_step: int, seed: int, rank: int, world_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
        if self._train is None:
            self._train = TrainEpisodes(self.train_paths, self.train_max_depth)
        sampler = RepeatedDistributedBatchSampler(len(self._train), batch_size, steps - start_step, seed=seed, rank=rank, world_size=world_size, start_step=start_step)
        return DataLoader(self._train, batch_sampler=sampler, collate_fn=passthrough, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)


DATASETS["babytheorist_episodes"] = BabyTheoristEpisodes
