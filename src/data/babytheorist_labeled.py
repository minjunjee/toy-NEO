"""BabyTheorist training pairs with the hidden primitive's index. Registered as DATASETS["babytheorist_pairs_labeled"].

For diagnostics only: the label is never available to a learner in the benchmark's protocol. Only single-primitive pairs
(depth 1) are kept. `label` is the position of the pair's primitive in the sorted list of the primitives that occur in the
train files. Evaluation is the plain BabyTheorist evaluation.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.registry import DATASETS
from data.babytheorist import BabyTheoristPairs, RepeatedDistributedBatchSampler, passthrough


class LabeledPairs(Dataset):
    def __init__(self, paths: list[str]) -> None:
        xs, ys, names = [], [], []
        for path in paths:
            with h5py.File(path, "r") as file:
                keep = np.nonzero(file["depths"][:] == 1)[0]
                xs.append(torch.from_numpy(file["inputs"][:, 0][keep].astype(np.uint8)))
                ys.append(torch.from_numpy(file["outputs"][:, 0][keep].astype(np.uint8)))
                programs = file["transformation_suites"]
                names += [json.loads(p.decode() if isinstance(p, bytes) else str(p))[0]["primitive"] for p in programs[keep]]
        self.x, self.y = torch.cat(xs), torch.cat(ys)
        self.primitives = sorted(set(names))
        self.label = torch.tensor([self.primitives.index(n) for n in names])

    def __len__(self) -> int:
        return len(self.x)

    def __getitems__(self, indices: list[int]) -> dict:
        index = torch.as_tensor(indices)
        return {"x": self.x[index].long(), "y": self.y[index].long(), "label": self.label[index]}


class BabyTheoristPairsLabeled(BabyTheoristPairs):
    def train_loader(self, batch_size: int, steps: int, start_step: int, seed: int, rank: int, world_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
        if self._train is None:
            self._train = LabeledPairs(self.train_paths)
        sampler = RepeatedDistributedBatchSampler(len(self._train), batch_size, steps - start_step, seed=seed, rank=rank, world_size=world_size, start_step=start_step)
        return DataLoader(self._train, batch_sampler=sampler, collate_fn=passthrough, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)


DATASETS["babytheorist_pairs_labeled"] = BabyTheoristPairsLabeled
