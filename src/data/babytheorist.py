"""BabyTheorist HDF5 files as NEO batches. Registered as DATASETS["babytheorist_pairs"].

A train file holds one (x, y) pair per task. An eval file holds, per task, K support pairs in slots [0, K) and the query in
slot K (K is 2..5 and can vary by row). The stored program and its depth are passed through for diagnostics only; nothing
here feeds them to a loss.

    data: {type: babytheorist_pairs, train: [data/babytheorist/single_a/lv1_train.h5], evals: {lv1: data/babytheorist/single_a/lv1_eval.h5}}
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.registry import DATASETS
from data.datasets import RepeatedDistributedBatchSampler


class TrainPairs(Dataset):
    """All pairs of the given train files in memory; a batch of indices is returned as one dict."""

    def __init__(self, paths: list[str]) -> None:
        xs, ys = [], []
        for path in paths:
            with h5py.File(path, "r") as file:
                xs.append(torch.from_numpy(file["inputs"][:, 0].astype(np.uint8)))
                ys.append(torch.from_numpy(file["outputs"][:, 0].astype(np.uint8)))
        self.x, self.y = torch.cat(xs), torch.cat(ys)

    def __len__(self) -> int:
        return len(self.x)

    def __getitems__(self, indices: list[int]) -> dict:
        index = torch.as_tensor(indices)
        return {"x": self.x[index].long(), "y": self.y[index].long()}


class EvalTasks(Dataset):
    def __init__(self, path: str, max_samples: int | None = None) -> None:
        with h5py.File(path, "r") as file:
            count = len(file["depths"]) if max_samples is None else min(max_samples, len(file["depths"]))
            self.inputs = torch.from_numpy(file["inputs"][:count].astype(np.uint8))
            self.outputs = torch.from_numpy(file["outputs"][:count].astype(np.uint8))
            self.depth = torch.from_numpy(file["depths"][:count].astype(np.int64))
            slots = self.inputs.shape[1]
            self.n_support = torch.from_numpy(file["n_support"][:count].astype(np.int64)) if "n_support" in file else torch.full((count,), slots - 1)
            self.programs = [p.decode() if isinstance(p, bytes) else str(p) for p in file["transformation_suites"][:count]]

    def __len__(self) -> int:
        return len(self.depth)

    def __getitems__(self, indices: list[int]) -> dict:
        index = torch.as_tensor(indices)
        inputs, outputs, k = self.inputs[index].long(), self.outputs[index].long(), self.n_support[index]
        rows = torch.arange(len(index))
        most = int(k.max())
        return {
            "support_x": inputs[:, 0], "support_y": outputs[:, 0],  # the first support pair, for the single-support protocol
            "supports_x": inputs[:, :most], "supports_y": outputs[:, :most], "n_support": k,
            "query_x": inputs[rows, k], "query_y": outputs[rows, k],
            "depth": self.depth[index], "transitions": [self.programs[i] for i in indices], "index": index,
        }


def passthrough(batch: dict) -> dict:
    return batch


class BabyTheoristPairs:
    def __init__(self, train: list[str] | str, evals: dict[str, str] | None = None) -> None:
        self.train_paths = [train] if isinstance(train, str) else list(train)
        self.eval_paths = dict(evals or {})
        for path in [*self.train_paths, *self.eval_paths.values()]:
            if not Path(path).exists():
                raise FileNotFoundError(path)
        self._train: TrainPairs | None = None

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self.eval_paths)

    def train_loader(self, batch_size: int, steps: int, start_step: int, seed: int, rank: int, world_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
        if self._train is None:
            self._train = TrainPairs(self.train_paths)
        sampler = RepeatedDistributedBatchSampler(len(self._train), batch_size, steps - start_step, seed=seed, rank=rank, world_size=world_size, start_step=start_step)
        return DataLoader(self._train, batch_sampler=sampler, collate_fn=passthrough, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)

    def eval_loader(self, split: str, batch_size: int, num_workers: int = 0, max_samples: int | None = None) -> DataLoader:
        return DataLoader(EvalTasks(self.eval_paths[split], max_samples), batch_size=batch_size, shuffle=False, collate_fn=passthrough, num_workers=num_workers)


DATASETS["babytheorist_pairs"] = BabyTheoristPairs
