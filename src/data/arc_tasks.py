"""ARC tasks as NEO batches. Registered as DATASETS["arc_tasks"].

The train file (scripts/arc/prepare_arc.py) holds many examples of each task on a padded square canvas. A train sample is
an episode of one task variant: `pairs` input and output grids that share one rule. A variant is the task seen through one
of the eight symmetries of the square and one of `color_perms` fixed permutations of the colours 1 to 9 (0 and the pad
value stay); the same symmetry and permutation are applied to the inputs and outputs of all the episode's pairs, so the
pairs still share one rule, though it is usually a different rule from the original task's. The batch carries the task
index and the variant index, which the model may use to look up a code of its own for that variant; nothing else about the
task is given.

Evaluation files hold ARC tasks as they are given: demonstration pairs and test pairs, never transformed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.registry import DATASETS
from data.babytheorist import passthrough
from data.datasets import RepeatedDistributedBatchSampler

PAD = 10


def color_tables(count: int, seed: int = 0) -> np.ndarray:
    """`count` lookup tables over the 11 cell values; the first is the identity, 0 and the pad value never move."""
    rng = np.random.default_rng(seed)
    tables = np.tile(np.arange(11, dtype=np.uint8), (count, 1))
    for k in range(1, count):
        tables[k, 1:10] = 1 + rng.permutation(9)
    return tables


def transform(grid: np.ndarray, height: int, width: int, symmetry: int, table: np.ndarray, size: int) -> np.ndarray:
    g = grid[:height, :width]
    if symmetry >= 4:
        g = g[:, ::-1]
    g = np.rot90(g, symmetry % 4)
    out = np.full((size, size), PAD, dtype=np.uint8)
    out[: g.shape[0], : g.shape[1]] = table[g]
    return out


class ArcEpisodes(Dataset):
    def __init__(self, path: str, pairs: int, tables: np.ndarray, symmetries: int, seed: int, tasks: int | None = None) -> None:
        with np.load(path) as f:
            self.inputs, self.outputs, self.shapes, self.task = f["inputs"], f["outputs"], f["shapes"], f["task"].astype(np.int64)
        if tasks is not None:                      # a prefix of the tasks, for small trials
            keep = self.task < tasks
            self.inputs, self.outputs, self.shapes, self.task = self.inputs[keep], self.outputs[keep], self.shapes[keep], self.task[keep]
        self.size = self.inputs.shape[-1]
        self.tasks = int(self.task.max()) + 1
        order = np.argsort(self.task, kind="stable")
        bounds = np.searchsorted(self.task[order], np.arange(self.tasks + 1))
        self.members = [order[bounds[t]: bounds[t + 1]] for t in range(self.tasks)]
        self.pairs, self.symmetries, self.seed = pairs, symmetries, seed
        self.tables = tables
        self.variants = symmetries * len(tables)

    def __len__(self) -> int:
        return len(self.task)

    def __getitems__(self, indices: list[int]) -> dict:
        rng = np.random.default_rng([self.seed, indices[0], indices[-1], len(indices)])
        n, k, size = len(indices), self.pairs, self.size
        grids = np.empty((n, k, 2, size, size), dtype=np.uint8)
        variant = rng.integers(0, self.variants, size=n)
        for row, index in enumerate(indices):
            members = self.members[self.task[index]]
            others = members[rng.integers(0, len(members), size=k - 1)]
            symmetry, table = int(variant[row] % self.symmetries), self.tables[variant[row] // self.symmetries]
            for slot, example in enumerate([index, *others]):
                h_in, w_in, h_out, w_out = self.shapes[example]
                grids[row, slot, 0] = transform(self.inputs[example], h_in, w_in, symmetry, table, size)
                grids[row, slot, 1] = transform(self.outputs[example], h_out, w_out, symmetry, table, size)
        grids = torch.from_numpy(grids).long()
        return {"inputs": grids[:, :, 0], "outputs": grids[:, :, 1], "task": torch.from_numpy(self.task[indices]), "variant": torch.from_numpy(variant)}


class ArcTasks:
    def __init__(self, train: str, evals: dict[str, str] | None = None, pairs: int = 4, color_perms: int = 8, symmetries: int = 8,
                 seed: int = 0, tasks: int | None = None) -> None:
        self.train_path, self.eval_paths = train, dict(evals or {})
        for path in [train, *self.eval_paths.values()]:
            if not Path(path).exists():
                raise FileNotFoundError(path)
        self.pairs, self.symmetries, self.seed, self.tasks = pairs, symmetries, seed, tasks
        self.tables = color_tables(color_perms, seed)
        self.variants = symmetries * color_perms
        self._train: ArcEpisodes | None = None

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self.eval_paths)

    def episodes(self) -> ArcEpisodes:
        if self._train is None:
            self._train = ArcEpisodes(self.train_path, self.pairs, self.tables, self.symmetries, self.seed, self.tasks)
        return self._train

    def train_loader(self, batch_size: int, steps: int, start_step: int, seed: int, rank: int, world_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
        data = self.episodes()
        sampler = RepeatedDistributedBatchSampler(len(data), batch_size, steps - start_step, seed=seed, rank=rank, world_size=world_size, start_step=start_step)
        return DataLoader(data, batch_sampler=sampler, collate_fn=passthrough, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)

    def eval_tasks(self, split: str) -> dict[str, np.ndarray]:
        with np.load(self.eval_paths[split]) as f:
            return {key: f[key] for key in f.files}


DATASETS["arc_tasks"] = ArcTasks
