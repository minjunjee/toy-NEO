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
    """Episodes of one train file. `named` False hides the task index (-1), for tasks that own no code in the model's table."""

    def __init__(self, path: str, pairs: int, tables: np.ndarray, symmetries: int, seed: int, tasks: int | None = None, named: bool = True) -> None:
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
        self.place = np.empty(len(self.task), dtype=np.int64)          # where an example sits among its task's members
        for members in self.members:
            self.place[members] = np.arange(len(members))
        self.pairs, self.symmetries, self.seed, self.named = pairs, symmetries, seed, named
        self.tables = tables
        self.variants = symmetries * len(tables)

    def __len__(self) -> int:
        return len(self.task)

    def rows(self, indices, rng: np.random.Generator) -> dict:
        """One episode per index: that example and up to pairs - 1 other examples of its task, all different. A task with
        fewer examples than `pairs` gives a shorter episode; `mask` marks the pairs that are there."""

        n, k, size = len(indices), self.pairs, self.size
        grids = np.full((n, k, 2, size, size), PAD, dtype=np.uint8)
        mask = np.zeros((n, k), dtype=bool)
        variant = rng.integers(0, self.variants, size=n)
        for row, index in enumerate(indices):
            members = self.members[self.task[index]]
            others = rng.choice(len(members) - 1, size=min(k - 1, len(members) - 1), replace=False)
            others = members[others + (others >= self.place[index])]
            symmetry, table = int(variant[row] % self.symmetries), self.tables[variant[row] // self.symmetries]
            for slot, example in enumerate([index, *others]):
                h_in, w_in, h_out, w_out = self.shapes[example]
                grids[row, slot, 0] = transform(self.inputs[example], h_in, w_in, symmetry, table, size)
                grids[row, slot, 1] = transform(self.outputs[example], h_out, w_out, symmetry, table, size)
                mask[row, slot] = True
        task = self.task[np.asarray(indices)] if self.named else np.full(n, -1, dtype=np.int64)
        return {"inputs": grids[:, :, 0], "outputs": grids[:, :, 1], "mask": mask, "task": task, "variant": variant}

    def __getitems__(self, indices: list[int]) -> dict:
        rng = np.random.default_rng([self.seed, indices[0], indices[-1], len(indices)])
        return {key: torch.from_numpy(value).long() if value.dtype != bool else torch.from_numpy(value) for key, value in self.rows(indices, rng).items()}


class ArcMixedEpisodes(Dataset):
    """Batches drawn from several train files in fixed shares: the first rows of a batch come from the first file, and so
    on. The sampler's indices only seed the draw. Used for a sleep: new tasks mixed with replayed old ones."""

    def __init__(self, sources: list[ArcEpisodes], shares: list[float], seed: int) -> None:
        if len(sources) != len(shares) or abs(sum(shares) - 1.0) > 1e-6:
            raise ValueError("one share per train file, summing to 1")
        self.sources, self.shares, self.seed = sources, shares, seed

    def __len__(self) -> int:
        return sum(len(source) for source in self.sources)

    def __getitems__(self, indices: list[int]) -> dict:
        rng = np.random.default_rng([self.seed, indices[0], indices[-1], len(indices)])
        n = len(indices)
        counts = [int(round(share * n)) for share in self.shares[:-1]]
        counts.append(n - sum(counts))
        parts = []
        for number, (source, count) in enumerate(zip(self.sources, counts)):
            if count > 0:
                parts.append({**source.rows(rng.integers(0, len(source), size=count), rng), "source": np.full(count, number, dtype=np.int64)})
        merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        return {key: torch.from_numpy(value).long() if value.dtype != bool else torch.from_numpy(value) for key, value in merged.items()}


class ArcTasks:
    """train: one file, or several with `shares` (the share of a batch drawn from each) and `named` (whether each file's
    task index is passed to the model; by default only the first file's is)."""

    def __init__(self, train: str | list[str], evals: dict[str, str] | None = None, pairs: int = 4, color_perms: int = 8, symmetries: int = 8,
                 seed: int = 0, tasks: int | None = None, shares: list[float] | None = None, named: list[bool] | None = None) -> None:
        self.train_paths = [train] if isinstance(train, str) else list(train)
        self.eval_paths = dict(evals or {})
        for path in [*self.train_paths, *self.eval_paths.values()]:
            if not Path(path).exists():
                raise FileNotFoundError(path)
        self.shares = shares
        self.named = named if named is not None else [i == 0 for i in range(len(self.train_paths))]
        self.pairs, self.symmetries, self.seed, self.tasks = pairs, symmetries, seed, tasks
        self.tables = color_tables(color_perms, seed)
        self.variants = symmetries * color_perms
        self._train: Dataset | None = None

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self.eval_paths)

    def episodes(self) -> Dataset:
        if self._train is None:
            sources = [ArcEpisodes(path, self.pairs, self.tables, self.symmetries, self.seed, self.tasks, named) for path, named in zip(self.train_paths, self.named)]
            self._train = sources[0] if len(sources) == 1 else ArcMixedEpisodes(sources, self.shares or [1 / len(sources)] * len(sources), self.seed)
        return self._train

    def train_loader(self, batch_size: int, steps: int, start_step: int, seed: int, rank: int, world_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
        data = self.episodes()
        sampler = RepeatedDistributedBatchSampler(len(data), batch_size, steps - start_step, seed=seed, rank=rank, world_size=world_size, start_step=start_step)
        return DataLoader(data, batch_sampler=sampler, collate_fn=passthrough, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)

    def eval_tasks(self, split: str) -> dict[str, np.ndarray]:
        with np.load(self.eval_paths[split]) as f:
            return {key: f[key] for key in f.files}


DATASETS["arc_tasks"] = ArcTasks
