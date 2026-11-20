"""Random GridWorld-shaped data for CPU tests; same loader contract as GridworldPairs."""

from __future__ import annotations

import torch


class SyntheticData:
    def __init__(self, n: int = 8, grid: int = 10, colors: int = 9, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.grids = torch.randint(0, colors, (n, 3, grid, grid), generator=g)
        self.answer = torch.randint(0, colors, (n, grid, grid), generator=g)
        self.splits = ("id",)

    def train_loader(self, batch_size, steps, start_step, seed, rank, world_size, num_workers, pin_memory):
        for _ in range(steps - start_step):
            idx = torch.arange(min(batch_size, len(self.grids)))
            yield {"x": torch.cat([self.grids[idx, 0], self.grids[idx, 2]]), "y": torch.cat([self.grids[idx, 1], self.answer[idx]])}

    def eval_loader(self, split, batch_size, num_workers=0, max_samples=None):
        n = len(self.grids) if max_samples is None else min(max_samples, len(self.grids))
        for start in range(0, n, batch_size):
            idx = torch.arange(start, min(start + batch_size, n))
            yield {"support_x": self.grids[idx, 0], "support_y": self.grids[idx, 1], "query_x": self.grids[idx, 2], "query_y": self.answer[idx], "transitions": [None] * len(idx)}
