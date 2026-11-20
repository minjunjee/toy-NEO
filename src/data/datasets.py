"""Batch composition over h5 samples. Registered under DATASETS."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset

from data.reader import H5Samples
from core.registry import DATASETS


class RepeatedDistributedBatchSampler(Sampler[list[int]]):
    """Exact global batches across shuffled-epoch boundaries, per rank (WTM data/sampler.py)."""

    def __init__(self, dataset_size: int, global_batch_size: int, num_steps: int, *, seed: int, rank: int = 0, world_size: int = 1, start_step: int = 0) -> None:
        if dataset_size < 1 or global_batch_size % world_size:
            raise ValueError("dataset size must be positive and batch size divisible by world size")
        self.dataset_size = dataset_size
        self.global_batch_size = global_batch_size
        self.num_steps = num_steps
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.start_step = start_step

    def __len__(self) -> int:
        return self.num_steps

    def __iter__(self) -> Iterator[list[int]]:
        position = self.start_step * self.global_batch_size
        epoch, offset = divmod(position, self.dataset_size)
        permutation = self._permutation(epoch)
        local_size = self.global_batch_size // self.world_size
        for _ in range(self.num_steps):
            pieces = []
            remaining = self.global_batch_size
            while remaining:
                take = min(remaining, self.dataset_size - offset)
                pieces.append(permutation[offset : offset + take])
                offset += take
                remaining -= take
                if offset == self.dataset_size:
                    epoch += 1
                    offset = 0
                    permutation = self._permutation(epoch)
            batch = torch.cat(pieces)
            begin = self.rank * local_size
            yield batch[begin : begin + local_size].tolist()

    def _permutation(self, epoch: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + epoch)
        return torch.randperm(self.dataset_size, generator=generator)


def collate_pairs(samples: list[dict]) -> dict:
    """Each stored episode gives two independent pairs: (grids[0] -> grids[1]) and (grids[2] -> answer)."""

    grids = torch.stack([s["grids"] for s in samples])
    answer = torch.stack([s["answer"] for s in samples])
    return {"x": torch.cat([grids[:, 0], grids[:, 2]]), "y": torch.cat([grids[:, 1], answer])}


def collate_episodes(samples: list[dict]) -> dict:
    grids = torch.stack([s["grids"] for s in samples])
    return {
        "support_x": grids[:, 0], "support_y": grids[:, 1], "query_x": grids[:, 2],
        "query_y": torch.stack([s["answer"] for s in samples]),
        "transitions": [s.get("transitions") for s in samples],
    }


class FlatPairs(Dataset):
    """WTM GridWorldPairDataset view: index 2i -> (grids[0], grids[1]) of episode i, 2i+1 -> (grids[2], answer)."""

    def __init__(self, episodes: H5Samples) -> None:
        self.episodes = episodes

    def __len__(self) -> int:
        return 2 * len(self.episodes)

    def __getitem__(self, index: int) -> dict:
        sample, pair = divmod(index, 2)
        s = self.episodes[sample]
        return {"x": s["grids"][0], "y": s["grids"][1]} if pair == 0 else {"x": s["grids"][2], "y": s["answer"]}


def collate_flat_pairs(samples: list[dict]) -> dict:
    return {"x": torch.stack([s["x"] for s in samples]), "y": torch.stack([s["y"] for s in samples])}


class GridworldPairs:
    """Official GridWorld artifacts: training on independent pairs, evaluation on (support, query) episodes.

    sampling="episode": a batch of batch_size episodes, each contributing both pairs (official NEO loader).
    sampling="pair": a batch of batch_size pairs drawn i.i.d. from the 2N flattened pairs (WTM loader).
    """

    def __init__(self, train: str, id: str, ood_comp: str | None = None, ood_len: str | None = None, sampling: str = "pair") -> None:
        if sampling not in ("episode", "pair"):
            raise ValueError(f"sampling must be 'episode' or 'pair', got {sampling!r}")
        self.sampling = sampling
        self.paths = {"train": train, "id": id}
        if ood_comp:
            self.paths["ood_comp"] = ood_comp
        if ood_len:
            self.paths["ood_len"] = ood_len

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(k for k in self.paths if k != "train")

    def train_loader(self, batch_size: int, steps: int, start_step: int, seed: int, rank: int, world_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
        dataset, collate = H5Samples(self.paths["train"]), collate_pairs
        if self.sampling == "pair":
            dataset, collate = FlatPairs(dataset), collate_flat_pairs
        sampler = RepeatedDistributedBatchSampler(len(dataset), batch_size, steps - start_step, seed=seed, rank=rank, world_size=world_size, start_step=start_step)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)

    def eval_loader(self, split: str, batch_size: int, num_workers: int = 0, max_samples: int | None = None) -> DataLoader:
        dataset = H5Samples(self.paths[split])
        if max_samples is not None:
            dataset = Subset(dataset, range(min(max_samples, len(dataset))))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_episodes, num_workers=num_workers)


DATASETS["gridworld_pairs"] = GridworldPairs
