"""h5 contract: one group per sample with ``grids`` (N,H,W) and ``answer`` (H,W); every other dataset/attr passes through."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5Samples(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with h5py.File(self.path, "r") as file:
            self.length = int(file.attrs.get("dataset_length", sum(1 for k in file if k.startswith("sample_"))))
        self._file = None
        self._pid = None

    def _open(self) -> h5py.File:
        if self._file is None or self._pid != os.getpid():
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._file

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        group = self._open()[f"sample_{index}"]
        sample = {key: torch.from_numpy(np.asarray(group[key][()])) for key in group}
        sample.update({key: value for key, value in group.attrs.items()})
        sample["grids"] = sample["grids"].long()
        sample["answer"] = sample["answer"].long()
        return sample
