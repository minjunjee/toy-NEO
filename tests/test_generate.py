"""Tiny profile -> h5 -> reader round trip, using the vendored generator."""

from pathlib import Path

import pytest

from data.datasets import GridworldPairs
from data.gridworld.hdf5 import write_profile
from data.gridworld.profiles import GridWorldProfile, SplitSpec
from data.gridworld.shape_pool import load_shape_pool
from data.reader import H5Samples

SHAPE_POOL = Path(__file__).resolve().parents[1] / "data" / "gridworld" / "shape_pool.pkl"


@pytest.mark.skipif(not SHAPE_POOL.exists(), reason="shape pool not available")
def test_generate_and_read(tmp_path: Path) -> None:
    profile = GridWorldProfile(
        name="tiny", output_prefix="tiny", min_program_length=1, max_program_length=3, split_mode="alpha", alpha=0.33,
        splits=(SplitSpec("practice", 6, batch_size=32), SplitSpec("exam", 2, batch_size=1)),
    )
    paths = write_profile(profile, load_shape_pool(SHAPE_POOL), tmp_path)
    samples = H5Samples(paths[0])
    assert len(samples) == 6
    sample = samples[0]
    assert tuple(sample["grids"].shape) == (3, 10, 10) and tuple(sample["answer"].shape) == (10, 10)
    assert isinstance(sample["transitions"], str)
    data = GridworldPairs(train=str(paths[0]), id=str(paths[1]))  # default sampling="pair": batch_size counts pairs
    batch = next(iter(data.train_loader(3, 1, 0, 0, 0, 1, 0, False)))
    assert tuple(batch["x"].shape) == (3, 10, 10) and tuple(batch["y"].shape) == (3, 10, 10)
    episode_data = GridworldPairs(train=str(paths[0]), id=str(paths[1]), sampling="episode")
    batch = next(iter(episode_data.train_loader(3, 1, 0, 0, 0, 1, 0, False)))
    assert tuple(batch["x"].shape) == (6, 10, 10) and tuple(batch["y"].shape) == (6, 10, 10)
    episodes = next(iter(data.eval_loader("id", 2)))
    assert tuple(episodes["query_y"].shape) == (2, 10, 10)
