"""BabyTheorist reader and evaluations; every config under configs/babytheorist builds, takes one step and evaluates on CPU."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from components.cnn_any_size import CNNVAEAnyDecoder, CNNVAEAnyEncoder
from components.decoders import CNNVAEDecoder
from components.encoders import CNNVAEEncoder
from core.config import load_config
from core.registry import METHODS, build, import_all
from data.babytheorist import BabyTheoristPairs
from training.trainer import run_evals

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs" / "babytheorist").glob("*.yaml"))
PRIMITIVES = ("rot90", "flip_h", "translate_up")


def write_files(directory: Path, tasks: int = 6, grid: int = 15) -> tuple[Path, Path]:
    """Tiny files in the benchmark's layout: random grids, two or three supports per eval row, a program string per row."""

    rng = np.random.default_rng(0)
    program = lambda depth: json.dumps([{"selector": {"type": "by_index", "param": 1}, "primitive": PRIMITIVES[i % 3], "target": None} for i in range(depth)])  # noqa: E731
    depths = np.array([1, 1, 2, 3, 5, 7][:tasks], dtype=np.int32)
    train, evaluation = directory / "train.h5", directory / "eval.h5"
    with h5py.File(train, "w") as f:
        f["inputs"] = rng.integers(0, 10, (tasks, 1, grid, grid), dtype=np.int8)
        f["outputs"] = rng.integers(0, 10, (tasks, 1, grid, grid), dtype=np.int8)
        f["depths"] = depths
        f.create_dataset("transformation_suites", data=[program(d) for d in depths], dtype=h5py.string_dtype())
    with h5py.File(evaluation, "w") as f:
        f["inputs"] = rng.integers(0, 10, (tasks, 4, grid, grid), dtype=np.int8)
        f["outputs"] = rng.integers(0, 10, (tasks, 4, grid, grid), dtype=np.int8)
        f["depths"] = depths
        f["n_support"] = np.array([2, 3, 2, 3, 2, 3][:tasks], dtype=np.int32)
        f.create_dataset("transformation_suites", data=[program(d) for d in depths], dtype=h5py.string_dtype())
    return train, evaluation


def test_reader_puts_the_query_after_each_rows_supports(tmp_path: Path) -> None:
    train, evaluation = write_files(tmp_path)
    data = BabyTheoristPairs(str(train), {"lv": str(evaluation)})
    batch = next(iter(data.train_loader(4, 1, 0, 0, 0, 1, 0, False)))
    assert batch["x"].shape == (4, 15, 15) and batch["x"].dtype == torch.long
    rows = next(iter(data.eval_loader("lv", 6)))
    with h5py.File(evaluation, "r") as f:
        inputs, outputs = f["inputs"][:], f["outputs"][:]
    for row, k in enumerate(rows["n_support"].tolist()):
        assert torch.equal(rows["query_x"][row], torch.from_numpy(inputs[row, k]).long())
        assert torch.equal(rows["query_y"][row], torch.from_numpy(outputs[row, k]).long())
    assert rows["supports_x"].shape[1] == 3 and rows["depth"].tolist() == [1, 1, 2, 3, 5, 7]


def test_any_size_cnn_is_the_official_cnn_at_ten() -> None:
    """Same layers and shapes as the lab's parts at 10x10: the official weights load and give the same output."""

    torch.manual_seed(0)
    official_encoder, official_decoder = CNNVAEEncoder(10, 9, 32), CNNVAEDecoder(10, 9, 32)
    encoder, decoder = CNNVAEAnyEncoder(10, 9, 32), CNNVAEAnyDecoder(10, 9, 32)
    encoder.load_state_dict(official_encoder.state_dict())
    decoder.load_state_dict(official_decoder.state_dict())
    for module in (official_encoder, official_decoder, encoder, decoder):
        module.eval()
    grid = torch.randint(0, 9, (3, 10, 10))
    assert torch.equal(encoder(grid), official_encoder(grid))
    assert torch.equal(decoder(encoder(grid)), official_decoder(official_encoder(grid)))
    assert CNNVAEAnyDecoder(15, 10, 32)(CNNVAEAnyEncoder(15, 10, 32)(torch.randint(0, 10, (2, 15, 15)))).shape == (2, 15, 15, 10)


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_config_builds_steps_and_evaluates(path: Path, tmp_path: Path) -> None:
    import_all()
    torch.manual_seed(0)
    config = load_config(path)
    train, evaluation = write_files(tmp_path)
    data = BabyTheoristPairs(str(train), {split: str(evaluation) for split in config["data"]["evals"]})
    method = build(METHODS, config["method"])
    model = method.build_model(config["model"], init_from=None)
    method.attach(model, torch.device("cpu"), "fp32")
    method.set_train_mode(model)
    metrics = method.train_step(model, next(iter(data.train_loader(4, 1, 0, 0, 0, 1, 0, False))), 0)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values()), metrics
    blocks = [{**b, "batch_size": 3, "max_samples": 6} for b in config["evals"]]
    results = run_evals(model, data, blocks, 1, torch.device("cpu"))
    for block in blocks:
        prefix = f"{block.get('name', block['type'])}/{block['split']}"
        if block["type"] == "babytheorist_observe":
            assert results[f"{prefix}/total"] == 6 and 0.0 <= results[f"{prefix}/transfer_consistent"] <= 1.0
            assert sum(results[f"{prefix}/tasks_depth{d}"] for d in (1, 2, 3, 4, 5)) == 6
        else:
            assert f"{prefix}/exact" in results
