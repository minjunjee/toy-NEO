"""Every run config must build, take one train step, and run each of its evals on CPU."""

from pathlib import Path

import pytest
import torch

from core.config import load_config
from training.trainer import run_evals
from core.registry import METHODS, build, import_all
from tests.synthetic import SyntheticData

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs" / "runs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_config_builds_steps_and_evaluates(path: Path) -> None:
    import_all()
    torch.manual_seed(0)
    config = load_config(path)
    method = build(METHODS, config["method"])
    model = method.build_model(config["model"], init_from=None)
    method.attach(model, torch.device("cpu"), "fp32")
    data = SyntheticData(n=4)
    method.set_train_mode(model)
    batch = next(iter(data.train_loader(4, 1, 0, 0, 0, 1, 0, False)))
    metrics = method.train_step(model, batch, 0)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values()), metrics
    blocks = [{**b, "batch_size": 4, "max_samples": 4} for b in config["evals"]]
    for block in blocks:
        block.pop("group_by_program", None)
    results = run_evals(model, data, blocks, 1, torch.device("cpu"))
    for block in blocks:
        assert f"{block.get('name', block['type'])}/{block['split']}/exact" in results


def test_unknown_config_key_is_rejected() -> None:
    import_all()
    with pytest.raises(KeyError):
        build(METHODS, {"type": "neo", "lr": 1e-3, "not_a_key": 1})
