"""Transformer NEO (neo_transformer_gw033): prefix code loss and the MDL carry/hard evaluation rollouts."""

from pathlib import Path

import pytest
import torch

from core.config import load_config
from core.registry import METHODS, build, import_all
from tests.synthetic import SyntheticData

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "runs" / "neo_transformer_gw033_s42.yaml"


def build_run():
    import_all()
    torch.manual_seed(0)
    config = load_config(CONFIG)
    method = build(METHODS, config["method"])
    model = method.build_model(config["model"], init_from=None)
    method.attach(model, torch.device("cpu"), "fp32")
    return method, model, SyntheticData(n=6)


class Replay:
    """Stands in for a DDP-wrapped model whose forward output is fixed."""

    def __init__(self, module, out):
        self.module, self.out = module, out

    def __call__(self, batch, step):
        return self.out


def test_prefix_code_loss_averages_every_step_up_to_the_selected_length():
    method, model, data = build_run()
    batch = next(iter(data.train_loader(6, 1, 0, 0, 0, 1, 0, False)))
    out = model(batch)
    out.selected_lengths = torch.tensor([1, 2, 3, 4] * 3)
    code = method.compute_loss(Replay(model, out), batch, 0).metrics["code"]
    losses = torch.stack(out.code_loss_by_length, dim=1)
    expected = torch.cat([losses[b, :length] for b, length in enumerate(out.selected_lengths.tolist())]).mean()
    torch.testing.assert_close(code, expected)


@pytest.mark.parametrize("rollout", ["carry", "hard"])
def test_transfer_onto_the_support_itself_reproduces_self_explanation(rollout):
    _, model, data = build_run()
    model.eval()
    batch = next(iter(data.eval_loader("id", 6)))
    sx, sy = batch["support_x"], batch["support_y"]
    explained = model.self_explain(sx, sy, max_steps=4, rollout=rollout)
    transferred = model.infer(sx, sy, sx, max_steps=4, rollout=rollout)
    assert torch.equal(transferred["pred"], explained["pred"])
    assert torch.equal(transferred["n_applied"], explained["n_steps"])
    with pytest.raises(ValueError):
        model.infer(sx, sy, sx, rollout="soft")


def test_length_cost_ramps_in_and_trades_cells_for_codes():
    from models.neo import mdl_select

    _, model, _ = build_run()
    model.length_cost, model.length_cost_ramp = 3.0, (1955, 11730)  # the baseline config uses the official rule instead
    for step, cost in [(0, 0.0), (1955, 0.0), ((1955 + 11730) // 2, 1.5), (11730, 3.0), (78200, 3.0)]:
        model.set_length_cost(step)
        assert abs(model.length_cost_now - cost) < 1e-3
    y = torch.zeros(1, 2, 2, dtype=torch.long)
    exact = torch.full((1, 2, 2, 3), -20.0)
    exact[..., 0] = 20.0
    near = exact.clone()
    near[0, 0, 0] = torch.tensor([0.0, 1.0, -20.0])  # one cell wrong, cross entropy about 1.3 nats
    # the official rule takes the exact rollout; a length cost of 3 nats per code prefers one code and a missed cell
    assert mdl_select([near, exact], y, 1.0).tolist() == [2]
    assert mdl_select([near, exact], y, 1.0, length_cost=3.0).tolist() == [1]
    assert mdl_select([near, exact], y, 1.0, length_cost=1.0).tolist() == [2]
