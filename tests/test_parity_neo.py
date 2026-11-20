"""Our NEO rollout must reproduce the official model given the official weights."""

from pathlib import Path

import pytest
import torch

from core.config import load_config
from core.registry import MODELS, build, import_all

FIXTURE = Path(__file__).with_name("fixtures") / "neo_official.pt"
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "runs" / "neo_gw033_k4_cb6_s42.yaml"
RENAME = {"theory_programmer.": "programmer.", "program_executor.": "executor.", "quantizer.": "code."}


def map_keys(state: dict) -> dict:
    mapped = {}
    for key, value in state.items():
        for old, new in RENAME.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        mapped[key] = value
    return mapped


@pytest.mark.skipif(not FIXTURE.exists(), reason="run tests/fixtures/make_neo_fixture.py first")
def test_neo_matches_official() -> None:
    import_all()
    fixture = torch.load(FIXTURE, map_location="cpu", weights_only=False)
    model = build(MODELS, load_config(CONFIG)["model"]).eval()
    model.load_state_dict(map_keys(fixture["state_dict"]), strict=True)
    episodes = fixture["episodes"]
    x, y = episodes[:, 0], episodes[:, 1]
    with torch.no_grad():
        state = model.encoder(x)
        torch.testing.assert_close(state, fixture["state"])
        action = model.programmer(state, model.encoder(y))
        torch.testing.assert_close(action, fixture["action"])
        code = model.code(action, training=False)
        assert torch.equal(code.indices, fixture["indices"])
        torch.testing.assert_close(code.values, fixture["values"])
        next_state = model.executor(state, code.values)
        torch.testing.assert_close(next_state, fixture["next_state"])
        torch.testing.assert_close(model.decoder(next_state), fixture["logits"])
        transfer = model.infer(episodes[:, 0], episodes[:, 1], episodes[:, 2], episodes[:, 3], max_steps=4)
        assert torch.equal(transfer["pred"], fixture["transfer_pred"])
        assert torch.equal(transfer["n_extracted"], fixture["transfer_extracted"])
        assert torch.equal(transfer["n_applied"], fixture["transfer_applied"])
        selfx = model.self_explain(episodes[:, 0], episodes[:, 1], max_steps=4)
        assert torch.equal(selfx["pred"], fixture["self_pred"])
        assert torch.equal(selfx["n_steps"], fixture["self_steps"])
