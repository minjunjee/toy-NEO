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


def test_growable_codebook_is_the_lab_codebook_when_every_row_is_active_and_never_picks_an_inactive_row() -> None:
    from components.codes import VQ
    from components.codes_growable import GrowableVQ

    torch.manual_seed(0)
    lab = VQ(8, 16, stochastic=False, commitment_grad=True)
    grown = GrowableVQ(8, 16, stochastic=False, commitment_grad=True)
    grown.embedding.load_state_dict(lab.embedding.state_dict())
    actions = torch.randn(64, 1, 16)
    a, b = lab(actions, training=True), grown(actions, training=True)
    assert torch.equal(a.indices, b.indices) and torch.equal(a.values, b.values) and torch.equal(a.loss_per_sample, b.loss_per_sample)
    partial = GrowableVQ(8, 16, initial=3, stochastic=False, commitment_grad=True)
    assert int(partial(actions, training=False).indices.max()) <= 2 and partial.active.tolist() == [True] * 3 + [False] * 5
    partial.active[6] = True
    assert set(partial(actions, training=False).indices.tolist()) <= {0, 1, 2, 6}


def test_visible_state_model_computes_what_reencode_hard_computes_and_lets_the_gradient_through() -> None:
    """Same weights: the same decoded grids as the lab NEO with reencode_hard at every length. The second step's output
    has a gradient with respect to the first step's output state, which reencode_hard cuts."""

    from core.registry import MODELS

    import_all()
    config = load_config(next(p for p in CONFIGS if p.stem == "neo_transformer_36k_codes8_visible_single_a_lv1_s42"))
    spec = {**config["model"], "max_length": 2}
    torch.manual_seed(0)
    visible = build(MODELS, spec)
    lab = build(MODELS, {**spec, "type": "neo", "state_update": "reencode_hard"})
    lab.load_state_dict(visible.state_dict())
    batch = {"x": torch.randint(0, 10, (2, 15, 15)), "y": torch.randint(0, 10, (2, 15, 15))}
    reached = []
    for model in (visible, lab):
        model.eval()
        outputs = []

        def keep(module, args, output, outputs=outputs) -> None:
            output.retain_grad()
            outputs.append(output)

        handle = model.executor.register_forward_hook(keep)
        out = model(batch)
        handle.remove()
        out.logits_by_length[1].sum().backward()
        reached.append(outputs[0].grad is not None and bool(outputs[0].grad.abs().sum() > 0))
        model.logits = [logits.detach() for logits in out.logits_by_length]
    for mine, theirs in zip(visible.logits, lab.logits):
        assert torch.allclose(mine, theirs, atol=1e-5)
    assert reached == [True, False]


def test_kind_and_amount_codebook_is_the_lab_codebook_when_free_and_ties_the_codes_of_one_kind_by_one_rotation() -> None:
    from components.codes import VQ
    from components.codes_orbit import OrbitVQ

    torch.manual_seed(0)
    lab = VQ(8, 16, stochastic=False, commitment_grad=True)
    free = OrbitVQ(2, [1, 2, 3, 4], 16, family="free", stochastic=False, commitment_grad=True)
    free.embedding.load_state_dict(lab.embedding.state_dict())
    actions = torch.randn(64, 1, 16)
    a, b = lab(actions, training=True), free(actions, training=True)
    assert torch.equal(a.indices, b.indices) and torch.allclose(a.values, b.values) and torch.allclose(a.loss_per_sample, b.loss_per_sample)

    amounts = [-2, -1, 1, 2, 3]
    for family in ("rotation", "line"):
        codes = OrbitVQ(3, amounts, 16, family=family, stochastic=False, commitment_grad=True)
        table = codes.table()
        assert table.shape == (15, 16) and torch.allclose(table.norm(dim=-1), torch.ones(15), atol=1e-5)
        out = codes(actions, training=True)
        out.loss_per_sample.mean().backward()
        assert all(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in codes.parameters())
        assert torch.allclose(out.values, table[out.indices].view(actions.shape), atol=1e-5)
    # the index table: code (g, a) then code (g, b) is code (g, a + b); nothing across kinds or outside the list
    for i in range(15):
        for j in range(15):
            (gi, ai), (gj, aj) = codes.kind_and_amount(torch.tensor(i)), codes.kind_and_amount(torch.tensor(j))
            k = int(codes.sums[i, j])
            if gi == gj and int(ai + aj) in amounts:
                gk, ak = codes.kind_and_amount(torch.tensor(k))
                assert gk == gi and ak == ai + aj
            else:
                assert k == -1
    # rotation: the code of amount a + b is the code of amount b turned by exp(a * A) of its kind
    codes = OrbitVQ(3, amounts, 16, family="rotation")
    table = codes.table().view(3, 5, 16)
    skew = (codes.generator - codes.generator.transpose(1, 2)).detach()
    for g in range(3):
        turned = torch.linalg.matrix_exp(2.0 * skew[g]) @ table[g, amounts.index(1)]
        assert torch.allclose(turned, table[g, amounts.index(3)], atol=1e-5)
        assert torch.allclose(torch.linalg.matrix_exp(1.0 * skew[g]) @ table[g, amounts.index(-1)], torch.nn.functional.normalize(codes.base[g], dim=-1), atol=1e-5)


def test_repeat_model_applies_one_code_a_counted_number_of_times() -> None:
    from core.registry import MODELS

    import_all()
    torch.manual_seed(0)
    spec = {
        "encoder": {"type": "token_embed", "vocab": 10, "d_model": 32}, "decoder": {"type": "linear_head", "d_model": 32, "vocab": 10, "layer_norm_eps": 1.0e-6, "spatial": [6, 6]},
        "programmer": {"type": "token_pair_transformer", "state_dim": 32, "d_model": 32, "action_dim": 8, "num_heads": 4, "d_ff": 64, "num_layers": 1, "grid_dim": 6, "dropout": 0.0, "layer_norm_eps": 1.0e-6},
        "code": {"type": "vq", "size": 4, "dim": 8, "num_tokens": 1, "commitment": 0.25, "stochastic": False, "commitment_grad": True},
        "executor": {"type": "adaln_transformer", "state_dim": 32, "cond_dim": 8, "d_model": 32, "num_layers": 1, "num_heads": 4, "d_ff": 64, "dropout": 0.0, "grid_dim": 6},
        "max_length": 3, "length_coefficient": 0.95,
    }
    model = MODELS["neo_repeat"](**spec)
    x, y = torch.randint(0, 10, (5, 6, 6)), torch.randint(0, 10, (5, 6, 6))
    outputs = []

    def keep(module, inputs, output) -> None:
        output.retain_grad()
        outputs.append(output)

    handle = model.executor.register_forward_hook(keep)
    out = model({"x": x, "y": y})
    handle.remove()
    assert len(out.logits_by_length) == 3 and all(torch.equal(i, out.code_indices_by_length[0]) for i in out.code_indices_by_length)
    out.logits_by_length[2].sum().backward()
    # the loss after the third application reaches the first application through the grids handed over
    assert len(outputs) == 3 and outputs[0].grad is not None and bool(outputs[0].grad.abs().sum() > 0)
    model.eval()
    explained = model.self_explain(x, y, max_steps=3, rollout="hard")
    # the explanation is the picked code applied n_steps times by hand, the grid handed between the applications
    code = model.code.codebook(explained["indices"][0].view(5, -1)[:, 0])[:, None]
    grid, by_count = x, []
    with torch.no_grad():
        for _ in range(3):
            grid = model.decoder(model.executor(model.encoder(grid), code)).argmax(dim=-1)
            by_count.append(grid)
    assert torch.equal(explained["pred"], torch.stack(by_count, dim=1)[torch.arange(5), explained["n_steps"] - 1])
    answered = model.infer(x, y, x, max_steps=3, rollout="hard")
    assert torch.equal(answered["pred"], explained["pred"])
    with pytest.raises(ValueError):
        model.self_explain(x, y, max_steps=3, rollout="official")


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
