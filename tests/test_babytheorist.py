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
    circle = OrbitVQ(3, amounts, 16, family="circle", angle=0.4).table().view(3, 5, 16)
    # a fixed angle per unit of amount: neighbouring amounts are 0.4 rad apart, whatever the learned plane
    assert torch.allclose((circle[:, 2] * circle[:, 3]).sum(dim=-1), torch.cos(torch.tensor(0.4)).expand(3), atol=1e-5)
    assert torch.allclose((circle[:, 0] * circle[:, 4]).sum(dim=-1), torch.cos(torch.tensor(2.0)).expand(3), atol=1e-5)
    for family in ("rotation", "circle", "line"):
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


def test_counted_model_applies_the_unit_of_the_picked_kind_as_often_as_the_picked_amount_says() -> None:
    from core.registry import MODELS

    import_all()
    torch.manual_seed(0)
    spec = {
        "encoder": {"type": "token_embed", "vocab": 10, "d_model": 32}, "decoder": {"type": "linear_head", "d_model": 32, "vocab": 10, "layer_norm_eps": 1.0e-6, "spatial": [6, 6]},
        "programmer": {"type": "token_pair_transformer", "state_dim": 32, "d_model": 32, "action_dim": 8, "num_heads": 4, "d_ff": 64, "num_layers": 1, "grid_dim": 6, "dropout": 0.0, "layer_norm_eps": 1.0e-6},
        "code": {"type": "vq_orbit", "kinds": 2, "amounts": [1, 2, 3], "family": "rotation", "dim": 8, "num_tokens": 1, "commitment": 0.25, "stochastic": False, "commitment_grad": True},
        "executor": {"type": "adaln_transformer", "state_dim": 32, "cond_dim": 8, "d_model": 32, "num_layers": 1, "num_heads": 4, "d_ff": 64, "dropout": 0.0, "grid_dim": 6},
        "max_length": 3, "length_coefficient": 0.95,
    }
    model = MODELS["neo_counted"](**spec)
    x, y = torch.randint(0, 10, (7, 6, 6)), torch.randint(0, 10, (7, 6, 6))
    out = model({"x": x, "y": y})
    out.code_loss_by_length[0].mean().backward()
    assert model.code.generator.grad is not None and bool(model.code.generator.grad.abs().sum() > 0)
    model.eval()
    explained = model.self_explain(x, y, max_steps=3, rollout="hard")
    index = explained["indices"][0].view(7, -1)[:, 0]
    kind, amount = model.code.kind_and_amount(index)
    assert torch.equal(explained["n_steps"], amount)      # the count is the one the programmer named, not one chosen by looking at y
    unit = model.code.table()[kind * 3][:, None]          # the entry of amount 1 of the picked kind
    grid, by_count = x, []
    with torch.no_grad():
        for _ in range(3):
            grid = model.decoder(model.executor(model.encoder(grid), unit)).argmax(dim=-1)
            by_count.append(grid)
    assert torch.equal(explained["pred"], torch.stack(by_count, dim=1)[torch.arange(7), amount - 1])
    assert torch.equal(model.infer(x, y, x, max_steps=3, rollout="hard")["pred"], explained["pred"])
    model.count_from = "rule"
    assert model.self_explain(x, y, max_steps=3, rollout="hard")["n_steps"].shape == (7,)
    with pytest.raises(ValueError):
        MODELS["neo_counted"](**{**spec, "max_length": 4})


def test_per_code_executor_runs_only_the_network_of_the_code_it_is_given() -> None:
    from core.registry import MODELS

    import_all()
    torch.manual_seed(0)
    spec = {
        "encoder": {"type": "token_embed", "vocab": 10, "d_model": 32}, "decoder": {"type": "linear_head", "d_model": 32, "vocab": 10, "layer_norm_eps": 1.0e-6, "spatial": [6, 6]},
        "programmer": {"type": "token_pair_transformer", "state_dim": 32, "d_model": 32, "action_dim": 8, "num_heads": 4, "d_ff": 64, "num_layers": 1, "grid_dim": 6, "dropout": 0.0, "layer_norm_eps": 1.0e-6},
        "code": {"type": "vq", "size": 4, "dim": 8, "num_tokens": 1, "commitment": 0.25, "stochastic": False, "commitment_grad": True},
        "executor": {"type": "per_code", "state_dim": 32, "cond_dim": 8, "size": 4, "d_model": 16, "num_layers": 1, "num_heads": 4, "d_ff": 32, "grid_dim": 6},
        "max_length": 1, "length_coefficient": 0.95,
    }
    model = MODELS["neo_expert"](**spec)
    for expert in model.executor.experts:   # AdaLN-Zero starts as the identity: make each network do something different
        for p in expert.parameters():
            torch.nn.init.normal_(p, std=0.2)
    x = torch.randint(0, 10, (6, 6, 6))
    state = model.encoder(x)
    table = model.code.codebook(torch.arange(4))
    picks = torch.tensor([0, 2, 2, 3, 1, 0])
    out = model.executor(state, table[picks][:, None])
    for row, code in enumerate(picks.tolist()):
        alone = model.executor.experts[code](state[row:row + 1], table[code][None, None])
        assert torch.allclose(out[row], alone[0], atol=1e-5)
    out[1].sum().backward()   # row 1 ran code 2: only network 2 gets a gradient
    touched = [any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in e.parameters()) for e in model.executor.experts]
    assert touched == [False, False, True, False]


def test_lora_code_executor_shares_the_up_projection_and_keeps_each_codes_own_down_projection() -> None:
    from core.registry import MODELS

    import_all()
    torch.manual_seed(0)
    spec = {
        "encoder": {"type": "token_embed", "vocab": 10, "d_model": 32}, "decoder": {"type": "linear_head", "d_model": 32, "vocab": 10, "layer_norm_eps": 1.0e-6, "spatial": [6, 6]},
        "programmer": {"type": "token_pair_transformer", "state_dim": 32, "d_model": 32, "action_dim": 8, "num_heads": 4, "d_ff": 64, "num_layers": 1, "grid_dim": 6, "dropout": 0.0, "layer_norm_eps": 1.0e-6},
        "code": {"type": "vq", "size": 4, "dim": 8, "num_tokens": 1, "commitment": 0.25, "stochastic": False, "commitment_grad": True},
        "executor": {"type": "lora_code", "state_dim": 32, "cond_dim": 8, "size": 4, "d_model": 16, "num_layers": 1, "num_heads": 4, "d_ff": 32, "grid_dim": 6, "rank": 2,
                     "shared_speed": 0.1, "slow_base": True},
        "max_length": 1, "length_coefficient": 0.95,
    }
    model = MODELS["neo_expert"](**spec)
    layer = model.executor.stack.blocks[0].ffn[0]
    torch.nn.init.normal_(model.executor.stack.blocks[0].adaLN_modulation[-1].weight, std=0.3)   # open the block (AdaLN-Zero starts closed)
    torch.nn.init.normal_(layer.up, std=0.3)   # the shared up-projection starts at zero, as in LoRA; give it a value so the down-projections get a gradient
    state = model.encoder(torch.randint(0, 10, (3, 6, 6)))
    table = model.code.codebook(torch.arange(4))
    out = model.executor(state, table[torch.tensor([1, 1, 3])][:, None])
    out[0].sum().backward()   # row 0 ran code 1: the shared up-projection and code 1's down-projection learn, no other code's does
    assert layer.up.grad is not None and bool(layer.up.grad.abs().sum() > 0)
    moved = [bool(layer.down.grad[k].abs().sum() > 0) for k in range(4)]
    assert moved == [False, True, False, False]


def test_competing_codes_train_only_the_winner_and_pull_the_programmer_to_it() -> None:
    from core.registry import MODELS

    import_all()
    torch.manual_seed(0)
    spec = {
        "encoder": {"type": "token_embed", "vocab": 10, "d_model": 32}, "decoder": {"type": "linear_head", "d_model": 32, "vocab": 10, "layer_norm_eps": 1.0e-6, "spatial": [6, 6]},
        "programmer": {"type": "token_pair_transformer", "state_dim": 32, "d_model": 32, "action_dim": 8, "num_heads": 4, "d_ff": 64, "num_layers": 1, "grid_dim": 6, "dropout": 0.0, "layer_norm_eps": 1.0e-6},
        "code": {"type": "vq", "size": 5, "dim": 8, "num_tokens": 1, "commitment": 0.25, "stochastic": False, "commitment_grad": True},
        "executor": {"type": "per_code", "state_dim": 32, "cond_dim": 8, "size": 5, "d_model": 16, "num_layers": 1, "num_heads": 4, "d_ff": 32, "grid_dim": 6},
        "max_length": 1, "length_coefficient": 0.95, "candidates": 3,
    }
    model = MODELS["neo_compete"](**spec)
    for expert in model.executor.experts:
        for p in expert.parameters():
            torch.nn.init.normal_(p, std=0.2)
    x, y = torch.randint(0, 10, (4, 6, 6)), torch.randint(0, 10, (4, 6, 6))
    out = model({"x": x, "y": y})
    winners = out.code_indices_by_length[0]
    # the winner is, among the 3 nearest codes, the one whose executed grid is closest to y
    state, target = model.encoder(x), model.encoder(y)
    table = model.code.codebook(torch.arange(5))
    near = (torch.nn.functional.normalize(model.programmer(state, target).flatten(1), dim=-1) @ table.T).topk(3, dim=1).indices
    for b in range(4):
        losses = [torch.nn.functional.cross_entropy(model.decoder(model.executor(state[b:b + 1], table[c][None, None])).reshape(-1, 10), y[b].reshape(-1)) for c in near[b].tolist()]
        assert int(winners[b]) == near[b][int(torch.stack(losses).argmin())]
    torch.nn.functional.cross_entropy(out.logits_by_length[0].reshape(-1, 10), y.reshape(-1)).backward()
    touched = {k for k, e in enumerate(model.executor.experts) if any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in e.parameters())}
    assert touched == set(winners.tolist())


def test_soft_em_weights_codes_by_how_well_they_reproduce_and_alternate_freezes_one_side() -> None:
    from core.registry import METHODS, MODELS

    import_all()
    torch.manual_seed(0)
    spec = {
        "encoder": {"type": "token_embed", "vocab": 10, "d_model": 32}, "decoder": {"type": "linear_head", "d_model": 32, "vocab": 10, "layer_norm_eps": 1.0e-6, "spatial": [6, 6]},
        "programmer": {"type": "token_pair_transformer", "state_dim": 32, "d_model": 32, "action_dim": 8, "num_heads": 4, "d_ff": 64, "num_layers": 1, "grid_dim": 6, "dropout": 0.0, "layer_norm_eps": 1.0e-6},
        "code": {"type": "vq", "size": 5, "dim": 8, "num_tokens": 1, "commitment": 0.25, "stochastic": False, "commitment_grad": True},
        "executor": {"type": "adaln_transformer", "state_dim": 32, "cond_dim": 8, "d_model": 32, "num_layers": 1, "num_heads": 4, "d_ff": 64, "dropout": 0.0, "grid_dim": 6},
        "max_length": 1, "length_coefficient": 0.95,
    }
    model = MODELS["neo_em"](**spec, train_top=2, temperature_start=5.0, temperature_end=5.0)
    torch.nn.init.normal_(model.executor.stack.blocks[0].adaLN_modulation[-1].weight, std=0.3)
    x, y = torch.randint(0, 10, (3, 6, 6)), torch.randint(0, 10, (3, 6, 6))
    out = model({"x": x, "y": y}, step=0)
    # the responsibilities are a softmax of the summed cell cross-entropy of every code, at the temperature
    state = model.encoder(x)
    table = model.code.codebook(torch.arange(5))
    with torch.no_grad():
        losses = torch.stack([torch.nn.functional.cross_entropy(model.decoder(model.executor(state, table[k].expand(3, 1, -1))).reshape(3, -1, 10).transpose(1, 2), y.reshape(3, -1),
                                                                 reduction="none").sum(dim=1) for k in range(5)], dim=1)
    expected = torch.softmax(-losses / 5.0, dim=1)
    assert torch.equal(out.code_indices_by_length[0], expected.argmax(dim=1))
    out.code_loss_by_length[0].mean().backward()
    assert model.code.embedding.weight.grad is not None and any(p.grad is not None for p in model.programmer.parameters())
    # alternating: during the executor's turn the programmer does not move
    model = MODELS["neo"](**spec)
    method = METHODS["neo_alternate"](lr=1e-2, programmer_lr_scale=1.0, executor_lr_scale=1.0, weight_decay=0.0, warmup_ratio=0.0, min_lr_ratio=1.0, grad_clip=1.0,
                                      grounding_weight=0.0, code_weight=1.0, auto_weight=0.0, total_steps=10, period=1)
    method.attach(model, torch.device("cpu"), "fp32")
    before = [p.detach().clone() for p in model.programmer.parameters()]
    executor_before = [p.detach().clone() for p in model.executor.parameters()]
    method.train_step(model, {"x": x, "y": y}, step=1)   # step 1 with period 1: the executor's turn
    assert all(torch.equal(a, b) for a, b in zip(before, model.programmer.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(executor_before, model.executor.parameters()))


def test_hard_em_rescues_only_dead_codes_up_to_a_cap() -> None:
    from core.registry import MODELS

    import_all()
    torch.manual_seed(0)
    spec = {
        "encoder": {"type": "token_embed", "vocab": 10, "d_model": 32}, "decoder": {"type": "linear_head", "d_model": 32, "vocab": 10, "layer_norm_eps": 1.0e-6, "spatial": [6, 6]},
        "programmer": {"type": "token_pair_transformer", "state_dim": 32, "d_model": 32, "action_dim": 8, "num_heads": 4, "d_ff": 64, "num_layers": 1, "grid_dim": 6, "dropout": 0.0, "layer_norm_eps": 1.0e-6},
        "code": {"type": "vq", "size": 5, "dim": 8, "num_tokens": 1, "commitment": 0.25, "stochastic": False, "commitment_grad": True},
        "executor": {"type": "adaln_transformer", "state_dim": 32, "cond_dim": 8, "d_model": 32, "num_layers": 1, "num_heads": 4, "d_ff": 64, "dropout": 0.0, "grid_dim": 6},
        "max_length": 1, "length_coefficient": 0.95,
    }
    model = MODELS["neo_em"](**spec, train_top=1, temperature_start=0.01, temperature_end=0.01, dead_share=1.0, bonus_step=0.05, bonus_cap=0.5)
    torch.nn.init.normal_(model.executor.stack.blocks[0].adaLN_modulation[-1].weight, std=0.3)
    x, y = torch.randint(0, 10, (3, 6, 6)), torch.randint(0, 10, (3, 6, 6))
    model.usage.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    model.bonus.copy_(torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0]))
    model({"x": x, "y": y}, step=0)
    # code 0 is in use and gets no bonus; the dead codes gain one step, and code 2 stays at the cap
    assert model.bonus[0] == 0.0 and torch.allclose(model.bonus[[1, 3, 4]], torch.full((3,), 0.05)) and model.bonus[2] == 0.5
    # a dead code with a large enough bonus takes every pair, and once it is in use its bonus returns to zero
    model.bonus.copy_(torch.tensor([0.0, 0.0, 0.0, 1.0e6, 0.0]))
    model.usage_decay = 0.0
    out = model({"x": x, "y": y}, step=0)
    assert torch.equal(out.code_indices_by_length[0], torch.full((3,), 3))
    assert model.bonus[3] == 0.0


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_config_builds_steps_and_evaluates(path: Path, tmp_path: Path) -> None:
    import_all()
    torch.manual_seed(0)
    config = load_config(path)
    train, evaluation = write_files(tmp_path)
    from core.registry import DATASETS

    data = DATASETS[config["data"]["type"]](str(train), {split: str(evaluation) for split in config["data"]["evals"]})
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


def test_code_nmi_and_entropy():
    from evals.babytheorist import normalized_mutual_information, usage_entropy

    # every primitive has a code of its own: NMI 1; two primitives on one code: lower; one primitive spread over two codes: lower
    assert abs(normalized_mutual_information(torch.tensor([[5, 0, 0], [0, 7, 0], [0, 0, 3]])) - 1.0) < 1e-9
    shared = normalized_mutual_information(torch.tensor([[5, 0], [7, 0], [0, 3]]))
    spread = normalized_mutual_information(torch.tensor([[5, 5, 0], [0, 0, 7]]))
    assert 0.0 < shared < 1.0 and 0.0 < spread < 1.0
    assert normalized_mutual_information(torch.tensor([[4], [6]])) == 0.0
    assert abs(usage_entropy(torch.tensor([3, 3, 3, 3])) - 1.0) < 1e-9
    assert usage_entropy(torch.tensor([9, 0, 0, 0])) == 0.0
