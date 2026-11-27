"""ARC task data, model and evaluation; every config under configs/arc builds, takes one step and evaluates on CPU."""

from pathlib import Path

import numpy as np
import pytest
import torch

from core.config import load_config
from core.registry import DATASETS, METHODS, build, import_all
from data.arc_tasks import PAD, color_tables, transform
from evals.arc import crop, undo, view
from training.trainer import run_evals

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs" / "arc").glob("*.yaml"))


def canvas(grid: np.ndarray, size: int) -> np.ndarray:
    out = np.full((size, size), PAD, dtype=np.uint8)
    out[: grid.shape[0], : grid.shape[1]] = grid
    return out


def write_files(folder: Path, size: int, tasks: int = 3, examples: int = 6) -> tuple[Path, Path]:
    """Tasks whose rule is "add the task number to every colour, modulo 10", on grids of random shapes."""

    rng = np.random.default_rng(0)
    inputs, outputs, shapes, task = [], [], [], []
    demo = {k: [] for k in ("demo_inputs", "demo_outputs", "demo_task", "test_inputs", "test_outputs", "test_task")}
    for t in range(tasks):
        for e in range(examples):
            h, w = rng.integers(2, size + 1, size=2)
            x = rng.integers(0, 10, size=(h, w)).astype(np.uint8)
            y = ((x + t + 1) % 10).astype(np.uint8)
            if e < examples - 3:
                inputs.append(canvas(x, size)); outputs.append(canvas(y, size)); shapes.append((h, w, h, w)); task.append(t)
            else:
                kind = "demo" if e < examples - 1 else "test"
                demo[f"{kind}_inputs"].append(canvas(x, size)); demo[f"{kind}_outputs"].append(canvas(y, size)); demo[f"{kind}_task"].append(t)
    names = np.asarray([f"task{t}" for t in range(tasks)])
    train, evaluation = folder / "train.npz", folder / "tasks.npz"
    np.savez(train, inputs=np.asarray(inputs), outputs=np.asarray(outputs), shapes=np.asarray(shapes, dtype=np.uint8), task=np.asarray(task, dtype=np.int32), names=names)
    np.savez(evaluation, **{k: np.asarray(v) for k, v in demo.items()}, names=names, train_index=np.arange(tasks))
    return train, evaluation


def test_a_view_is_undone_exactly_and_bad_shapes_are_dropped() -> None:
    rng = np.random.default_rng(1)
    tables = color_tables(4, seed=0)
    assert (tables[0] == np.arange(11)).all() and all(sorted(t) == list(range(11)) and t[0] == 0 and t[10] == 10 for t in tables)
    grids = np.stack([canvas(rng.integers(0, 10, size=(3, 5)).astype(np.uint8), 8), canvas(rng.integers(0, 10, size=(8, 2)).astype(np.uint8), 8)])
    for symmetry in range(8):
        for table in tables:
            seen = view(grids, symmetry, table)
            assert all((undo(s, symmetry, table) == g).all() for s, g in zip(seen, grids))
    assert (view(grids, 0, tables[0]) == grids).all()
    assert crop(view(grids, 1, tables[0])[0]).shape == (5, 3)            # a quarter turn swaps height and width
    ragged = grids[0].copy()
    ragged[1, 1] = PAD
    assert crop(ragged) is None and undo(ragged, 0, tables[0]) is None
    assert crop(np.full((8, 8), PAD, dtype=np.uint8)) is None


def test_an_episode_is_pairs_of_one_task_seen_through_one_view(tmp_path: Path) -> None:
    import_all()
    train, evaluation = write_files(tmp_path, size=8)
    data = DATASETS["arc_tasks"](str(train), {"tasks": str(evaluation)}, pairs=3, color_perms=2, symmetries=8, seed=0)
    batch = next(iter(data.train_loader(4, 1, 0, 0, 0, 1, 0, False)))
    assert batch["inputs"].shape == (4, 3, 8, 8) and batch["outputs"].shape == (4, 3, 8, 8) and batch["variant"].max() < 16
    for row in range(4):
        symmetry, table = int(batch["variant"][row]) % 8, data.tables[int(batch["variant"][row]) // 8]
        for slot in range(3):
            x = undo(batch["inputs"][row, slot].numpy().astype(np.uint8), symmetry, table)
            y = undo(batch["outputs"][row, slot].numpy().astype(np.uint8), symmetry, table)
            inside = x != PAD
            assert ((x[inside] + int(batch["task"][row]) + 1) % 10 == y[inside]).all() and (y[~inside] == PAD).all()
    assert (transform(np.arange(6, dtype=np.uint8).reshape(2, 3) % 10, 2, 3, 4, data.tables[0], 8)[:2, :3] == np.array([[2, 1, 0], [5, 4, 3]])).all()
    assert batch["mask"].all()
    plain = DATASETS["arc_tasks"](str(train), {}, pairs=3, color_perms=2, symmetries=8, seed=0, plain_share=1.0)
    assert (next(iter(plain.train_loader(4, 1, 0, 0, 0, 1, 0, False)))["variant"] == 0).all()
    for row in range(4):                                             # the pairs of an episode are different examples
        assert len({batch["inputs"][row, slot].numpy().tobytes() for slot in range(3)}) == 3


def test_short_tasks_give_masked_episodes_and_files_are_mixed_in_fixed_shares(tmp_path: Path) -> None:
    import_all()
    old, _ = write_files(tmp_path, size=8)
    rng = np.random.default_rng(3)
    grids = rng.integers(0, 10, size=(5, 8, 8)).astype(np.uint8)       # two new tasks with two and three examples
    new = tmp_path / "new.npz"
    np.savez(new, inputs=grids, outputs=grids, shapes=np.full((5, 4), 8, dtype=np.uint8), task=np.asarray([0, 0, 1, 1, 1], dtype=np.int32), names=np.asarray(["a", "b"]))
    data = DATASETS["arc_tasks"]([str(new), str(old)], {}, pairs=4, color_perms=2, symmetries=8, seed=0, shares=[0.25, 0.75], named=[False, True])
    batch = next(iter(data.train_loader(8, 1, 0, 0, 0, 1, 0, False)))
    assert batch["source"].tolist() == [0, 0, 1, 1, 1, 1, 1, 1] and (batch["task"][:2] == -1).all() and (batch["task"][2:] >= 0).all()
    assert all(int(batch["mask"][row].sum()) in (2, 3) for row in range(2)) and int(batch["mask"][2:].sum()) == 6 * 3      # old tasks have three examples
    assert (batch["inputs"][~batch["mask"]] == PAD).all() and batch["mask"][:, 0].all()

    from methods.neo_arc import NEOArcMethod
    from models.neo_arc import NEOArc

    torch.manual_seed(0)
    part = {"d_model": 32, "num_heads": 4, "d_ff": 64, "num_layers": 1, "dropout": 0.0}
    model = NEOArc(encoder={"type": "token_embed", "vocab": 11, "d_model": 16}, decoder={"type": "linear_head", "d_model": 16, "vocab": 11, "spatial": [8, 8]},
                   programmer={"type": "token_pair_transformer", "state_dim": 16, "action_dim": 8, "grid_dim": 8, **part},
                   executor={"type": "adaln_transformer", "state_dim": 16, "cond_dim": 8, "grid_dim": 8, **part}, code_dim=8, tasks=3, variants=16, table_share=1.0)
    for block in model.executor.stack.blocks:
        for parameter in block.parameters():
            torch.nn.init.normal_(parameter, std=0.2)
    out = model.train()(batch)
    assert not out.from_table[:8].any() and out.from_table[8:].all()          # tasks without a name never use the table
    assert out.valid.tolist() == batch["mask"].flatten().tolist()
    model.eval()
    row = int(batch["mask"][:2].sum(dim=1).argmin())                         # an episode of two pairs: each is predicted from the other alone
    if int(batch["mask"][row].sum()) == 2:
        own = model.pair_codes(batch["inputs"][row, :2], batch["outputs"][row, :2])
        assert torch.allclose(model.execute(batch["inputs"][row, :2], own.flip(0)), model(batch).logits_by_loop[0].view(8, 4, 8, 8, 11)[row, :2], atol=1e-5)
    method = NEOArcMethod(lr=1e-3, weight_decay=0.0, warmup_ratio=0.0, min_lr_ratio=1.0, grad_clip=1.0, total_steps=1)
    step = method.compute_loss(model.train(), batch, 0)
    assert torch.isfinite(step.total) and "grid_acc_file0" in step.metrics and "grid_acc_file1" in step.metrics


def test_a_pair_is_predicted_from_the_other_pairs_and_the_table_is_used_only_in_training(tmp_path: Path) -> None:
    import_all()
    from models.neo_arc import NEOArc

    torch.manual_seed(0)
    part = {"d_model": 32, "num_heads": 4, "d_ff": 64, "num_layers": 1, "dropout": 0.0}
    model = NEOArc(encoder={"type": "token_embed", "vocab": 11, "d_model": 16}, decoder={"type": "linear_head", "d_model": 16, "vocab": 11, "spatial": [6, 6]},
                   programmer={"type": "token_pair_transformer", "state_dim": 16, "action_dim": 8, "grid_dim": 6, **part},
                   executor={"type": "adaln_transformer", "state_dim": 16, "cond_dim": 8, "grid_dim": 6, **part}, code_dim=8, tasks=3, variants=2, table_share=1.0)
    for block in model.executor.stack.blocks:          # AdaLN-Zero starts as the identity; make the code matter
        for parameter in block.parameters():
            torch.nn.init.normal_(parameter, std=0.2)
    batch = {"inputs": torch.randint(0, 11, (2, 3, 6, 6)), "outputs": torch.randint(0, 11, (2, 3, 6, 6)), "task": torch.tensor([0, 2]), "variant": torch.tensor([1, 0])}
    model.eval()
    out = model(batch)
    assert not out.from_table.any() and len(out.logits_by_loop) == 1 and out.logits_by_loop[0].shape == (6, 6, 6, 11) and float(out.tie) > 0
    own = model.pair_codes(batch["inputs"][0], batch["outputs"][0])
    others = (own.sum(dim=0, keepdim=True) - own) / 2
    assert torch.allclose(model.execute(batch["inputs"][0], others), out.logits_by_loop[0][:3], atol=1e-5)
    changed = {**batch, "outputs": batch["outputs"].clone()}
    changed["outputs"][0, 0] = (changed["outputs"][0, 0] + 1) % 11          # the first pair's own answer must not reach its prediction
    again = model(changed).logits_by_loop[0]
    assert torch.allclose(again[0], out.logits_by_loop[0][0], atol=1e-5) and not torch.allclose(again[1], out.logits_by_loop[0][1], atol=1e-5)
    model.train()
    out = model(batch)
    assert out.from_table.all()
    given = model.table(batch["task"] * 2 + batch["variant"])
    assert torch.allclose(model.execute(batch["inputs"][1], given[1:2].expand(3, -1)), out.logits_by_loop[0][3:], atol=1e-5)
    model.loops = 3
    assert len(model(batch).logits_by_loop) == 3


def test_a_code_of_several_vectors_is_read_as_tokens_and_handled_as_one_flat_vector() -> None:
    import_all()
    from models.neo_arc import NEOArc

    torch.manual_seed(0)
    part = {"d_model": 32, "num_heads": 4, "d_ff": 64, "num_layers": 2, "dropout": 0.0}
    model = NEOArc(encoder={"type": "token_embed", "vocab": 11, "d_model": 16}, decoder={"type": "linear_head", "d_model": 16, "vocab": 11, "spatial": [6, 6]},
                   programmer={"type": "token_pair_queries", "state_dim": 16, "action_dim": 8, "grid_dim": 6, "num_queries": 3, **part},
                   executor={"type": "token_context_transformer", "state_dim": 16, "cond_dim": 8, "grid_dim": 6, "num_code_tokens": 3, **part},
                   code_dim=24, code_tokens=3).eval()
    x, y = torch.randint(0, 11, (4, 6, 6)), torch.randint(0, 11, (4, 6, 6))
    own = model.pair_codes(x, y)
    assert own.shape == (4, 24)
    out = model({"inputs": x[None], "outputs": y[None], "task": torch.tensor([-1]), "variant": torch.tensor([0])})
    others = (own.sum(dim=0, keepdim=True) - own) / 3
    assert torch.allclose(model.execute(x, others), out.logits_by_loop[0], atol=1e-5)
    changed = others.clone()
    changed[:, 16:] += 1.0                                      # the third vector of the code alone changes the prediction
    assert not torch.allclose(model.execute(x, changed), out.logits_by_loop[0], atol=1e-4)
    state = model.encoder(x)
    assert model.executor(state, others.view(4, 3, 8)).shape == state.shape


def test_fitting_a_code_lowers_the_loss_on_the_pairs_and_leaves_the_weights_alone(tmp_path: Path) -> None:
    import_all()
    from evals.arc import fit_codes
    from models.neo_arc import NEOArc

    torch.manual_seed(0)
    part = {"d_model": 32, "num_heads": 4, "d_ff": 64, "num_layers": 1, "dropout": 0.0}
    model = NEOArc(encoder={"type": "token_embed", "vocab": 11, "d_model": 16}, decoder={"type": "linear_head", "d_model": 16, "vocab": 11, "spatial": [6, 6]},
                   programmer={"type": "token_pair_transformer", "state_dim": 16, "action_dim": 8, "grid_dim": 6, **part},
                   executor={"type": "adaln_transformer", "state_dim": 16, "cond_dim": 8, "grid_dim": 6, **part}, code_dim=8).eval()
    for block in model.executor.stack.blocks:
        for parameter in block.parameters():
            torch.nn.init.normal_(parameter, std=0.2)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    x, y, task = torch.randint(0, 11, (5, 6, 6)), torch.randint(0, 11, (5, 6, 6)), torch.tensor([0, 0, 1, 1, 1])
    loss = lambda code: float(torch.nn.functional.cross_entropy(model.execute(x, code[task]).flatten(0, 2), y.flatten()))  # noqa: E731
    start = torch.zeros(2, 8)
    fitted = fit_codes(model, x, y, task, start, steps=20, lr=0.05, batch_size=2)
    assert loss(fitted) < loss(start) and (start == 0).all()
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items()) and all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_config_builds_steps_and_evaluates(path: Path, tmp_path: Path) -> None:
    import_all()
    torch.manual_seed(0)
    config = load_config(path)
    size = config["model"]["decoder"]["spatial"][0]
    train, evaluation = write_files(tmp_path, size)
    several = isinstance(config["data"]["train"], list)
    block = {**config["data"], "train": [str(train)] * len(config["data"]["train"]) if several else str(train),
             "evals": {split: str(evaluation) for split in config["data"]["evals"]}}
    data = build(DATASETS, block)
    assert data.variants == config["model"]["variants"]
    method = build(METHODS, config["method"])
    model = method.build_model(config["model"], init_from=None)
    method.attach(model, torch.device("cpu"), "fp32")
    method.set_train_mode(model)
    metrics = method.train_step(model, next(iter(data.train_loader(2, 1, 0, 0, 0, 1, 0, False))), 0)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values()), metrics
    blocks = [{**b, "batch_size": 4, **({"fit_steps": 2, "fit_report": [1], "votes": 3} if b.get("votes") else {"fit_steps": 2} if b.get("fit_steps") else {})}
              for b in config["evals"]]
    results = run_evals(model, data, blocks, 1, torch.device("cpu"))
    for b in blocks:
        prefix = f"{b.get('name', b['type'])}/{b['split']}"
        assert results[f"{prefix}/tasks"] == 3 and 0.0 <= results[f"{prefix}/zero"] <= 1.0 and 0.0 <= results[f"{prefix}/verify_tasks"] <= 1.0
        if b.get("votes"):
            assert all(0.0 <= results[f"{prefix}/{key}"] <= 1.0 for key in ("fit1", "fit2", "fit2_vote1", "fit2_vote2", "zero_vote2", "fit_demo_tasks"))
        elif b.get("fit_steps"):
            assert 0.0 <= results[f"{prefix}/fit2"] <= 1.0
        if config["model"]["tasks"] and b.get("table"):
            assert 0.0 <= results[f"{prefix}/table"] <= 1.0 and results[f"{prefix}/table"] == results[f"{prefix}/given"]
