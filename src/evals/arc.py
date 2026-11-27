"""Evaluation on ARC task files with frozen weights. Registered as EVALS["arc_tasks"].

A task is its demonstration pairs and its test inputs. A test counts only when the whole predicted canvas equals the
answer, pad cells included, so a wrong output size is a miss. A task's score is the share of its tests solved; every
reported score is the mean over tasks.

What is measured, from the cheapest way of naming the task's code to the dearest:
  zero        the programmer reads every demonstration pair once; the code is the mean of the pair codes
  fit@k       that code is then moved by k steps of Adam on the cross-entropy of the demonstration pairs, weights frozen
              (`fit_steps`, reported at every count in `fit_report` and at the end)
  *_vote1/2   the same done for `votes` views of the task (symmetries of the square and the colour permutations the
              training set used), each prediction mapped back, and the most frequent grid taken (vote1) or the two most
              frequent (vote2: solved when either is right). Predictions whose non-pad cells are not a rectangle are dropped.
  table       for tasks of the training set (`table: true`): executed with the code the table holds for the task
Checks that need no answers:
  verify_pairs, verify_tasks   demonstration pairs reproduced from the mean code of the task's other pairs, and tasks
              where all of them are
  fit_demo_tasks               tasks whose demonstration pairs are all reproduced by the fitted code
seconds_per_task_zero and seconds_per_task_fit are wall-clock times of the first view, per task.
With `detail: true` the per-task results are returned too (names, solved by each way, the two checks).
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from core.registry import EVALS
from data.arc_tasks import PAD, transform


def crop(canvas: np.ndarray) -> np.ndarray | None:
    """The grid on a canvas; None when its non-pad cells are not a rectangle at the top left."""

    mask = canvas != PAD
    h, w = int(mask[:, 0].sum()), int(mask[0, :].sum())
    if h == 0 or w == 0 or int(mask.sum()) != h * w or not mask[:h, :w].all():
        return None
    return canvas[:h, :w]


def view(canvases: np.ndarray, symmetry: int, table: np.ndarray) -> np.ndarray:
    size = canvases.shape[-1]
    out = np.empty_like(canvases)
    for i, canvas in enumerate(canvases):
        g = crop(canvas)
        out[i] = transform(g, g.shape[0], g.shape[1], symmetry, table, size)
    return out


def undo(canvas: np.ndarray, symmetry: int, table: np.ndarray) -> np.ndarray | None:
    g = crop(canvas)
    if g is None:
        return None
    g = np.rot90(np.argsort(table).astype(np.uint8)[g], -(symmetry % 4))
    if symmetry >= 4:
        g = g[:, ::-1]
    out = np.full_like(canvas, PAD)
    out[: g.shape[0], : g.shape[1]] = g
    return out


def autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def read_codes(model, x: Tensor, y: Tensor, batch_size: int) -> Tensor:
    out = []
    for i in range(0, len(x), batch_size):
        with autocast(x.device):
            out.append(model.pair_codes(x[i: i + batch_size], y[i: i + batch_size]))
    return torch.cat(out).float()


@torch.no_grad()
def predict(model, x: Tensor, codes: Tensor, batch_size: int) -> Tensor:
    out = []
    for i in range(0, len(x), batch_size):
        with autocast(x.device):
            out.append(model.execute(x[i: i + batch_size], codes[i: i + batch_size]).argmax(dim=-1))
    return torch.cat(out)


def fit_codes(model, x: Tensor, y: Tensor, task: Tensor, start: Tensor, steps: int, lr: float, batch_size: int, report: dict | None = None):
    """Adam on one code per task against the cross-entropy of its pairs, weights untouched. `report` maps a step count to
    a function called with the codes at that count."""

    code = start.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([code], lr=lr)
    for step in range(1, steps + 1):
        grad = torch.zeros_like(code)
        for i in range(0, len(x), batch_size):
            with torch.enable_grad():
                with autocast(x.device):
                    logits = model.execute(x[i: i + batch_size], code[task[i: i + batch_size]])
                target = y[i: i + batch_size]
                loss = F.cross_entropy(logits.float().flatten(0, 2), target.flatten(), reduction="sum") / target[0].numel()
                grad += torch.autograd.grad(loss, code)[0]
        code.grad = grad
        optimizer.step()
        if report and step in report:
            report[step](code.detach())
    return code.detach()


def task_mean(values: Tensor, task: Tensor, tasks: int) -> Tensor:
    total = torch.zeros(tasks, *values.shape[1:], device=values.device, dtype=values.dtype).index_add_(0, task, values)
    count = torch.zeros(tasks, device=values.device).index_add_(0, task, torch.ones(len(task), device=values.device))
    return total / count.view(-1, *[1] * (values.dim() - 1))


def arc_tasks(model, data, cfg: dict, step: int, device: torch.device) -> dict:
    cfg = dict(cfg)
    split, batch_size, max_tasks = cfg["split"], int(cfg.get("batch_size", 256)), cfg.get("max_tasks")
    fit_steps, fit_lr, votes = int(cfg.get("fit_steps", 0)), float(cfg.get("fit_lr", 0.05)), int(cfg.get("votes", 1))
    fit_report = sorted({int(k) for k in cfg.get("fit_report", []) if int(k) < fit_steps} | ({fit_steps} if fit_steps else set()))
    file = data.eval_tasks(split)
    tasks = len(file["names"]) if max_tasks is None else min(int(max_tasks), len(file["names"]))
    keep_demo, keep_test = file["demo_task"] < tasks, file["test_task"] < tasks
    demo_x, demo_y, demo_task = file["demo_inputs"][keep_demo], file["demo_outputs"][keep_demo], file["demo_task"][keep_demo].astype(np.int64)
    test_x, test_y, test_task = file["test_inputs"][keep_test], file["test_outputs"][keep_test], file["test_task"][keep_test].astype(np.int64)
    d_task, t_task = torch.from_numpy(demo_task).to(device), torch.from_numpy(test_task).to(device)
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).long().to(device)  # noqa: E731
    tests_of = np.bincount(test_task, minlength=tasks)

    def score(solved: np.ndarray) -> float:
        return float((np.bincount(test_task, weights=solved.astype(float), minlength=tasks) / tests_of).mean())

    symmetries, tables = data.symmetries, data.tables
    views = [(s, c) for c in range(len(tables)) for s in range(symmetries)][:votes]
    ways = ["zero"] + [f"fit{k}" for k in fit_report]
    guesses: dict[str, list[list[bytes | None]]] = {way: [[] for _ in range(len(test_x))] for way in ways}   # way -> test -> view -> grid
    result: dict = {"tasks": tasks, "tests": int(len(test_x))}
    detail: dict = {}

    def keep(way: str, prediction: Tensor, symmetry: int, table: np.ndarray) -> None:
        grids = prediction.cpu().numpy().astype(np.uint8)
        for q, grid in enumerate(grids):
            back = undo(grid, symmetry, table)
            guesses[way][q].append(None if back is None else back.tobytes())

    for number, (symmetry, color) in enumerate(views):
        table = tables[color]
        first = number == 0
        vx, vy, vt = (demo_x, demo_y, test_x) if first else (view(demo_x, symmetry, table), view(demo_y, symmetry, table), view(test_x, symmetry, table))
        vx, vy, vt = tensor(vx), tensor(vy), tensor(vt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.monotonic()
        own = read_codes(model, vx, vy, batch_size)
        mean = task_mean(own, d_task, tasks)
        zero = predict(model, vt, mean[t_task], batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        zero_seconds = time.monotonic() - started
        keep("zero", zero, symmetry, table)
        if first:
            result["seconds_per_task_zero"] = zero_seconds / tasks
            count = torch.zeros(tasks, device=device).index_add_(0, d_task, torch.ones(len(d_task), device=device))
            others = (mean[d_task] * count[d_task, None] - own) / (count[d_task, None] - 1).clamp(min=1)
            reproduced = predict(model, vx, others, batch_size).eq(vy).flatten(1).all(dim=1).float()
            result["verify_pairs"] = float(reproduced.mean())
            verified = task_mean(reproduced, d_task, tasks).eq(1)
            result["verify_tasks"] = float(verified.float().mean())
            detail["verify"] = verified.cpu().tolist()
        if fit_steps:
            started = time.monotonic()
            report = {k: (lambda code, k=k: keep(f"fit{k}", predict(model, vt, code[t_task], batch_size), symmetry, table)) for k in fit_report}
            fitted = fit_codes(model, vx, vy, d_task, mean, fit_steps, fit_lr, batch_size, report)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if first:
                result["seconds_per_task_fit"] = (time.monotonic() - started) / tasks
                fit_demo = task_mean(predict(model, vx, fitted[d_task], batch_size).eq(vy).flatten(1).all(dim=1).float(), d_task, tasks).eq(1)
                result["fit_demo_tasks"] = float(fit_demo.float().mean())
                detail["fit_demo"] = fit_demo.cpu().tolist()

    answers = [grid.tobytes() for grid in test_y.astype(np.uint8)]
    for way in ways:
        solved = np.array([g[0] == a for g, a in zip(guesses[way], answers)])
        result[way] = score(solved)
        detail[way] = (np.bincount(test_task, weights=solved.astype(float), minlength=tasks) / tests_of).tolist()
        if votes > 1:
            top = [[grid for grid, _ in Counter(x for x in g if x is not None).most_common(2)] for g in guesses[way]]
            result[f"{way}_vote1"] = score(np.array([bool(t) and t[0] == a for t, a in zip(top, answers)]))
            result[f"{way}_vote2"] = score(np.array([a in t for t, a in zip(top, answers)]))
            if way == ways[-1]:
                detail["vote2"] = (np.bincount(test_task, weights=np.array([a in t for t, a in zip(top, answers)], dtype=float), minlength=tasks) / tests_of).tolist()

    if cfg.get("table") and getattr(model, "table", None) is not None and "train_index" in file:
        index = file["train_index"][:tasks].astype(np.int64)
        known = (index >= 0) & (index < model.tasks)
        if known.any():
            row = torch.from_numpy(index[test_task] * model.variants).to(device).clamp(min=0)
            given = predict(model, tensor(test_x), model.table.weight[row].float(), batch_size).eq(tensor(test_y)).flatten(1).all(dim=1).cpu().numpy()
            per_task = np.bincount(test_task, weights=given.astype(float), minlength=tasks) / tests_of
            result["table"] = float(per_task[known].mean())
            for way in ("zero", ways[-1]):
                result[f"{way}_known"] = float(np.asarray(detail[way])[known].mean())
    if cfg.get("detail"):
        result["detail"] = {"names": [str(n) for n in file["names"][:tasks]], **detail}
    return result


EVALS["arc_tasks"] = arc_tasks
