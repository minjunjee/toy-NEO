"""Evaluations on BabyTheorist eval files. Registered as "reconstruct_exact" and "babytheorist_observe".

reconstruct_exact      the state encoder and decoder alone: share of grids decoded back exactly.
babytheorist_observe   what a trained NEO does on the tasks of one eval file, split by the depth of the hidden program:
    explain            the program taken from the first support pair reproduces that pair's output
    transfer_lab       that program replayed on the query, by the lab evaluator (with rollout "official" the replay stops
                       as soon as the prediction equals the query answer)
    transfer_strict    the same replay without showing the query answer
    transfer_consistent  a program is taken from every support pair; the one that reproduces the most support pairs is
                       replayed on the query (ties go to the earlier pair), without showing the query answer
  and, as tables: the number of codes extracted against the true depth, how often each code is used, and for tasks whose
  program is one primitive, which code was chosen for which primitive (counted both when the length rule picked one
  code, and when that one code also reproduces the support output exactly). The hidden program is read here only to sort the
  results; the model never sees it.
"""

from __future__ import annotations

import json

import torch
from torch import nn

from core.registry import EVALS

DEPTH_BUCKETS = (1, 2, 3, 4, 5)  # 5 stands for 5 and deeper


def on_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def exact(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return pred.eq(target).flatten(1).all(dim=1)


@torch.no_grad()
def reconstruct_exact(model: nn.Module, data, cfg: dict, step: int, device: torch.device) -> dict:
    solved = total = 0
    cells = 0.0
    for batch in data.eval_loader(cfg["split"], cfg.get("batch_size", 512), max_samples=cfg.get("max_samples")):
        batch = on_device(batch, device)
        grids = torch.cat([batch["support_x"], batch["support_y"], batch["query_x"], batch["query_y"]])
        state, posterior = model.encoder.encode(grids) if hasattr(model.encoder, "encode") else (model.encoder(grids), None)
        if posterior is not None:
            state = posterior.mean  # the state a frozen encoder hands to NEO
        pred = model.decoder(state).argmax(dim=-1)
        solved += int(exact(pred, grids).sum())
        cells += float(pred.eq(grids).float().flatten(1).mean(dim=1).sum())
        total += len(grids)
    return {"exact": solved / max(total, 1), "cell": cells / max(total, 1), "total": total}


@torch.no_grad()
def babytheorist_observe(model: nn.Module, data, cfg: dict, step: int, device: torch.device) -> dict:
    infer = dict(cfg.get("infer", {}))
    official = infer.get("rollout", "official") == "official"
    max_steps = infer.get("max_steps") or model.max_length
    names = ("explain", "transfer_lab", "transfer_strict", "transfer_consistent")
    hits = {name: torch.zeros(len(DEPTH_BUCKETS) + 1) for name in names}
    counts = torch.zeros(len(DEPTH_BUCKETS) + 1)
    extracted_by_depth = torch.zeros(len(DEPTH_BUCKETS) + 1, max_steps + 1)
    codes = getattr(model.code, "size", None)
    usage = torch.zeros(codes) if codes else None
    one_step: dict[str, dict] = {}
    for batch in data.eval_loader(cfg["split"], cfg.get("batch_size", 256), max_samples=cfg.get("max_samples")):
        batch = on_device(batch, device)
        sx, sy, k, qx, qy = batch["supports_x"], batch["supports_y"], batch["n_support"], batch["query_x"], batch["query_y"]
        rows, most = sx.shape[0], sx.shape[1]
        lab = model.infer(sx[:, 0], sy[:, 0], qx, qy, **infer)
        strict = model.infer(sx[:, 0], sy[:, 0], qx, None, **infer) if official else lab
        explained = torch.full((rows, most), -1, device=device)
        query_preds = []
        first_explains = None
        for i in range(most):
            own = exact(model.self_explain(sx[:, i], sy[:, i], **infer)["pred"], sy[:, i])
            if i == 0:
                first_explains = own
            count = own.long()
            for j in range(most):
                if j != i:
                    count = count + (exact(model.infer(sx[:, i], sy[:, i], sx[:, j], None, **infer)["pred"], sy[:, j]) & (j < k)).long()
            explained[:, i] = torch.where(i < k, count, torch.full_like(count, -1))
            query_preds.append((strict if i == 0 else model.infer(sx[:, i], sy[:, i], qx, None, **infer))["pred"])
        best = explained.argmax(dim=1)  # the first maximum
        consistent = torch.stack(query_preds, dim=1)[torch.arange(rows, device=device), best]
        results = {"explain": first_explains, "transfer_lab": exact(lab["pred"], qy), "transfer_strict": exact(strict["pred"], qy), "transfer_consistent": exact(consistent, qy)}
        bucket = batch["depth"].clamp(max=DEPTH_BUCKETS[-1]) - 1
        for b in range(len(DEPTH_BUCKETS)):
            mine = bucket == b
            counts[b] += int(mine.sum())
            for name in names:
                hits[name][b] += int(results[name][mine].sum())
            lengths = lab["n_extracted"][mine].clamp(max=max_steps)
            extracted_by_depth[b] += torch.bincount(lengths.cpu(), minlength=max_steps + 1)[: max_steps + 1]
        indices = lab.get("indices")
        if usage is not None and indices and indices[0] is not None:
            steps = torch.stack([i.view(rows, -1)[:, 0] for i in indices], dim=1)  # (rows, max_steps), the first code token of each step
            used = torch.arange(steps.shape[1], device=device)[None] < lab["n_extracted"][:, None]
            usage += torch.bincount(steps[used].cpu(), minlength=codes)
            for row in torch.nonzero(batch["depth"] == 1).flatten().tolist():
                primitive = json.loads(batch["transitions"][row])[0]["primitive"]
                entry = one_step.setdefault(primitive, {"tasks": 0, "explained": 0, "one_code_extracted": 0, "one_code_exact": 0, "transfer_strict": 0, "code": [0] * codes, "exact_code": [0] * codes})
                entry["tasks"] += 1
                entry["explained"] += int(first_explains[row])
                entry["transfer_strict"] += int(results["transfer_strict"][row])
                if int(lab["n_extracted"][row]) == 1:  # the length rule chose one code, whether or not it reproduces the support output
                    entry["one_code_extracted"] += 1
                    entry["code"][int(steps[row, 0])] += 1
                    if bool(first_explains[row]):  # and that one code reproduces the support output exactly
                        entry["one_code_exact"] += 1
                        entry["exact_code"][int(steps[row, 0])] += 1
    total = float(counts.sum())
    out: dict = {"total": int(total)}
    for name in names:
        out[name] = float(hits[name].sum()) / max(total, 1.0)
        for b, depth in enumerate(DEPTH_BUCKETS):
            out[f"{name}_depth{depth}"] = float(hits[name][b]) / max(float(counts[b]), 1.0)
    for b, depth in enumerate(DEPTH_BUCKETS):
        out[f"tasks_depth{depth}"] = int(counts[b])
    if cfg.get("tables", True):
        out["extracted_length_by_depth"] = {str(depth): extracted_by_depth[b].int().tolist() for b, depth in enumerate(DEPTH_BUCKETS)}
        if usage is not None:
            out["code_usage"] = usage.int().tolist()
            out["one_step_tasks"] = dict(sorted(one_step.items()))
    return out


EVALS["reconstruct_exact"] = reconstruct_exact
EVALS["babytheorist_observe"] = babytheorist_observe
