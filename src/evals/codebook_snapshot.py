"""A light record of the codebook, taken often during training to watch it settle. Registered as "codebook_snapshot".

The full evaluation (babytheorist.py) replays programs several times and is too slow to run every hundred steps. This one
looks at single-primitive tasks only: the first `max_tasks` tasks of the eval split whose hidden program is one primitive,
from the first support pair. The hidden primitive only sorts the record; the model never sees it.

    chosen      per primitive, how often the programmer picked each code (one step, as in training)
    reproduces  per primitive, how often each code, run by the executor on the support input, gives the support output
                exactly; this is what a code means to the executor, whatever the programmer picks
    transfer    per primitive, how many tasks the chosen code also solves on the query
    codes       the code vectors at unit length, as the executor reads them
    em_usage, em_bonus   the hard EM usage average and dead-code bonus, when the model has them

Scalars: codes_used (codes the programmer picked at least once), code_nmi (normalized mutual information between the
primitive and the chosen code, as in babytheorist.py), transfer (share over all probe tasks).
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from torch import nn

from core.registry import EVALS
from evals.babytheorist import normalized_mutual_information

_PROBES: dict = {}  # the probe tasks are read once per data object and split


def _probe(data, split: str, max_tasks: int, max_samples: int | None, device: torch.device):
    key = (id(data), split, max_tasks, max_samples)
    if key not in _PROBES:
        sx, sy, qx, qy, names = [], [], [], [], []
        for batch in data.eval_loader(split, 512, max_samples=max_samples):
            for row in torch.nonzero(batch["depth"] == 1).flatten().tolist():
                if len(names) >= max_tasks:
                    break
                sx.append(batch["supports_x"][row, 0])
                sy.append(batch["supports_y"][row, 0])
                qx.append(batch["query_x"][row])
                qy.append(batch["query_y"][row])
                names.append(json.loads(batch["transitions"][row])[0]["primitive"])
            if len(names) >= max_tasks:
                break
        stacked = tuple(torch.stack(t) if t else None for t in (sx, sy, qx, qy))
        _PROBES[key] = (*stacked, names)
    sx, sy, qx, qy, names = _PROBES[key]
    if sx is None:
        return None
    return sx.to(device), sy.to(device), qx.to(device), qy.to(device), names


@torch.no_grad()
def codebook_snapshot(model: nn.Module, data, cfg: dict, step: int, device: torch.device) -> dict:
    table = F.normalize(model.code.embedding.weight.float(), dim=-1)
    size = table.shape[0]
    out: dict = {"codes": [[round(v, 4) for v in row] for row in table.cpu().tolist()]}
    for name in ("usage", "bonus"):
        buffer = getattr(model, name, None)
        if isinstance(buffer, torch.Tensor):
            out[f"em_{name}"] = [round(v, 5) for v in buffer.float().cpu().tolist()]
    probe = _probe(data, cfg["split"], int(cfg.get("max_tasks", 1024)), cfg.get("max_samples"), device)
    if probe is None:
        out.update({"codes_used": 0, "code_nmi": 0.0, "transfer": 0.0, "primitives": {}})
        return out
    sx, sy, qx, qy, names = probe
    n = sx.shape[0]
    state, target = model.encoder(sx), model.encoder(sy)
    chosen = model.programmer(state, target)
    chosen = model.code(chosen, False).indices.view(n, -1)[:, 0]
    query_state = model.encoder(qx)
    query_pred = model.decoder(model.executor(query_state, table[chosen][:, None])).argmax(dim=-1)
    solved = query_pred.eq(qy).flatten(1).all(dim=1)
    reproduces = torch.zeros(size, n, dtype=torch.bool, device=device)
    chunk = max(1, 8192 // max(n, 1))
    for start in range(0, size, chunk):
        codes = torch.arange(start, min(size, start + chunk), device=device)
        k = len(codes)
        values = table[codes][:, None, None].expand(k, n, 1, table.shape[1]).reshape(k * n, 1, table.shape[1])
        pred = model.decoder(model.executor(state.repeat(k, *[1] * (state.dim() - 1)), values)).argmax(dim=-1)
        reproduces[codes] = pred.view(k, n, *sy.shape[1:]).eq(sy[None]).flatten(2).all(dim=2)
    primitives: dict[str, dict] = {}
    chosen_list, solved_list = chosen.cpu().tolist(), solved.cpu().tolist()
    hits = reproduces.cpu()
    for i, name in enumerate(names):
        entry = primitives.setdefault(name, {"tasks": 0, "transfer": 0, "chosen": [0] * size, "reproduces": [0] * size})
        entry["tasks"] += 1
        entry["transfer"] += int(solved_list[i])
        entry["chosen"][chosen_list[i]] += 1
    for name, entry in primitives.items():
        rows = torch.tensor([i for i, other in enumerate(names) if other == name])
        entry["reproduces"] = hits[:, rows].sum(dim=1).int().tolist()
    out["primitives"] = dict(sorted(primitives.items()))
    out["codes_used"] = int(len(set(chosen_list)))
    out["code_nmi"] = normalized_mutual_information(torch.tensor([entry["chosen"] for entry in out["primitives"].values()]))
    out["transfer"] = float(solved.float().mean())
    return out


EVALS["codebook_snapshot"] = codebook_snapshot
