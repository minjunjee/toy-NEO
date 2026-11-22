"""NEO training plus one request to the executor: amounts of one kind add. Registered as METHODS["neo_compose"].

It needs a codebook that indexes its codes by a kind and an integer amount (components/codes_orbit.py). For random
inputs of the batch and random codes (g, a) and (g, b) whose sum (g, a + b) is in the codebook, the executor runs (g, a),
the result is decoded to a grid and encoded again, and the executor runs (g, b) on it. The grid that comes out is the
target, with no gradient. The executor then runs (g, a + b) once on the same input and is trained (cell cross-entropy)
to give that grid. Nothing here looks at the task: the inputs are the batch's input grids and the codes are drawn from
the index table, so a code that no training pair ever selected still gets a target, built from codes that were trained.

The code vectors are not trained by this term (they are detached), only the executor is. `compose_weight` scales the
term and `compose_from` is the training step at which it starts.

`compose_known` > 0 builds the target only from codes in use: (g, a) and (g, b) must each have been picked by at least
that fraction of the uniform share 1/size of recent training pairs (the codebook's running count). What the data has
taught then teaches the code for the sum, and an idle code, which the executor has no reason to treat in any particular
way, is never the teacher. The code for the sum may be idle; that is the case the term is for. With 0 every pair of
codes with a sum is used.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from core.registry import METHODS
from methods.base import TrainingStep, unwrap_model
from methods.neo import NEOMethod


class NEOComposeMethod(NEOMethod):
    def __init__(self, *args, compose_weight: float = 1.0, compose_from: int = 0, compose_known: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compose_weight, self.compose_from, self.compose_known = compose_weight, compose_from, compose_known

    def compute_loss(self, model: nn.Module, batch: dict, step: int) -> TrainingStep:
        out = super().compute_loss(model, batch, step)
        if not self.compose_weight or step < self.compose_from:
            return out
        module = unwrap_model(model)
        code, x = module.code, batch["x"]
        with torch.no_grad():
            table = code.table()
            allowed = code.sums >= 0
            if self.compose_known:
                known = code.usage >= self.compose_known / code.size
                allowed = allowed & known[:, None] & known[None, :]
            pairs = torch.nonzero(allowed)
            if not len(pairs):
                return TrainingStep(out.total, {**out.metrics, "compose": 0.0, "compose_agree": 0.0, "compose_pairs": 0.0})
            pick = pairs[torch.randint(len(pairs), (x.shape[0],), device=x.device)]
            first, second = table[pick[:, 0]][:, None], table[pick[:, 1]][:, None]
            total = table[code.sums[pick[:, 0], pick[:, 1]]][:, None]
            state = module.encoder(x)
            middle = module.decoder(module.executor(state, first)).argmax(dim=-1)
            target = module.decoder(module.executor(module.encoder(middle), second)).argmax(dim=-1)
        logits = module.decoder(module.executor(state, total))
        compose = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1).long())
        agree = logits.argmax(dim=-1).eq(target).flatten(1).all(dim=1).float().mean()
        metrics = {**out.metrics, "loss": out.total + self.compose_weight * compose, "compose": compose, "compose_agree": agree, "compose_pairs": float(len(pairs))}
        return TrainingStep(out.total + self.compose_weight * compose, metrics)


METHODS["neo_compose"] = NEOComposeMethod
