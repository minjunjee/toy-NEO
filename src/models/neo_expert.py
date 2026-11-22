"""The lab NEO with an executor that has one network per code (components/executors_expert.py).
Registered as MODELS["neo_expert"]. The only change is that the executor is handed the codebook, so it can tell which
code it was given; the rollout, the losses and the evaluation are the lab NEO's (state_update as configured)."""

from __future__ import annotations

import torch

from core.registry import MODELS
from models.neo import NEO


class NEOExpert(NEO):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if getattr(self.executor, "size", None) != self.code.size:
            raise ValueError("neo_expert needs a per_code executor with one network per code (executor.size = code.size)")
        code = self.code
        object.__setattr__(self.executor, "table", lambda: code.codebook(torch.arange(code.size, device=next(code.parameters()).device)))


MODELS["neo_expert"] = NEOExpert
