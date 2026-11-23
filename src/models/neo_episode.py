"""The lab NEO, one program step, trained on tasks of two pairs that share a program. Registered as MODELS["neo_episode"].

Why. Trained on single pairs, a code only has to reproduce the pair it was named for, so a code can do one thing on wide
objects and another on tall ones, and a primitive can be spread over codes by object shape: the support is reproduced and
the query is not (hard EM over codes made this plain: 98% to 100% of support pairs reproduced, 69% transferred).

What. The programmer names a code from the first pair only, as it will at evaluation. The executor runs that code on the
first pair's input and on the second pair's input, and both outputs are trained by the cell cross-entropy; the second is
the transfer the evaluation asks for. Everything else is the lab NEO's. Without x2 in the batch the model is the lab NEO.
"""

from __future__ import annotations

import torch

from core.registry import MODELS
from models.neo import NEO, NEOOutput


class NEOEpisode(NEO):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.max_length != 1:
            raise ValueError("neo_episode is a one-step model: max_length must be 1")

    def forward(self, batch: dict, step: int = 0) -> NEOOutput:
        if "x2" not in batch:
            return super().forward(batch, step)
        x, y, x2, y2 = batch["x"], batch["y"], batch["x2"], batch["y2"]
        observations = torch.cat([x, y, x2, y2], dim=0)
        states = self.encoder(observations)
        auto_logits = self.decoder(states)
        n = x.shape[0]
        state, target_state, state2 = states[:n], states[n: 2 * n], states[2 * n: 3 * n]
        action = self.programmer(state.detach(), target_state)
        code = self.code(action, self.training)
        logits = self.decoder(self.executor(torch.cat([state, state2], dim=0), torch.cat([code.values, code.values], dim=0)))
        both = torch.cat([y, y2], dim=0)
        return NEOOutput([logits], torch.ones(2 * n, dtype=torch.long, device=x.device), torch.zeros((), device=x.device),
                         [torch.cat([code.loss_per_sample, code.loss_per_sample])], [torch.cat([code.indices, code.indices])],
                         code.temperature, auto_logits, observations, both)


MODELS["neo_episode"] = NEOEpisode
