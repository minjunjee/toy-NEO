"""NEO whose program steps hand over only what is visible on the grid. Registered as MODELS["neo_visible_state"].

With the lab's `reencode_soft` the next step starts from the encoding of the decoded cell probabilities. Besides the grid,
that state tells the next step how sure each cell was, and trained models use it: one code applied to its own soft output
does something else than the same code applied to the clean grid it decoded to (docs/reports/2026-11-20-first-numbers.md,
section 5.11). The lab's `reencode_hard` closes that path but also cuts the gradient between steps.

This model closes the path and keeps the gradient: the next step starts from the encoding of the decoded grid itself
(one-hot of the argmax), and in the backward pass that encoding is treated as if it were the soft one (straight-through).
Everything else is the lab NEO. At inference every rollout re-encodes the decoded grid, as with `reencode_hard`.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from core.registry import MODELS
from models.neo import NEO, NEOOutput, mdl_select


class NEOVisibleState(NEO):
    def __init__(self, *args, **kwargs) -> None:
        if kwargs.pop("state_update", "reencode_hard") != "reencode_hard":
            raise ValueError("neo_visible_state fixes the state update; leave state_update out")
        super().__init__(*args, state_update="reencode_hard", **kwargs)

    def forward(self, batch: dict, step: int = 0) -> NEOOutput:
        """The lab forward pass with one line changed: the state handed to the next step."""

        x, y = batch["x"], batch["y"]
        observations = torch.cat([x, y], dim=0)
        states = self.encoder(observations)
        auto_logits = self.decoder(states)
        state, target_state = states[: x.shape[0]], states[x.shape[0] :]

        logits_by_length, code_loss_by_length, code_indices_by_length = [], [], []
        grounding = torch.zeros((), device=x.device, dtype=states.dtype)
        temperature = 0.0
        for _ in range(self.max_length):
            state, code = self.step(state, target_state, self.training)
            code_loss_by_length.append(code.loss_per_sample)
            code_indices_by_length.append(code.indices)
            temperature = code.temperature
            logits = self.decoder(state)
            logits_by_length.append(logits)
            with torch.no_grad():
                clean_state = self.encoder(logits.argmax(dim=-1))
            if self.max_length != 1:
                grounding = grounding + F.mse_loss(state, clean_state)
            soft = torch.softmax(logits, dim=-1)
            visible = F.one_hot(logits.argmax(dim=-1), soft.shape[-1]).to(soft.dtype)
            state = self.encoder(visible + soft - soft.detach())  # the value of the hard grid, the gradient of the soft one

        if self.training:
            self.set_length_cost(step)
        selected = mdl_select(logits_by_length, y, self.length_coefficient, self.length_cost_now or None)
        return NEOOutput(
            logits_by_length, selected, grounding, code_loss_by_length, code_indices_by_length,
            temperature, auto_logits, observations, y,
        )


MODELS["neo_visible_state"] = NEOVisibleState
