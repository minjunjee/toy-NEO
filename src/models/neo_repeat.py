"""NEO whose program is one code and a count: the code is applied that many times. Registered as MODELS["neo_repeat"].

In the lab NEO a program is a sequence of codes and the programmer picks a new code at every step. Here the programmer
looks at the pair once and picks one code, the executor applies that same code up to `max_length` times in a row, and
the lab's length rule (`mdl_select`: the shortest application count whose decoded grid matches, else the lowest
discounted loss) picks the count. So "move up four cells" can be written as (move up one cell, 4), and an amount that no
training pair had is reachable whenever the code for the unit exists.

Between applications only the decoded grid is handed over, with the straight-through gradient of
models/neo_visible_state.py, so the loss at count t trains all t applications of the one code together.
Everything else (state encoder and decoder, programmer, codebook, executor, losses) is the lab NEO's.

Evaluate with `infer: {max_steps: <count>, rollout: hard}`; the lab's "official" rollout picks a code per step and is
refused here.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from core.registry import MODELS
from models.neo import NEOOutput, mdl_select, select_by_length
from models.neo_visible_state import NEOVisibleState


class NEORepeat(NEOVisibleState):
    def forward(self, batch: dict, step: int = 0) -> NEOOutput:
        x, y = batch["x"], batch["y"]
        observations = torch.cat([x, y], dim=0)
        states = self.encoder(observations)
        auto_logits = self.decoder(states)
        state, target_state = states[: x.shape[0]], states[x.shape[0] :]
        code = self.code(self.programmer(state.detach(), target_state), self.training)  # one code for the whole program

        logits_by_length, code_loss_by_length, code_indices_by_length = [], [], []
        grounding = torch.zeros((), device=x.device, dtype=states.dtype)
        for _ in range(self.max_length):
            state = self.executor(state, code.values)
            code_loss_by_length.append(code.loss_per_sample)
            code_indices_by_length.append(code.indices)
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
            code.temperature, auto_logits, observations, y,
        )

    @torch.no_grad()
    def explain_mdl(self, x: Tensor, y: Tensor, max_steps: int, rollout: str):
        if rollout != "hard":
            raise ValueError('neo_repeat hands the grid between applications; evaluate with rollout "hard"')
        state, target_state = self.encoder(x), self.encoder(y)
        code = self.code(self.programmer(state, target_state), False)
        logits_by_length = []
        for _ in range(max_steps):
            logits = self.decoder(self.executor(state, code.values))
            logits_by_length.append(logits)
            state = self.encoder(logits.argmax(dim=-1))
        lengths = mdl_select(logits_by_length, y, self.length_coefficient, self.length_cost_now or None)
        return select_by_length(logits_by_length, lengths).argmax(dim=-1), lengths, [code.values] * max_steps, [code.indices] * max_steps

    def self_explain(self, x: Tensor, y: Tensor, max_steps: int | None = None, rollout: str = "hard") -> dict:
        if rollout != "hard":
            raise ValueError('neo_repeat hands the grid between applications; evaluate with rollout "hard"')
        return super().self_explain(x, y, max_steps, rollout)

    def infer(self, support_x: Tensor, support_y: Tensor, query_x: Tensor, query_y: Tensor | None = None, max_steps: int | None = None, rollout: str = "hard") -> dict:
        if rollout != "hard":
            raise ValueError('neo_repeat hands the grid between applications; evaluate with rollout "hard"')
        return super().infer(support_x, support_y, query_x, query_y, max_steps, rollout)


MODELS["neo_repeat"] = NEORepeat
