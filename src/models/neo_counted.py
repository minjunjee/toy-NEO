"""NEO whose code names a kind and a count, and the count is how often the kind's unit code is applied.
Registered as MODELS["neo_counted"].

It joins two things from this round. The codebook is the one indexed by kind g and integer amount t
(components/codes_orbit.py; with family "rotation" the entries are c(g, t) = exp(t A_g) b_g). The execution is the one
of models/neo_repeat.py: one code applied several times in a row with only the grid handed between applications. Here the
amount of the picked entry IS the number of applications:

  - the programmer looks at the pair and its output picks the nearest entry (g, t) over all kinds and amounts;
  - the executor is given the unit entry of that kind, c(g, 1), and applies it t times.

So "move up three cells" is written (up, 3) and executed as the code for one cell up, three times. That amounts add is
not asked by a loss; it holds because of how an entry is executed.

Training. The executor applies the unit of the picked kind `max_length` times and the lab's length rule finds the count
whose grid matches the target (as in neo_repeat). The reconstruction loss is taken at that count. The codebook loss and the
commitment loss pull the programmer's output and the entry (g, found count) together, so the programmer learns to name the
count that execution found. Evaluation with `count_from: programmer` (the default) uses the count the programmer names and
never looks at the target to choose it; `count_from: rule` uses the length rule, as neo_repeat does.

`max_length` must equal the largest amount and the amounts must be 1..max_length.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from components.codes import CodeOutput
from core.registry import MODELS
from models.neo import NEOOutput, mdl_select, select_by_length
from models.neo_repeat import NEORepeat
from models.neo_visible_state import NEOVisibleState

COUNT_FROM = ("programmer", "rule")


class NEOCounted(NEORepeat):
    def __init__(self, *args, count_from: str = "programmer", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if count_from not in COUNT_FROM:
            raise ValueError(f"count_from must be one of {COUNT_FROM}")
        if not hasattr(self.code, "kind_and_amount") or self.code.amounts.tolist() != list(range(1, self.max_length + 1)):
            raise ValueError("neo_counted needs a kind-and-amount codebook (vq_orbit) whose amounts are 1..max_length")
        self.count_from = count_from

    def pick(self, state: Tensor, target_state: Tensor, training: bool):
        """The picked entry, its kind and amount, and the unit code of that kind as the executor's input."""

        action = self.programmer(state, target_state)
        picked = self.code(action, training)
        kind, amount = self.code.kind_and_amount(picked.indices)
        unit = self.code.table()[self.code.index_of(kind, torch.ones_like(amount))].view(action.shape)
        normalized = F.normalize(action.float(), dim=-1)
        return action, picked, kind, amount, normalized + (unit - normalized).detach()  # the unit's value, the lab's gradient path

    def step(self, state: Tensor, target_state: Tensor, training: bool):
        """One application for the lab rollout (`replan`): the unit of the kind the programmer picks, applied once."""

        _, picked, _, _, unit = self.pick(state.detach(), target_state, training)
        return self.executor(state, unit), CodeOutput(unit, picked.loss_per_sample, picked.indices, picked.logits, picked.temperature)

    def forward(self, batch: dict, step: int = 0) -> NEOOutput:
        x, y = batch["x"], batch["y"]
        observations = torch.cat([x, y], dim=0)
        states = self.encoder(observations)
        auto_logits = self.decoder(states)
        state, target_state = states[: x.shape[0]], states[x.shape[0] :]
        action, picked, kind, amount, unit = self.pick(state.detach(), target_state, self.training)

        logits_by_length = []
        grounding = torch.zeros((), device=x.device, dtype=states.dtype)
        for _ in range(self.max_length):
            state = self.executor(state, unit)
            logits = self.decoder(state)
            logits_by_length.append(logits)
            with torch.no_grad():
                clean_state = self.encoder(logits.argmax(dim=-1))
            grounding = grounding + F.mse_loss(state, clean_state)
            if self.gradient_between:
                soft = torch.softmax(logits, dim=-1)
                visible = F.one_hot(logits.argmax(dim=-1), soft.shape[-1]).to(soft.dtype)
                state = self.encoder(visible + soft - soft.detach())
            else:
                state = clean_state

        if self.training:
            self.set_length_cost(step)
        found = mdl_select(logits_by_length, y, self.length_coefficient, self.length_cost_now or None)   # the count that execution found
        code_loss = self.code.loss_for(action, self.code.index_of(kind, found))
        self.named_right = (amount == found).float().mean().detach()   # how often the programmer named the count that was found
        return NEOOutput(
            logits_by_length, found, grounding, [code_loss] * self.max_length, [picked.indices] * self.max_length,
            picked.temperature, auto_logits, observations, y,
        )

    @torch.no_grad()
    def explain_mdl(self, x: Tensor, y: Tensor, max_steps: int, rollout: str):
        if rollout != "hard":
            raise ValueError('neo_counted hands the grid between applications; evaluate with rollout "hard"')
        if self.replan:
            return NEOVisibleState.explain_mdl(self, x, y, max_steps, rollout)
        if max_steps != self.max_length:
            raise ValueError("neo_counted is evaluated with max_steps equal to max_length")
        state, target_state = self.encoder(x), self.encoder(y)
        _, picked, _, amount, unit = self.pick(state, target_state, False)
        logits_by_length = []
        for _ in range(max_steps):
            logits = self.decoder(self.executor(state, unit))
            logits_by_length.append(logits)
            state = self.encoder(logits.argmax(dim=-1))
        lengths = amount if self.count_from == "programmer" else mdl_select(logits_by_length, y, self.length_coefficient, self.length_cost_now or None)
        return select_by_length(logits_by_length, lengths).argmax(dim=-1), lengths, [unit] * max_steps, [picked.indices] * max_steps


MODELS["neo_counted"] = NEOCounted
