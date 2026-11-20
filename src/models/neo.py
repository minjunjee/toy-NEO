"""NEO rollout: stepwise programmer -> code -> executor, MDL length selection, grounding.

Ported from learning-to-theorize ``models/neo.py`` and the paper evaluator rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.registry import CODES, DECODERS, ENCODERS, EXECUTORS, PROGRAMMERS, MODELS, build

STATE_UPDATES = ("carry_latent", "reencode_soft", "reencode_hard")
# official: paper evaluator (hard re-encode, stop at the first exact support match, query stops at its answer).
# carry / hard: MDL-selected support length, query always replays that many steps; carry keeps the
# training state_update between steps, hard re-encodes the argmax grid.
ROLLOUTS = ("official", "carry", "hard")


@dataclass
class NEOOutput:
    logits_by_length: list[Tensor]          # K x (B,H,W,C), decoded after k steps
    selected_lengths: Tensor                # (B,) in 1..K
    grounding: Tensor                       # sum_k mse(s_k, E(argmax D(s_k)))
    code_loss_by_length: list[Tensor]       # K x (B,)
    code_indices_by_length: list[Tensor | None]
    code_temperature: float
    auto_logits: Tensor                     # (2B,H,W,C) D(E(x)), D(E(y))
    observations: Tensor                    # (2B,H,W)
    targets: Tensor                         # (B,H,W)


def per_sample_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean over cells, one value per sample."""

    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1).long(), reduction="none")
    return ce.reshape(targets.shape[0], -1).mean(dim=1)


def mdl_select(logits_by_length: list[Tensor], targets: Tensor, coefficient: float, length_cost: float | None = None) -> Tensor:
    """Official rule: argmin_k coefficient**k * loss_k, then the shortest exact match overrides.
    With length_cost (nats per code): argmin_k of summed cell cross-entropy + length_cost * k, no exact override,
    so a longer exact rollout no longer beats a shorter one that misses a few cells."""

    with torch.no_grad():
        if length_cost is not None:
            cells = targets[0].numel()
            cost = torch.stack([per_sample_cross_entropy(logits, targets) * cells + length_cost * (k + 1)
                                for k, logits in enumerate(logits_by_length)], dim=1)
            return cost.argmin(dim=1) + 1
        weighted = torch.stack(
            [per_sample_cross_entropy(logits, targets) * coefficient ** (k + 1) for k, logits in enumerate(logits_by_length)],
            dim=1,
        )
        selected = weighted.argmin(dim=1) + 1
        for length in range(len(logits_by_length), 0, -1):
            correct = logits_by_length[length - 1].argmax(dim=-1).eq(targets).flatten(1).all(dim=1)
            selected[correct] = length
    return selected


def select_by_length(values_by_length: list[Tensor], lengths: Tensor) -> Tensor:
    """Pick values_by_length[lengths[b] - 1][b] for every sample b."""

    stacked = torch.stack(values_by_length, dim=1)
    return stacked[torch.arange(len(lengths), device=lengths.device), lengths - 1]


class NEO(nn.Module):
    def __init__(
        self,
        encoder: dict,
        decoder: dict,
        programmer: dict,
        code: dict,
        executor: dict,
        max_length: int,
        length_coefficient: float,
        state_update: str = "carry_latent",
        length_cost: float | None = None,
        length_cost_ramp: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if state_update not in STATE_UPDATES:
            raise ValueError(f"state_update must be one of {STATE_UPDATES}")
        self.encoder = build(ENCODERS, encoder)
        self.decoder = build(DECODERS, decoder)
        self.programmer = build(PROGRAMMERS, programmer)
        self.code = build(CODES, code)
        self.executor = build(EXECUTORS, executor)
        self.max_length = max_length
        self.length_coefficient = length_coefficient
        self.length_cost = length_cost
        # [start, end] training steps: the per-code cost is 0 (official rule) before start and grows linearly to
        # length_cost at end; evaluation uses the value reached at the last training step
        self.length_cost_ramp = tuple(length_cost_ramp) if length_cost_ramp else None
        self.length_cost_now = length_cost
        self.state_update = state_update

    def set_length_cost(self, step: int) -> None:
        if self.length_cost_ramp:
            start, end = self.length_cost_ramp
            self.length_cost_now = self.length_cost * min(1.0, max(0.0, (step - start) / max(1, end - start)))

    # ---- pieces reused by rollouts -------------------------------------------------
    def step(self, state: Tensor, target_state: Tensor, training: bool):
        action = self.programmer(state.detach(), target_state)
        code = self.code(action, training)
        return self.executor(state, code.values), code

    def decode_grid(self, state: Tensor) -> Tensor:
        return self.decoder(state).argmax(dim=-1)

    def rollout_state(self, state: Tensor, logits: Tensor, rollout: str) -> Tensor:
        if rollout == "hard" or self.state_update == "reencode_hard":
            return self.encoder(logits.argmax(dim=-1))
        if self.state_update == "reencode_soft":
            return self.encoder(torch.softmax(logits, dim=-1))
        return state

    @torch.no_grad()
    def explain_mdl(self, x: Tensor, y: Tensor, max_steps: int, rollout: str):
        """Roll out max_steps codes toward y, then pick the length by the training MDL rule."""

        if rollout not in ROLLOUTS[1:]:
            raise ValueError(f"rollout must be one of {ROLLOUTS}")
        state, target_state = self.encoder(x), self.encoder(y)
        logits_by_length, actions, indices = [], [], []
        for _ in range(max_steps):
            state, code = self.step(state, target_state, training=False)
            logits = self.decoder(state)
            logits_by_length.append(logits)
            actions.append(code.values)
            indices.append(code.indices)
            state = self.rollout_state(state, logits, rollout)
        lengths = mdl_select(logits_by_length, y, self.length_coefficient, self.length_cost_now or None)
        return select_by_length(logits_by_length, lengths).argmax(dim=-1), lengths, actions, indices

    # ---- training forward ---------------------------------------------------------
    def forward(self, batch: dict, step: int = 0) -> NEOOutput:
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
            if self.state_update == "reencode_hard":
                state = clean_state
            elif self.state_update == "reencode_soft":
                state = self.encoder(torch.softmax(logits, dim=-1))

        if self.training:
            self.set_length_cost(step)
        selected = mdl_select(logits_by_length, y, self.length_coefficient, self.length_cost_now or None)
        return NEOOutput(
            logits_by_length, selected, grounding, code_loss_by_length, code_indices_by_length,
            temperature, auto_logits, observations, y,
        )

    # ---- inference (paper evaluator protocol, vectorized) --------------------------
    @torch.no_grad()
    def self_explain(self, x: Tensor, y: Tensor, max_steps: int | None = None, rollout: str = "official") -> dict:
        """Infer actions toward y from x, re-encoding the decoded grid after every step."""

        max_steps = max_steps or self.max_length
        if rollout != "official":
            pred, lengths, _, indices = self.explain_mdl(x, y, max_steps, rollout)
            return {"pred": pred, "n_steps": lengths, "indices": indices}
        state, target_state = self.encoder(x), self.encoder(y)
        pred = x.clone()
        done = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        steps = torch.full((x.shape[0],), max_steps, dtype=torch.long, device=x.device)
        indices = []
        for k in range(max_steps):
            state, code = self.step(state, target_state, training=False)
            indices.append(code.indices)
            new_pred = self.decode_grid(state)
            pred = torch.where(done.view(-1, 1, 1), pred, new_pred)
            state = self.encoder(pred)
            matched = new_pred.eq(y).flatten(1).all(dim=1) & ~done
            steps[matched] = k + 1
            done |= matched
        return {"pred": pred, "n_steps": steps, "indices": indices}

    @torch.no_grad()
    def infer(self, support_x: Tensor, support_y: Tensor, query_x: Tensor, query_y: Tensor | None = None, max_steps: int | None = None, rollout: str = "official") -> dict:
        """Extract the support action sequence and replay it on the query (transfer)."""

        max_steps = max_steps or self.max_length
        if rollout != "official":
            _, lengths, actions, indices = self.explain_mdl(support_x, support_y, max_steps, rollout)
            state, pred = self.encoder(query_x), query_x.clone()
            for k, action in enumerate(actions):
                active = (k < lengths).view(-1, 1, 1)
                next_state = self.executor(state, action)
                logits = self.decoder(next_state)
                pred = torch.where(active, logits.argmax(dim=-1), pred)
                state = torch.where(active, self.rollout_state(next_state, logits, rollout), state)
            return {"pred": pred, "n_extracted": lengths, "n_applied": lengths, "indices": indices}
        batch = support_x.shape[0]
        state, target_state = self.encoder(support_x), self.encoder(support_y)
        done = torch.zeros(batch, dtype=torch.bool, device=support_x.device)
        n_extracted = torch.full((batch,), max_steps, dtype=torch.long, device=support_x.device)
        actions, indices = [], []
        for k in range(max_steps):
            state, code = self.step(state, target_state, training=False)
            actions.append(code.values)
            indices.append(code.indices)
            support_pred = self.decode_grid(state)
            state = self.encoder(support_pred)
            matched = support_pred.eq(support_y).flatten(1).all(dim=1) & ~done
            n_extracted[matched] = k + 1
            done |= matched

        state = self.encoder(query_x)
        pred = query_x.clone()
        query_done = torch.zeros(batch, dtype=torch.bool, device=support_x.device)
        n_applied = torch.zeros(batch, dtype=torch.long, device=support_x.device)
        for k, action in enumerate(actions):
            active = (k < n_extracted) & ~query_done
            if not active.any():
                break
            new_state = self.executor(state, action)
            new_pred = self.decode_grid(new_state)
            pred = torch.where(active.view(-1, 1, 1), new_pred, pred)
            state = torch.where(active.view(-1, 1, 1), self.encoder(new_pred), state)
            n_applied += active.long()
            if query_y is not None:
                query_done |= active & new_pred.eq(query_y).flatten(1).all(dim=1)
        return {"pred": pred, "n_extracted": n_extracted, "n_applied": n_applied, "indices": indices}


MODELS["neo"] = NEO
