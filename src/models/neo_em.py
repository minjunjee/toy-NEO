"""NEO trained by soft EM over the codes, with an annealed temperature. Registered as MODELS["neo_em"]. One program step.

Why. In the 24-primitive environment four orientation primitives share codes and stay mixed; every code was tried by
execution and none reproduces rot90 on more than 10% of its pairs, so choosing a single winner (models/neo_compete.py)
had nothing to choose. The code of a pair is a hidden variable; this model treats it as one.

E step (no gradient). For every pair, the executor runs every code (or `candidates` of them: the programmer's nearest
plus random ones). Each code gets a responsibility proportional to exp(-L_k / T), where L_k is the pair's summed cell
cross-entropy under code k. T falls geometrically from `temperature_start` to `temperature_end` over `anneal_steps`
training steps: at a high temperature a group of pairs is shared by many codes, and as it falls a group that is really
two groups splits between codes (deterministic annealing).

M step. The executor is trained on the `train_top` codes with the largest responsibility, each pair's loss weighted by the
renormalised responsibilities, so a code that is a little less wrong on a pair takes a little more of it. The code vectors
learn through the executor (they are its conditioning input). The programmer is trained to predict the responsibilities:
cross-entropy between them and a softmax over its cosine similarity to every code (scale `programmer_scale`). Evaluation
is the lab NEO's: the programmer names the nearest code, nothing is searched.

Hard EM is the same model with a temperature near zero and `train_top` 1: each pair goes to the code that reproduces it
best, the executor learns that code on it, and the programmer learns to name it.

Rescue of dead codes (`dead_share` > 0). A code whose moving share of first places (decay `usage_decay`) is below
`dead_share` times the uniform share 1/size is dead. Each training step a dead code's loss in the E step is lowered by a
further `bonus_step` nats, up to `bonus_cap`; the bonus returns to zero as soon as the code is used again. Codes in use get
no bonus, so the rescue only revives codes that nothing chooses and does not take part in the competition between used
codes. Exact ties in the E step (at the start every code executes the same, since the executor's modulation starts at
zero) are broken by a jitter of `jitter` nats.

Tasks of two pairs (batches with x2, y2 from the babytheorist_episodes dataset). A code's loss in the E step is its loss
on the first pair plus its loss on the second, so a code is chosen for what it does on both inputs, and the executor is
trained on both pairs with the chosen codes; the programmer still names the code from the first pair. Without x2 nothing
changes.

Selection by fit (`keep_fraction` < 1). Train files mix programs of one and of several primitives, and a one-step model
can reproduce only the first kind; the earlier runs kept them apart with the stored depth, which a learner is not given.
Here every task's smallest E-step loss (over the codes tried) ranks it, and only the `keep_fraction` of the batch that
the best code fits most closely enters the M step and the programmer's loss. A task that no single code can explain fits
badly and is left out, without reading its depth.

Prices for spreading early (`price_weight` > 0). Early in training the executor cannot tell similar primitives apart, so
every code fits their tasks about equally and whichever code wins first keeps winning: hard EM settles which tasks share a
code before the executor can execute the difference (seen in the 100-step codebook records of 2026-11-26). Here every code
carries a price in the E step, `price_weight` nats times (its moving share of first places times the codebook size, minus
one): a code used k times more than uniform costs (k - 1) times the weight, an unused one is a little cheaper. Tasks that
fit many codes about equally then spread over them instead of piling onto one, so codes are over-separated early. The
weight holds for `price_hold` steps and falls linearly to zero at `price_off`; after that hard EM's own pull towards the
winning code works again, and by then it can only join codes that the executor runs the same way. This changes how
tasks are assigned, not the losses the parts learn from. Off by default.

No label is used. With two-pair tasks the grouping of pairs is used, never the program.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from core.registry import MODELS
from models.neo import NEO, NEOOutput


class NEOEM(NEO):
    def __init__(self, *args, candidates: int = 0, train_top: int = 4, temperature_start: float = 10.0, temperature_end: float = 0.1,
                 anneal_steps: int = 10000, programmer_scale: float = 10.0, dead_share: float = 0.0, usage_decay: float = 0.99,
                 bonus_step: float = 0.05, bonus_cap: float = 20.0, jitter: float = 1.0e-3, keep_fraction: float = 1.0,
                 price_weight: float = 0.0, price_hold: int = 3000, price_off: int = 8000, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.max_length != 1:
            raise ValueError("neo_em is a one-step model: max_length must be 1")
        self.candidates, self.train_top = candidates, train_top
        self.temperature_start, self.temperature_end, self.anneal_steps = temperature_start, temperature_end, anneal_steps
        self.programmer_scale = programmer_scale
        self.dead_share, self.usage_decay, self.bonus_step, self.bonus_cap, self.jitter = dead_share, usage_decay, bonus_step, bonus_cap, jitter
        self.keep_fraction = keep_fraction
        self.price_weight, self.price_hold, self.price_off = price_weight, price_hold, price_off
        size = self.code.size
        self.register_buffer("usage", torch.full((size,), 1.0 / size), persistent=False)
        self.register_buffer("bonus", torch.zeros(size), persistent=False)
        self.last = {}

    def price_scale(self, step: int) -> float:
        if step <= self.price_hold:
            return self.price_weight
        return self.price_weight * max(0.0, 1.0 - (step - self.price_hold) / max(1, self.price_off - self.price_hold))

    def temperature(self, step: int) -> float:
        fraction = min(1.0, step / max(1, self.anneal_steps))
        return math.exp(math.log(self.temperature_start) + fraction * (math.log(self.temperature_end) - math.log(self.temperature_start)))

    def forward(self, batch: dict, step: int = 0) -> NEOOutput:
        x, y = batch["x"], batch["y"]
        observations = torch.cat([x, y], dim=0)
        states = self.encoder(observations)
        auto_logits = self.decoder(states)
        state, target_state = states[: x.shape[0]], states[x.shape[0]:]
        batch_size, size = x.shape[0], self.code.size
        action = self.programmer(state.detach(), target_state)
        table = self.code.codebook(torch.arange(size, device=x.device))                          # (size, dim), trainable
        with torch.autocast(device_type=x.device.type, enabled=False):
            similarity = F.normalize(action.float().flatten(1), dim=-1) @ F.normalize(table.float(), dim=-1).T   # (B, size)
        if self.candidates and self.candidates < size:
            near = similarity.detach().topk(self.candidates // 2, dim=1).indices
            rand = torch.randint(size, (batch_size, self.candidates - near.shape[1]), device=x.device)
            tried = torch.cat([near, rand], dim=1)
        else:
            tried = torch.arange(size, device=x.device).expand(batch_size, size)
        m = tried.shape[1]
        # E step: every tried code executed, no gradient
        pairs = [(state, y)]
        if "x2" in batch:
            pairs.append((self.encoder(batch["x2"]), batch["y2"]))
        with torch.no_grad():
            loss = 0
            for source, goal in pairs:  # summed over the task's pairs
                logits = self.decoder(self.executor(source.repeat_interleave(m, dim=0), table.detach()[tried.flatten()][:, None].to(source.dtype)))
                part = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), goal.repeat_interleave(m, dim=0).reshape(-1).long(), reduction="none")
                loss = loss + part.view(batch_size, m, -1).sum(dim=2)                              # summed over cells, nats
            if self.training:
                loss = loss - self.bonus[tried] + self.jitter * torch.rand_like(loss)
                if self.price_weight > 0:
                    loss = loss + self.price_scale(step) * (self.usage * size - 1.0)[tried]
            temperature = self.temperature(step) if self.training else self.temperature_end
            weights = torch.softmax(-loss / temperature, dim=1)                                     # (B, m)
            fit = loss.min(dim=1).values                                                           # how well the best code explains the task
            kept = torch.ones(batch_size, dtype=torch.bool, device=x.device)
            if self.training and self.keep_fraction < 1.0:
                kept = fit <= torch.quantile(fit.float(), self.keep_fraction)
            responsibility = torch.zeros(batch_size, size, device=x.device).scatter_add_(1, tried, weights)
        # M step: the executor on the top codes, weighted
        top = responsibility.topk(min(self.train_top, m), dim=1)
        share = top.values / top.values.sum(dim=1, keepdim=True)
        k = top.indices.shape[1]
        recon = 0
        for i, (source, goal) in enumerate(pairs):
            decoded = self.decoder(self.executor(source.repeat_interleave(k, dim=0), table[top.indices.flatten()][:, None].to(source.dtype)))
            part = F.cross_entropy(decoded.reshape(-1, decoded.shape[-1]).float(), goal.repeat_interleave(k, dim=0).reshape(-1).long(), reduction="none")
            recon = recon + (part.view(batch_size, k, -1).mean(dim=2) * share).sum(dim=1) / len(pairs)   # per task
            if i == 0:
                out = decoded
        # the programmer learns to predict the responsibilities
        programmer_loss = -(responsibility * F.log_softmax(self.programmer_scale * similarity, dim=1)).sum(dim=1)
        keep = kept.float() / kept.float().mean().clamp(min=1e-6)                                  # dropped tasks weigh nothing
        recon, programmer_loss = recon * keep, programmer_loss * keep
        best = out.view(batch_size, k, *out.shape[1:])[:, 0]
        if self.training:
            with torch.no_grad():
                firsts = torch.bincount(top.indices[kept, 0], minlength=size).float() / kept.sum().clamp(min=1)
                self.usage.mul_(self.usage_decay).add_((1 - self.usage_decay) * firsts)
                dead = self.usage < self.dead_share / size
                self.bonus.copy_(torch.where(dead, (self.bonus + self.bonus_step).clamp(max=self.bonus_cap), torch.zeros_like(self.bonus)))
        self.last = {"em_recon": recon.mean().detach(), "em_programmer": programmer_loss.mean().detach(), "em_temperature": temperature,
                     "em_entropy": -(responsibility.clamp_min(1e-9).log() * responsibility).sum(dim=1).mean(),
                     "em_programmer_agrees": (similarity.argmax(dim=1) == top.indices[:, 0]).float().mean(),
                     "em_dead": (self.usage < max(self.dead_share, 0.1) / size).float().sum(), "em_bonus_max": self.bonus.max(),
                     "em_price": self.price_scale(step) if self.price_weight > 0 else 0.0, "em_usage_max": self.usage.max() * size,
                     "em_fit_kept": fit[kept].float().mean(), "em_fit_dropped": fit[~kept].float().mean() if (~kept).any() else fit.new_zeros(())}
        return NEOOutput([best], torch.ones(batch_size, dtype=torch.long, device=x.device), torch.zeros((), device=x.device),
                         [recon + programmer_loss], [top.indices[:, 0]], 0.0, auto_logits, observations, y)


MODELS["neo_em"] = NEOEM
