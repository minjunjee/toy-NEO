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

No label and no grouping of pairs is used.
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
                 bonus_step: float = 0.05, bonus_cap: float = 20.0, jitter: float = 1.0e-3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.max_length != 1:
            raise ValueError("neo_em is a one-step model: max_length must be 1")
        self.candidates, self.train_top = candidates, train_top
        self.temperature_start, self.temperature_end, self.anneal_steps = temperature_start, temperature_end, anneal_steps
        self.programmer_scale = programmer_scale
        self.dead_share, self.usage_decay, self.bonus_step, self.bonus_cap, self.jitter = dead_share, usage_decay, bonus_step, bonus_cap, jitter
        size = self.code.size
        self.register_buffer("usage", torch.full((size,), 1.0 / size), persistent=False)
        self.register_buffer("bonus", torch.zeros(size), persistent=False)
        self.last = {}

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
        with torch.no_grad():
            logits = self.decoder(self.executor(state.repeat_interleave(m, dim=0), table.detach()[tried.flatten()][:, None].to(state.dtype)))
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.repeat_interleave(m, dim=0).reshape(-1).long(), reduction="none")
            loss = loss.view(batch_size, m, -1).sum(dim=2)                                         # summed over cells, nats
            if self.training:
                loss = loss - self.bonus[tried] + self.jitter * torch.rand_like(loss)
            temperature = self.temperature(step) if self.training else self.temperature_end
            weights = torch.softmax(-loss / temperature, dim=1)                                     # (B, m)
            responsibility = torch.zeros(batch_size, size, device=x.device).scatter_add_(1, tried, weights)
        # M step: the executor on the top codes, weighted
        top = responsibility.topk(min(self.train_top, m), dim=1)
        share = top.values / top.values.sum(dim=1, keepdim=True)
        k = top.indices.shape[1]
        out = self.decoder(self.executor(state.repeat_interleave(k, dim=0), table[top.indices.flatten()][:, None].to(state.dtype)))
        recon = F.cross_entropy(out.reshape(-1, out.shape[-1]).float(), y.repeat_interleave(k, dim=0).reshape(-1).long(), reduction="none")
        recon = (recon.view(batch_size, k, -1).mean(dim=2) * share).sum(dim=1)                     # per pair
        # the programmer learns to predict the responsibilities
        programmer_loss = -(responsibility * F.log_softmax(self.programmer_scale * similarity, dim=1)).sum(dim=1)
        best = out.view(batch_size, k, *out.shape[1:])[:, 0]
        if self.training:
            with torch.no_grad():
                firsts = torch.bincount(top.indices[:, 0], minlength=size).float() / batch_size
                self.usage.mul_(self.usage_decay).add_((1 - self.usage_decay) * firsts)
                dead = self.usage < self.dead_share / size
                self.bonus.copy_(torch.where(dead, (self.bonus + self.bonus_step).clamp(max=self.bonus_cap), torch.zeros_like(self.bonus)))
        self.last = {"em_recon": recon.mean().detach(), "em_programmer": programmer_loss.mean().detach(), "em_temperature": temperature,
                     "em_entropy": -(responsibility.clamp_min(1e-9).log() * responsibility).sum(dim=1).mean(),
                     "em_programmer_agrees": (similarity.argmax(dim=1) == top.indices[:, 0]).float().mean(),
                     "em_dead": (self.usage < max(self.dead_share, 0.1) / size).float().sum(), "em_bonus_max": self.bonus.max()}
        return NEOOutput([best], torch.ones(batch_size, dtype=torch.long, device=x.device), torch.zeros((), device=x.device),
                         [recon + programmer_loss], [top.indices[:, 0]], 0.0, auto_logits, observations, y)


MODELS["neo_em"] = NEOEM
