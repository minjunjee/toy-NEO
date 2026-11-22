"""NEO whose code for a training pair is chosen by executing candidates, not by the programmer alone.
Registered as MODELS["neo_compete"]. One program step only (max_length 1).

What it fixes. In the 24-primitive environment the seven orientation primitives fall to 12-24% transfer. A linear probe
on the programmer's output tells those seven apart on only 38% of pairs (97% for the three of them in the 8-primitive
environment), and rot90, rot270, transpose and anti_transpose are all sent to the same two codes. The executor then gets
one code for four different transformations and can fit none of them; giving every code its own executor network does
not help (docs/reports/2026-11-22-scaling.html), because the mixing happens before the executor, in the choice of code.

How. For each training pair the programmer's output names `candidates` codes, the nearest ones in the codebook. The
executor runs every candidate on the pair's state; the candidate whose decoded grid is closest to the next state (lowest
cell cross-entropy) wins. Only the winner's output is trained to reproduce the pair (so only the winner's behaviour is
shaped by it), and the lab VQ losses pull the programmer's output and the winning code toward each other, so the
programmer learns to name the code that execution chose. The choice is a discrete decision made by checking, the values
are trained by gradient. Evaluation is the lab NEO's: the programmer names the nearest code and nothing is searched, so
the numbers say how well the programmer has learned to imitate the checked choice.

This is the competitive scheme of Parascandolo et al., Learning Independent Causal Mechanisms (ICML 2018), where
experts compete for each transformed example and the best one is trained on it, with NEO's programmer added to learn
the outcome of the competition.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from core.registry import MODELS
from models.neo import NEO, NEOOutput


class NEOCompete(NEO):
    def __init__(self, *args, candidates: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.max_length != 1:
            raise ValueError("neo_compete is the one-step model: max_length must be 1")
        self.candidates = candidates
        if hasattr(self.executor, "table") and getattr(self.executor, "size", None) == self.code.size:   # a per-code executor
            code = self.code
            object.__setattr__(self.executor, "table", lambda: code.codebook(torch.arange(code.size, device=next(code.parameters()).device)))

    def forward(self, batch: dict, step: int = 0) -> NEOOutput:
        x, y = batch["x"], batch["y"]
        observations = torch.cat([x, y], dim=0)
        states = self.encoder(observations)
        auto_logits = self.decoder(states)
        state, target_state = states[: x.shape[0]], states[x.shape[0]:]
        action = self.programmer(state.detach(), target_state)
        batch_size, m = x.shape[0], min(self.candidates, self.code.size)
        with torch.autocast(device_type=x.device.type, enabled=False):
            table = self.code.codebook(torch.arange(self.code.size, device=x.device)).float()
            normalized = F.normalize(action.float(), dim=-1)                                       # (B, 1, dim)
            candidates = (normalized.flatten(1) @ table.T).topk(m, dim=1).indices                 # (B, m), nearest first
        # run every candidate; no gradient here, the choice is made by checking
        with torch.no_grad():
            tried = self.executor(state.repeat_interleave(m, dim=0), table[candidates.flatten()][:, None].to(state.dtype))
            logits = self.decoder(tried)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.repeat_interleave(m, dim=0).reshape(-1).long(), reduction="none")
            loss = loss.view(batch_size, m, -1).mean(dim=2)
            winner = candidates[torch.arange(batch_size, device=x.device), loss.argmin(dim=1)]     # ties go to the nearest
        quantized = table[winner][:, None]                                                         # (B, 1, dim)
        with torch.autocast(device_type=x.device.type, enabled=False):
            committed = (quantized.detach() - normalized) if self.code.commitment_grad else (quantized - normalized.detach())
            code_loss = self.code.commitment_cost * committed.square().mean(dim=(1, 2)) + (quantized - normalized.detach()).square().mean(dim=(1, 2))
        state = self.executor(state, quantized.to(state.dtype))
        logits = self.decoder(state)
        self.code.last_winner_is_nearest = (winner == candidates[:, 0]).float().mean().detach()
        return NEOOutput([logits], torch.ones(batch_size, dtype=torch.long, device=x.device), torch.zeros((), device=x.device),
                         [code_loss], [winner], 0.0, auto_logits, observations, y)


MODELS["neo_compete"] = NEOCompete
