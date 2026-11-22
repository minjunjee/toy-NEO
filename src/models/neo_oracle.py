"""NEO whose code for a training pair is the pair's hidden primitive. Registered as MODELS["neo_oracle"]. Diagnostic.

Code k is reserved for primitive k (data/babytheorist_labeled.py), so no code is ever shared by two primitives. The
executor is trained on the labelled code, and the lab VQ losses pull the programmer's output and code k toward each
other, so the programmer is trained to name the right code. Evaluation is the lab NEO's: the programmer names the
nearest code. Asked: in the 24-primitive environment every orientation primitive stays at 12-24% transfer whatever is
changed in the executor or the programmer; is the executor able to do them at all when each has a code of its own?
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from core.registry import MODELS
from models.neo import NEO, NEOOutput


class NEOOracle(NEO):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.max_length != 1:
            raise ValueError("neo_oracle is a one-step model: max_length must be 1")

    def forward(self, batch: dict, step: int = 0) -> NEOOutput:
        x, y = batch["x"], batch["y"]
        if "label" not in batch:
            raise ValueError("neo_oracle needs labelled pairs: data type babytheorist_pairs_labeled")
        label = batch["label"].to(x.device)
        observations = torch.cat([x, y], dim=0)
        states = self.encoder(observations)
        auto_logits = self.decoder(states)
        state, target_state = states[: x.shape[0]], states[x.shape[0]:]
        action = self.programmer(state.detach(), target_state)
        with torch.autocast(device_type=x.device.type, enabled=False):
            quantized = self.code.codebook(label)[:, None].float()
            normalized = F.normalize(action.float(), dim=-1)
            committed = (quantized.detach() - normalized) if self.code.commitment_grad else (quantized - normalized.detach())
            code_loss = self.code.commitment_cost * committed.square().mean(dim=(1, 2)) + (quantized - normalized.detach()).square().mean(dim=(1, 2))
        logits = self.decoder(self.executor(state, quantized.to(state.dtype)))
        return NEOOutput([logits], torch.ones(x.shape[0], dtype=torch.long, device=x.device), torch.zeros((), device=x.device),
                         [code_loss], [label], 0.0, auto_logits, observations, y)


MODELS["neo_oracle"] = NEOOracle
