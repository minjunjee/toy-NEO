"""Dump official NEO (random init, paper alpha-0.33 config) outputs for parity: needs L2T_OFFICIAL_NEO=<repo>."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

official = Path(os.environ["L2T_OFFICIAL_NEO"]).resolve()
sys.path.insert(0, str(official / "src"))

from tasks.gridworld.task import build_neo  # noqa: E402
from tasks.gridworld.theorizer_evaluation import self_explanation_rollout, transfer_rollout  # noqa: E402

torch.manual_seed(0)
model = build_neo("alpha-0.33").eval()
# Randomize BatchNorm running stats so the frozen codec is not the identity-ish init.
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.running_mean.normal_(0, 0.5)
        module.running_var.uniform_(0.5, 1.5)
g = torch.Generator().manual_seed(1)
episodes = torch.randint(0, 9, (6, 4, 10, 10), generator=g)
with torch.no_grad():
    x, y = episodes[:, 0], episodes[:, 1]
    state, _ = model.encoder(x)
    target, _ = model.encoder(y)
    action = model.theory_programmer(state, target)
    quantized = model.quantizer(action, training_mode=False)
    next_state = model.program_executor(state, quantized.values)
    logits = model.decoder(next_state)
    transfer = [transfer_rollout(model, e[0], e[1], e[2], e[3], max_steps=4, device=torch.device("cpu")) for e in episodes]
    selfx = [self_explanation_rollout(model, e[0], e[1], max_steps=4, device=torch.device("cpu")) for e in episodes]
torch.save(
    {
        "state_dict": model.state_dict(),
        "episodes": episodes,
        "state": state, "target": target, "action": action, "indices": quantized.indices, "values": quantized.values,
        "next_state": next_state, "logits": logits,
        "transfer_pred": torch.stack([t.final_prediction for t in transfer]),
        "transfer_extracted": torch.tensor([t.num_extracted_steps for t in transfer]),
        "transfer_applied": torch.tensor([t.num_applied_steps for t in transfer]),
        "self_pred": torch.stack([s.final_prediction for s in selfx]),
        "self_steps": torch.tensor([s.num_steps for s in selfx]),
    },
    Path(__file__).with_name("neo_official.pt"),
)
print("wrote neo_official.pt")
