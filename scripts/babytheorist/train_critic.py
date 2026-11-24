#!/usr/bin/env python3
"""Train a remaining-steps critic for a trained one-step model from its own dreams. No label is used.

The search (support_search.py) ranks partial programs by the programmer's preference alone, and with clean codes it
still reaches only about three quarters of what the hidden program, written in codes, reaches at length 8. In a dream
the number of steps left to the target is known exactly. This script dreams as methods/neo_dream.py does (codes the
programmer names on real pairs, in proportion to how often; the frozen executor; grids decoded between steps; steps
that do not change the grid or return to a passed grid are not counted) and trains a small network to predict, from a
grid and the target, how many grid-changing steps are left, 0 to --max-dream. The network has the programmer's form
(token_pair_transformer: the grid and the target joined cell by cell, a query token), smaller, with --max-dream + 1
outputs read as a distribution. The model itself is not changed.

    python scripts/babytheorist/train_critic.py runs/<run> --out runs/<run>/critic.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from core.registry import PROGRAMMERS, build, import_all  # noqa: E402
from training.evaluate import load_run  # noqa: E402


def critic_config(d_model: int, layers: int, max_dream: int, grid: int = 15) -> dict:
    return {"type": "token_pair_transformer", "state_dim": 128, "d_model": d_model, "action_dim": max_dream + 1, "num_heads": 8, "d_ff": 4 * d_model,
            "num_layers": layers, "grid_dim": grid, "dropout": 0.0, "layer_norm_eps": 1.0e-6}


@torch.no_grad()
def dream(model, x, y, table, usage, max_dream: int, min_share: float):
    """Grids along dreamed programs with the target and the number of grid-changing steps left."""

    size = table.shape[0]
    named = (F.normalize(model.programmer(model.encoder(x), model.encoder(y)).flatten(1).float(), dim=-1) @ table.T).argmax(dim=1)
    usage.mul_(0.99).add_(0.01 * torch.bincount(named, minlength=size).float() / len(x))
    used = torch.nonzero(usage >= min_share / size).flatten()
    n = len(x)
    lengths = torch.randint(1, max_dream + 1, (n,), device=x.device)
    programs = used[torch.multinomial(usage[used] / usage[used].sum(), n * max_dream, replacement=True)].view(n, max_dream)
    grids = [x]
    for t in range(max_dream):
        out = model.decoder(model.executor(model.encoder(grids[-1]), table[programs[:, t]][:, None])).argmax(dim=-1)
        grids.append(torch.where((t < lengths).view(-1, 1, 1), out.to(x.dtype), grids[-1]))
    target = grids[-1]
    moved = torch.stack([grids[t + 1].ne(grids[t]).flatten(1).any(dim=1) & (t < lengths) for t in range(max_dream)], dim=1)
    left = moved.flip(1).cumsum(1).flip(1)                     # grid-changing steps from position t to the end
    states, targets, labels = [], [], []
    for t in range(max_dream + 1):
        fresh = torch.ones(n, dtype=torch.bool, device=x.device)
        for later in range(t + 1, max_dream + 1):
            fresh &= ~(grids[later].eq(grids[t]).flatten(1).all(dim=1) & (later <= lengths) & grids[t].ne(target).flatten(1).any(dim=1))
        keep = (t <= lengths) & fresh & ((moved[:, t] if t < max_dream else torch.zeros_like(fresh)) | grids[t].eq(target).flatten(1).all(dim=1))
        rest = left[:, t] if t < max_dream else torch.zeros(n, dtype=torch.long, device=x.device)
        index = torch.nonzero(keep).flatten()
        states.append(grids[t][index])
        targets.append(target[index])
        labels.append(rest[index])
    return torch.cat(states), torch.cat(targets), torch.cat(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--max-dream", type=int, default=8)
    parser.add_argument("--min-share", type=float, default=0.25)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    import_all()
    model, data, config, step = load_run(args.run, None, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = critic_config(args.d_model, args.layers, args.max_dream)
    critic = build(PROGRAMMERS, dict(cfg)).to(device)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=0.01)
    table = model.code.codebook(torch.arange(model.code.size, device=device)).float()
    usage = torch.full((model.code.size,), 1.0 / model.code.size, device=device)
    loader = data.train_loader(args.batch, args.steps, 0, 0, 0, 1, 0, False)
    for i, batch in enumerate(loader):
        x, y = batch["x"].to(device), batch["y"].to(device)
        states, targets, labels = dream(model, x, y, table, usage, args.max_dream, args.min_share)
        pick = torch.randint(len(labels), (args.examples,), device=device)
        with torch.no_grad():
            s, t = model.encoder(states[pick]), model.encoder(targets[pick])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = critic(s, t).flatten(1).float()
        loss = F.cross_entropy(logits, labels[pick])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        optimizer.step()
        if i % 500 == 0:
            accuracy = (logits.argmax(dim=1) == labels[pick]).float().mean()
            print(f"step {i} loss {loss.item():.3f} exact {accuracy:.3f} mean left {labels[pick].float().mean():.2f}", flush=True)
        if i + 1 >= args.steps:
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "state": critic.state_dict(), "run": args.run.name, "steps": args.steps}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
