#!/usr/bin/env python3
"""An offline phase applied once to a trained NEO: merge codes that carry the same primitive, repair by replay, keep the
merge only if stored experience is still reproduced.

Why: in a trained transformer NEO a true primitive is often split across two or three codes (one code per group of
colours, for example). A program taken from the support pair then names a code that does not work on the query object.

What it does, with nothing read from the hidden programs:
1. Replay memory. A sample of training pairs is explained by the model as it is; pairs it reproduces exactly are stored
   with their code sequences.
2. Candidates. Pairs of codes, most similar code embeddings first.
3. For each candidate (a, b): code b is given a's embedding and every stored b becomes a. The executor, the codebook and
   the programmer are then trained for a few hundred steps on the stored pairs with their stored programs (the executor
   to reproduce y, the programmer to name the stored code). The merge is kept if the stored programs of a held-out part
   of the memory are reproduced as often as before the merge, within a margin; otherwise everything is restored.
4. The model is evaluated before and after, and the compute spent is counted in floating-point operations.

    python scripts/babytheorist/merge_sleep.py --run runs/<run> [--ckpt ...] --split lv1 --out docs/reports/<name>/merge_sleep.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.babytheorist import TrainPairs  # noqa: E402
from evals.babytheorist import babytheorist_observe  # noqa: E402
from training.evaluate import load_run  # noqa: E402
from transfer_breakdown import GROUPS, breakdown_model  # noqa: E402


def replay_programs(model, x, y, codes, lengths, with_programmer: bool = False):
    """Execute stored code sequences from x with the state update used in training. Returns the logits after the last
    stored code and, if asked, how far the programmer's action is from the stored code at every step."""

    state, target = model.encoder(x), model.encoder(y)
    final = None
    distance = x.new_zeros((), dtype=torch.float32)
    for k in range(codes.shape[1]):
        active = k < lengths
        if not bool(active.any()):
            break
        values = model.code.codebook(codes[:, k])
        if with_programmer:
            action = model.programmer(state.detach(), target)
            miss = 1.0 - F.cosine_similarity(action.reshape(len(x), -1), values.detach().reshape(len(x), -1), dim=-1)
            distance = distance + (miss * active).sum() / active.sum().clamp(min=1)
        next_state = model.executor(state, values.view(len(x), model.code.num_tokens, -1))
        logits = model.decoder(next_state)
        final = logits if final is None else torch.where(active.view(-1, 1, 1, 1), logits, final)
        state = torch.where(active.view(-1, 1, 1), model.rollout_state(next_state, logits, "carry"), state)
    return final, distance


@torch.no_grad()
def reproduced(model, memory: dict, rows: torch.Tensor, device, batch: int = 500) -> float:
    hits = 0
    for start in range(0, len(rows), batch):
        index = rows[start:start + batch]
        x, y = memory["x"][index].to(device).long(), memory["y"][index].to(device).long()
        logits, _ = replay_programs(model, x, y, memory["codes"][index].to(device), memory["lengths"][index].to(device))
        hits += int(logits.argmax(-1).eq(y).flatten(1).all(1).sum())
    return hits / max(len(rows), 1)


@torch.no_grad()
def build_memory(model, pairs: TrainPairs, count: int, infer: dict, device, seed: int) -> dict:
    chosen = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed))[:count]
    keep_x, keep_y, keep_codes, keep_lengths = [], [], [], []
    steps = infer.get("max_steps") or model.max_length
    for start in range(0, count, 500):
        index = chosen[start:start + 500]
        x, y = pairs.x[index].to(device).long(), pairs.y[index].to(device).long()
        out = model.self_explain(x, y, **infer)
        exact = out["pred"].eq(y).flatten(1).all(1)
        codes = torch.stack([i.view(len(x), -1)[:, 0] for i in out["indices"]], dim=1)[:, :steps]
        keep_x.append(pairs.x[index][exact.cpu()])
        keep_y.append(pairs.y[index][exact.cpu()])
        keep_codes.append(codes[exact].cpu())
        keep_lengths.append(out["n_steps"][exact].cpu())
    return {"x": torch.cat(keep_x), "y": torch.cat(keep_y), "codes": torch.cat(keep_codes), "lengths": torch.cat(keep_lengths), "sampled": count}


def evaluate(model, data, split: str, infer: dict, device, samples: int) -> dict:
    observed = babytheorist_observe(model, data, {"split": split, "batch_size": 500, "max_samples": samples, "tables": False, "infer": infer}, 0, device)
    groups = breakdown_model(model, data, split, infer, device)["all"]
    total = sum(v[1] for v in groups.values())
    return {"explain": observed["explain"], "transfer_strict": observed["transfer_strict"], "transfer_strict_depth1": observed["transfer_strict_depth1"],
            "single_primitive_tasks": {g: {"share": groups[g][1] / max(total, 1), "transferred": groups[g][0] / max(groups[g][1], 1)} for g in GROUPS}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--memory", type=int, default=30000, help="training pairs sampled for the replay memory")
    parser.add_argument("--holdout", type=float, default=0.15)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--min-use", type=int, default=50, help="a code must appear this often in the memory to be a candidate")
    parser.add_argument("--repair-steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, help="default 1e-2 for --repair embedding, 3e-5 for --repair all")
    parser.add_argument("--programmer-weight", type=float, default=1.0)
    parser.add_argument("--repair", choices=("embedding", "all"), default="embedding",
                        help="embedding: only the merged code's vector is fitted, on the stored pairs that use it, everything else frozen; all: executor, codebook and programmer are trained on the whole memory")
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--eval-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save", type=Path, help="write the model after the phase here")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.lr = args.lr or (1.0e-2 if args.repair == "embedding" else 3.0e-5)
    torch.manual_seed(args.seed)
    model, data, config, step = load_run(args.run, args.ckpt, device)
    infer = next((dict(b.get("infer", {})) for b in config["evals"] if b["type"] == "babytheorist_observe"), {})
    report: dict = {"run": args.run.name, "step": step, "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    report["before"] = evaluate(model, data, args.split, infer, device, args.eval_samples)
    print("before", json.dumps(report["before"]), flush=True)

    memory = build_memory(model, TrainPairs(data.train_paths, data.train_max_depth), args.memory, infer, device, args.seed)
    stored = len(memory["x"])
    order = torch.randperm(stored, generator=torch.Generator().manual_seed(args.seed + 1))
    held, train = order[: int(stored * args.holdout)], order[int(stored * args.holdout):]
    flops = {"memory": 0.0, "repair": 0.0, "checks": 0.0}
    probe = min(16, len(train))
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        index = train[:probe]
        replay_programs(model, memory["x"][index].to(device).long(), memory["y"][index].to(device).long(), memory["codes"][index].to(device), memory["lengths"][index].to(device), with_programmer=True)
    step_forward = float(counter.get_total_flops()) / probe * args.batch  # one repair batch forward, executor and programmer
    flops["memory"] = step_forward / args.batch * args.memory  # explaining the sampled pairs costs about one such forward per pair
    usage = torch.bincount(memory["codes"][torch.arange(memory["codes"].shape[1])[None] < memory["lengths"][:, None]], minlength=model.code.size)
    report["memory"] = {"sampled": args.memory, "stored": stored, "held_out": len(held), "code_usage": usage.tolist(), "reproduced_held_out": reproduced(model, memory, held, device)}
    print("memory", json.dumps(report["memory"]), flush=True)

    embedding = F.normalize(model.code.embedding.weight.detach().float(), dim=-1)
    similarity = embedding @ embedding.T
    used = [c for c in range(model.code.size) if int(usage[c]) >= args.min_use]
    pairs = sorted(((float(similarity[a, b]), a, b) for i, a in enumerate(used) for b in used[i + 1:]), reverse=True)[: args.candidates]
    alias = {c: c for c in range(model.code.size)}
    report["candidates"] = []
    for cosine, a, b in pairs:
        a, b = alias[a], alias[b]
        if a == b:
            continue
        keep, drop = (a, b) if int(usage[a]) >= int(usage[b]) else (b, a)
        baseline = reproduced(model, memory, held, device)
        snapshot = copy.deepcopy(model.state_dict())
        codes_before = memory["codes"].clone()
        with torch.no_grad():
            model.code.embedding.weight[drop] = model.code.embedding.weight[keep]
        memory["codes"][memory["codes"] == drop] = keep
        merged = reproduced(model, memory, held, device)
        only_vector = args.repair == "embedding"
        for p in model.parameters():
            p.requires_grad_(False)
        parameters = [model.code.embedding.weight] if only_vector else [p for part in (model.executor, model.code, model.programmer) for p in part.parameters()]
        for p in parameters:
            p.requires_grad_(True)
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
        uses = (memory["codes"] == keep) & (torch.arange(memory["codes"].shape[1])[None] < memory["lengths"][:, None])
        pool = train[uses.any(1)[train]] if only_vector else train  # the stored pairs whose program names the merged code
        generator = torch.Generator().manual_seed(args.seed + 7 * len(report["candidates"]))
        model.train()
        trace = []
        for repair_step in range(args.repair_steps):
            if repair_step and repair_step % 100 == 0:
                model.eval()
                trace.append([repair_step, reproduced(model, memory, held, device)])
                model.train()
            index = pool[torch.randint(len(pool), (args.batch,), generator=generator)]
            x, y = memory["x"][index].to(device).long(), memory["y"][index].to(device).long()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, distance = replay_programs(model, x, y, memory["codes"][index].to(device), memory["lengths"][index].to(device), with_programmer=not only_vector)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1)) + (0.0 if only_vector else args.programmer_weight * distance.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if only_vector:
                model.code.embedding.weight.grad[torch.arange(model.code.size, device=device) != keep] = 0.0  # every other code stays as it is
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            with torch.no_grad():
                model.code.embedding.weight[drop] = model.code.embedding.weight[keep]  # the two indices stay one code
        model.eval()
        repaired = reproduced(model, memory, held, device)
        flops["repair"] += 3.0 * step_forward * args.repair_steps
        flops["checks"] += (3 + len(trace)) * step_forward / args.batch * len(held) / 2  # passes over the held-out part, executor only
        accepted = repaired >= baseline - args.margin
        entry = {"codes": [keep, drop], "cosine": cosine, "use": [int(usage[keep]), int(usage[drop])], "held_out_reproduced": {"before": baseline, "after_merge": merged, "after_repair": repaired, "during_repair": trace}, "accepted": accepted}
        report["candidates"].append(entry)
        print("candidate", json.dumps(entry), flush=True)
        if accepted:
            for c in alias:
                if alias[c] == drop:
                    alias[c] = keep
            usage[keep] += usage[drop]
            usage[drop] = 0
        else:
            model.load_state_dict(snapshot)
            memory["codes"] = codes_before
    report["codes_in_use_after"] = sorted({alias[c] for c in used})
    report["after"] = evaluate(model, data, args.split, infer, device, args.eval_samples)
    report["flops"] = {**flops, "total": sum(flops.values()), "one_training_step_of_the_run": None}
    flops_file = REPO / "runs" / "flops.json"
    if flops_file.exists() and args.run.name in json.loads(flops_file.read_text()):
        per_step = json.loads(flops_file.read_text())[args.run.name]["train_flops_per_step"]
        report["flops"]["one_training_step_of_the_run"] = per_step
        report["flops"]["total_in_training_steps"] = report["flops"]["total"] / per_step
    print("after", json.dumps(report["after"]), flush=True)
    print("flops", json.dumps(report["flops"]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": config, "step": step, "merge_sleep": report["candidates"]}, args.save)


if __name__ == "__main__":
    main()
