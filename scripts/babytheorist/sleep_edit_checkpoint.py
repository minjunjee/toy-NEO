#!/usr/bin/env python3
"""Apply one offline phase to a training checkpoint, so that training can be resumed from the edited checkpoint.

The phase does two things, reading nothing from the hidden programs:
1. Merge duplicates. A replay memory is built as in merge_sleep.py. For pairs of codes, most similar vectors first, code b
   is given a's vector; if the stored programs of the held-out part of the memory are still reproduced as often as before
   (within a margin) with no repair at all, b was a duplicate: the merge is kept and index b becomes free.
2. Hand the freed index to what is not explained. Each freed index gets the programmer's first action on a training
   pair the model cannot reproduce (drawn at random among them), so the code starts where an unexplained pair asks for it.

The same edit is written into the averaged weights and the raw weights of the checkpoint; optimizer state is left as it is.
The edited checkpoint is saved under runs/<new run>/ckpt/ with the step unchanged, and the compute spent is written to
runs/<new run>/extra_flops.json for the learning curves.

    python scripts/babytheorist/sleep_edit_checkpoint.py --ckpt runs/<run>/ckpt/step-NNNNNN.pt --new-run <name> --split lv1
    python scripts/train.py --config configs/babytheorist/<name>.yaml --resume runs/<name>/ckpt/step-NNNNNN.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.registry import DATASETS, METHODS, build, import_all  # noqa: E402
from data.babytheorist import TrainPairs  # noqa: E402
from merge_sleep import build_memory, replay_programs, reproduced  # noqa: E402

EMBEDDING = "code.embedding.weight"


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--new-run", help="write the edited checkpoint under runs/<new run>/ckpt/ (a run that branches off)")
    parser.add_argument("--in-place", action="store_true", help="edit the checkpoint where it is (a run that sleeps as it trains); the checkpoint before the phase is kept beside it")
    parser.add_argument("--split", required=True)
    parser.add_argument("--memory", type=int, default=20000)
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--min-use", type=int, default=50)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freed", choices=("unexplained", "alias", "random"), default="unexplained",
                        help="what a freed index becomes: the programmer's action on an unexplained pair (the phase as described), a copy of the code it was merged into "
                             "(merge only), or a random direction (no pointing at what is unexplained). The last two are controls.")
    args = parser.parse_args()
    import_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build(METHODS, config["method"]).build_model(config["model"], init_from=None)
    model.load_state_dict(payload["model"])  # the averaged weights: what evaluation uses
    model.to(device).eval()
    data = build(DATASETS, config["data"])
    infer = next((dict(b.get("infer", {})) for b in config["evals"] if b["type"] == "babytheorist_observe"), {})
    pairs = TrainPairs(data.train_paths, data.train_max_depth)
    memory = build_memory(model, pairs, args.memory, infer, device, args.seed)
    stored = len(memory["x"])
    order = torch.randperm(stored, generator=torch.Generator().manual_seed(args.seed + 1))
    held = order[: int(stored * args.holdout)]
    used_positions = torch.arange(memory["codes"].shape[1])[None] < memory["lengths"][:, None]
    usage = torch.bincount(memory["codes"][used_positions], minlength=model.code.size)
    with FlopCounterMode(display=False) as counter:
        index = held[:16]
        replay_programs(model, memory["x"][index].to(device).long(), memory["y"][index].to(device).long(), memory["codes"][index].to(device), memory["lengths"][index].to(device), with_programmer=True)
    per_pair = float(counter.get_total_flops()) / 16
    flops = per_pair * args.memory  # explaining the sampled pairs
    report: dict = {"source": str(args.ckpt), "step": int(payload["step"]), "memory": {"sampled": args.memory, "stored": stored, "held_out": len(held), "code_usage": usage.tolist()}, "merges": [], "freed": []}

    vectors = F.normalize(model.code.embedding.weight.float(), dim=-1)
    similarity = vectors @ vectors.T
    used = [c for c in range(model.code.size) if int(usage[c]) >= args.min_use]
    candidates = sorted(((float(similarity[a, b]), a, b) for i, a in enumerate(used) for b in used[i + 1:]), reverse=True)[: args.candidates]
    gone: set[int] = set()
    for cosine, a, b in candidates:
        if a in gone or b in gone:
            continue
        keep, drop = (a, b) if int(usage[a]) >= int(usage[b]) else (b, a)
        baseline = reproduced(model, memory, held, device)
        old_vector, old_codes = model.code.embedding.weight[drop].clone(), memory["codes"].clone()
        model.code.embedding.weight[drop] = model.code.embedding.weight[keep]
        memory["codes"][memory["codes"] == drop] = keep
        merged = reproduced(model, memory, held, device)
        flops += 2 * per_pair / 2 * len(held)  # two passes over the held-out part, executor only
        accepted = merged >= baseline - args.margin
        report["merges"].append({"codes": [keep, drop], "cosine": cosine, "use": [int(usage[keep]), int(usage[drop])], "held_out_reproduced": {"before": baseline, "after_merge": merged}, "accepted": accepted})
        print("merge", json.dumps(report["merges"][-1]), flush=True)
        if accepted:
            gone.add(drop)
            usage[keep] += usage[drop]
            usage[drop] = 0
        else:
            model.code.embedding.weight[drop] = old_vector
            memory["codes"] = old_codes

    # the freed indices go to pairs the model cannot reproduce
    new_vectors: dict[int, torch.Tensor] = {}
    if gone:
        generator = torch.Generator().manual_seed(args.seed + 2)
        chosen = torch.randperm(len(pairs), generator=generator)[:4000]
        x, y = pairs.x[chosen].long(), pairs.y[chosen].long()
        missed = []
        for start in range(0, len(chosen), 500):
            bx, by = x[start:start + 500].to(device), y[start:start + 500].to(device)
            out = model.self_explain(bx, by, **infer)
            missed.append(~out["pred"].eq(by).flatten(1).all(1).cpu())
        missed = torch.cat(missed)
        flops += per_pair * len(chosen)
        unexplained = torch.nonzero(missed).flatten()
        report["unexplained_share_of_training_pairs"] = float(missed.float().mean())
        for slot, drop in enumerate(sorted(gone)):
            pick = int(unexplained[torch.randint(len(unexplained), (1,), generator=generator)])
            action = model.programmer(model.encoder(x[pick:pick + 1].to(device)), model.encoder(y[pick:pick + 1].to(device)))
            scale = model.code.embedding.weight.norm(dim=-1).mean()
            if args.freed == "unexplained":
                new_vectors[drop] = (F.normalize(action.reshape(-1).float(), dim=-1) * scale).cpu()
            elif args.freed == "random":
                new_vectors[drop] = (F.normalize(torch.randn(action.numel(), generator=generator), dim=-1) * scale.cpu())
            report["freed"].append({"code": drop, "becomes": args.freed, "from_training_pair": int(chosen[pick]) if args.freed == "unexplained" else None})

    merged_into = {m["codes"][1]: m["codes"][0] for m in report["merges"] if m["accepted"]}
    for key in ("model", "raw_model"):
        if key in payload and gone:
            weights = payload[key][EMBEDDING]
            for drop in gone:
                weights[drop] = weights[merged_into[drop]].clone() if args.freed == "alias" else new_vectors[drop].to(weights.dtype)
    report["flops"] = flops
    if args.in_place:
        output = args.ckpt.parents[1]
        if gone:
            args.ckpt.rename(args.ckpt.with_name(args.ckpt.stem + ".before_sleep.pt"))
            torch.save(payload, args.ckpt)
        phases = json.loads((output / "extra_flops.json").read_text()) if (output / "extra_flops.json").exists() else []
        (output / "extra_flops.json").write_text(json.dumps([*phases, {"at_step": int(payload["step"]), "flops": flops}]) + "\n")
        with open(output / "sleep_log.jsonl", "a") as log:
            log.write(json.dumps(report) + "\n")
    else:
        if not args.new_run:
            raise SystemExit("give --new-run or --in-place")
        output = REPO / "runs" / args.new_run
        (output / "ckpt").mkdir(parents=True, exist_ok=True)
        torch.save(payload, output / "ckpt" / args.ckpt.name)
        (output / "extra_flops.json").write_text(json.dumps({"at_step": int(payload["step"]), "flops": flops}) + "\n")
        (output / "sleep_edit.json").write_text(json.dumps(report, indent=1) + "\n")
    print("freed", report["freed"], "flops", f"{flops:.3e}", "->", output)


if __name__ == "__main__":
    main()
