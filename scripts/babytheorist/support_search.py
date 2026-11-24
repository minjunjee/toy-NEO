#!/usr/bin/env python3
"""Search for a program with the trained parts, verified on the support pairs, then replay it on the query.

The model is used as it is; nothing is trained. A beam search runs on the first support pair: at every step each kept
grid is encoded, the programmer proposes an action toward the support output, and the `width` codes nearest to that
action are executed. Grids are decoded and re-encoded between steps (grid-only handover, as rollout "hard"). Candidates
are ranked by the programmer's preference summed over the program's steps: the log-softmax of the cosine between the
action and each code, at temperature --tau (with --rank ce, by the cell cross-entropy to the support output instead, which
drops paths whose intermediate grids are far from the output); identical grids are kept once. A candidate whose grid
equals the support output exactly is recorded as a program and leaves the beam, so the search keeps looking for others.

Every recorded program is then run on the other support pairs. The program used for the query is the first one (shortest,
then best ranked) that reproduces all support pairs; if none does, the first one that reproduces the first support pair;
if none does, the program of the grid closest to the first support output, by cell cross-entropy, over all steps. Width 1 is the greedy stepwise rollout that stops
at the first exact reproduction.

With --choose vote, every program that reproduces all support pairs is run on the query and the most common answer is
used (ties go to the earlier program); without it, the first such program is used.

Reported per depth of the hidden program (the hidden program is read only to sort the results):
    transfer             the query answer is reproduced by the program chosen as above
    transfer_unverified  the same without the check on the other supports (first program reproducing support 1)
    found_s1             some program reproduces the first support pair within max_steps
    found_all            some program reproduces every support pair
    s1_rejected          the first program found for support 1 fails another support pair (share of tasks with found_s1)
    rescued              of those, another program reproducing every support pair was found
    query_wrong_verified of the tasks with found_all, the chosen program misses the query answer

    python scripts/babytheorist/support_search.py --split lv5 --eval-file data/babytheorist/single_a/lv5_eval.h5 \
        --widths 1 4 8 16 --max-steps 10 --max-samples 2000 --out runs/queue/support_search_lv5.json runs/<run> [...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from training.evaluate import load_run  # noqa: E402

METRICS = ("transfer", "transfer_unverified", "found_s1", "found_all", "s1_rejected", "rescued", "query_wrong_verified")


def code_table(model) -> torch.Tensor:
    return F.normalize(model.code.embedding.weight.float(), dim=-1)


@torch.no_grad()
def run_programs(model, table: torch.Tensor, grids: torch.Tensor, programs: torch.Tensor) -> torch.Tensor:
    """Run programs (N, L) of code indices, -1 meaning stop, on grids (N, H, W); grid-only handover between steps."""

    grids = grids.clone()
    for t in range(programs.shape[1]):
        codes = programs[:, t]
        active = codes >= 0
        if not active.any():
            break
        rows = torch.nonzero(active).flatten()
        state = model.encoder(grids[rows])
        out = model.decoder(model.executor(state, table[codes[rows]][:, None])).argmax(dim=-1)
        grids[rows] = out.to(grids.dtype)
    return grids


@torch.no_grad()
def beam_search(model, table: torch.Tensor, x: torch.Tensor, y: torch.Tensor, width: int, max_steps: int, rank: str = "policy", tau: float = 0.1, keep: int = 32,
                critic=None, critic_weight: float = 0.0):
    """Programs reproducing y from x, in order of discovery, and the best beam program at the end (the fallback)."""

    n, h, w = x.shape
    device = x.device
    target = model.encoder(y)
    grids = x[:, None]  # (n, beam, h, w)
    programs = torch.empty(n, 1, 0, dtype=torch.long, device=device)
    valid = torch.ones(n, 1, dtype=torch.bool, device=device)
    found: list[list[list[int]]] = [[] for _ in range(n)]
    salt = torch.randint(1, 2**31 - 1, (h * w,), device=device, dtype=torch.long)
    cost = torch.zeros(n, 1, device=device)  # summed negative log preference of the kept programs
    k = min(width, table.shape[0])
    best_program = torch.full((n, max_steps), -1, dtype=torch.long, device=device)
    best_score = torch.full((n,), float("inf"), device=device)
    for t in range(max_steps):
        beam = grids.shape[1]
        flat = grids.reshape(n * beam, h, w)
        state = model.encoder(flat)
        action = model.programmer(state, target.repeat_interleave(beam, dim=0)).reshape(n * beam, -1).float()
        similarity = F.normalize(action, dim=-1) @ table.T
        nearest = similarity.topk(k, dim=1).indices  # (n*beam, k)
        preference = torch.log_softmax(similarity / tau, dim=1).gather(1, nearest).reshape(n, beam * k)
        logits = model.decoder(model.executor(state.repeat_interleave(k, dim=0), table[nearest.flatten()][:, None]))
        new = logits.argmax(dim=-1).reshape(n, beam * k, h, w)
        targets = y.repeat_interleave(beam * k, dim=0)
        score = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1).long(), reduction="none")
        score = score.reshape(n, beam * k, h * w).sum(dim=-1)
        candidate_programs = torch.cat([programs.repeat_interleave(k, dim=1), nearest.reshape(n, beam * k, 1)], dim=2)
        alive = valid.repeat_interleave(k, dim=1)
        step_cost = cost.repeat_interleave(k, dim=1) - preference
        heuristic = torch.zeros_like(step_cost)
        if critic is not None and critic_weight:  # expected grid-changing steps left, from the critic trained on dreams
            left = critic(model.encoder(new.reshape(n * beam * k, h, w)), target.repeat_interleave(beam * k, dim=0)).reshape(n * beam * k, -1).float()
            heuristic = (torch.softmax(left, dim=1) * torch.arange(left.shape[1], device=device)).sum(dim=1).reshape(n, beam * k) * critic_weight
        closeness = torch.where(alive, score, torch.full_like(score, float("inf")))
        score = torch.where(alive, step_cost + heuristic if rank == "policy" else score, torch.full_like(score, float("inf")))
        order = score.argsort(dim=1)
        score = score.gather(1, order)
        closeness = closeness.gather(1, order)
        step_cost = step_cost.gather(1, order)
        new = new.gather(1, order[:, :, None, None].expand_as(new))
        candidate_programs = candidate_programs.gather(1, order[:, :, None].expand_as(candidate_programs))
        alive = torch.isfinite(score)
        nearest_grid = closeness.argmin(dim=1)  # the grid closest to y over all steps is the fallback program
        closest = closeness.gather(1, nearest_grid[:, None])[:, 0]
        better = closest < best_score
        best_score = torch.where(better, closest, best_score)
        rows = torch.nonzero(better).flatten()
        best_program[rows] = -1
        best_program[rows, : t + 1] = candidate_programs[rows, nearest_grid[rows]]
        matched = new.eq(y[:, None]).flatten(2).all(dim=2) & alive
        for row, col in torch.nonzero(matched).tolist():
            if len(found[row]) < keep:
                found[row].append(candidate_programs[row, col].tolist())
        # identical grids are kept once (the best ranked); matched grids leave the beam
        hashes = (new.flatten(2).long() * salt).sum(dim=2)
        same = hashes[:, :, None].eq(hashes[:, None, :])
        earlier = torch.ones(beam * k, beam * k, dtype=torch.bool, device=device).tril(diagonal=-1)
        duplicate = (same & earlier & alive[:, None, :]).any(dim=2)
        score = torch.where(duplicate | matched, torch.full_like(score, float("inf")), score)
        order = score.argsort(dim=1)[:, :width]
        grids = new.gather(1, order[:, :, None, None].expand(n, order.shape[1], h, w))
        programs = candidate_programs.gather(1, order[:, :, None].expand(n, order.shape[1], candidate_programs.shape[2]))
        valid = torch.isfinite(score.gather(1, order))
        cost = step_cost.gather(1, order)
    return found, best_program


def pad(programs: list[list[int]], length: int) -> list[int]:
    return [list(p) + [-1] * (length - len(p)) for p in programs]


@torch.no_grad()
def evaluate(model, data, split: str, width: int, max_steps: int, max_samples: int | None, batch_tasks: int, device, deepest: int, rank: str, tau: float,
             critic=None, critic_weight: float = 0.0, choose: str = "first"):
    table = code_table(model)
    buckets = deepest
    sums = {m: torch.zeros(buckets) for m in METRICS}
    denominators = {m: torch.zeros(buckets) for m in METRICS}
    counts = torch.zeros(buckets)
    lengths_chosen = torch.zeros(buckets, max_steps + 1)
    for batch in data.eval_loader(split, batch_tasks, max_samples=max_samples):
        sx, sy = batch["supports_x"].to(device), batch["supports_y"].to(device)
        k_support, qx, qy = batch["n_support"].to(device), batch["query_x"].to(device), batch["query_y"].to(device)
        depth = batch["depth"].clamp(max=deepest) - 1
        n = sx.shape[0]
        found, fallback = beam_search(model, table, sx[:, 0], sy[:, 0], width, max_steps, rank, tau, critic=critic, critic_weight=critic_weight)
        # check every found program on the other support pairs
        rows, programs = [], []
        for i, progs in enumerate(found):
            for p in progs:
                rows.append(i)
                programs.append(p)
        passes = torch.zeros(len(rows), dtype=torch.bool, device=device)
        if rows:
            rows_t = torch.tensor(rows, device=device)
            prog_t = torch.tensor(pad(programs, max_steps), device=device)
            passes[:] = True
            for j in range(1, sx.shape[1]):
                use = j < k_support[rows_t]
                out = run_programs(model, table, sx[rows_t, j], prog_t)
                passes &= out.eq(sy[rows_t, j]).flatten(1).all(dim=1) | ~use
        chosen, unverified = [], []
        stats = {m: [] for m in METRICS}
        cursor = 0
        for i, progs in enumerate(found):
            mine = passes[cursor : cursor + len(progs)].tolist()
            cursor += len(progs)
            first_all = next((p for p, ok in zip(progs, mine) if ok), None)
            first_s1 = progs[0] if progs else None
            fallback_program = [c for c in fallback[i].tolist() if c >= 0]
            chosen.append(first_all or first_s1 or fallback_program)
            unverified.append(first_s1 or fallback_program)
            stats["found_s1"].append((bool(progs), True))
            stats["found_all"].append((first_all is not None, True))
            stats["s1_rejected"].append((bool(progs) and not mine[0], bool(progs)))
            stats["rescued"].append((bool(progs) and not mine[0] and first_all is not None, bool(progs) and not mine[0]))
        chosen_pred = run_programs(model, table, qx, torch.tensor(pad(chosen, max_steps), device=device))
        unverified_pred = run_programs(model, table, qx, torch.tensor(pad(unverified, max_steps), device=device))
        if choose == "vote" and rows:  # every program that reproduces all supports answers the query; the most common answer wins
            keep = torch.nonzero(passes).flatten()
            if len(keep):
                voters = rows_t[keep]
                answers = run_programs(model, table, qx[voters], prog_t[keep])
                salt = torch.arange(1, answers[0].numel() + 1, device=device, dtype=torch.long) * 7919
                keys = (answers.flatten(1).long() * salt).sum(dim=1).tolist()
                tally: dict[int, dict] = {}
                for v, key, answer_index in zip(voters.tolist(), keys, range(len(keys))):
                    entry = tally.setdefault(v, {})
                    count, first = entry.get(key, (0, answer_index))
                    entry[key] = (count + 1, first)
                for v, entry in tally.items():
                    best = max(entry.items(), key=lambda item: (item[1][0], -item[1][1]))
                    chosen_pred[v] = answers[best[1][1]]
        right = chosen_pred.eq(qy).flatten(1).all(dim=1).tolist()
        right_unverified = unverified_pred.eq(qy).flatten(1).all(dim=1).tolist()
        for i in range(n):
            stats["transfer"].append((right[i], True))
            stats["transfer_unverified"].append((right_unverified[i], True))
            stats["query_wrong_verified"].append((not right[i], stats["found_all"][i][0]))
        for i in range(n):
            b = int(depth[i])
            counts[b] += 1
            lengths_chosen[b, min(len(chosen[i]), max_steps)] += 1
            for m in METRICS:
                value, counted = stats[m][i]
                if counted:
                    sums[m][b] += float(value)
                    denominators[m][b] += 1
    out = {"tasks": int(counts.sum())}
    for m in METRICS:
        out[m] = float(sums[m].sum() / denominators[m].sum().clamp(min=1))
        out[f"{m}_by_depth"] = [float(s / d) if d > 0 else None for s, d in zip(sums[m], denominators[m])]
    out["tasks_by_depth"] = counts.int().tolist()
    out["chosen_length_by_depth"] = lengths_chosen.int().tolist()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--deepest", type=int, default=10, help="depths are reported one by one up to this, which also holds deeper")
    parser.add_argument("--rows", type=int, default=4096, help="executor rows per batch; tasks per batch = rows // width**2")
    parser.add_argument("--ckpt", type=Path, help="a checkpoint other than the run's last, e.g. one written by a sleep step")
    parser.add_argument("--name", help="the name recorded for the run (default: the run directory's)")
    parser.add_argument("--rank", choices=("policy", "ce"), default="policy")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--critic", type=Path, help="a remaining-steps critic written by train_critic.py, added to the ranking")
    parser.add_argument("--critic-weight", type=float, default=1.0)
    parser.add_argument("--choose", choices=("first", "vote"), default="first",
                        help="first: the first program that reproduces all supports; vote: the query answer most of those programs give")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    report = json.loads(args.out.read_text()) if args.out.exists() else []
    done = {(r["run"], r["split"], r["width"], r.get("rank", "ce")) for r in report}
    rank_name = args.rank + (f"+critic{args.critic_weight:g}" if args.critic else "") + ("+vote" if args.choose == "vote" else "")
    critic = None
    if args.critic:
        sys.path.insert(0, str(REPO / "src"))
        from core.registry import PROGRAMMERS, build, import_all
        import_all()
        saved = torch.load(args.critic, map_location="cpu", weights_only=False)
        critic = build(PROGRAMMERS, dict(saved["config"])).to(device).eval()
        critic.load_state_dict(saved["state"])
    for run in args.runs:
        model, data, config, step = load_run(run, args.ckpt, device)
        model.eval()
        label = args.name or run.name
        if args.eval_file:
            data.eval_paths[args.split] = str(args.eval_file)
        for width in args.widths:
            if (label, args.split, width, rank_name) in done:
                continue
            start = time.time()
            batch_tasks = max(4, args.rows // (width * width))
            result = evaluate(model, data, args.split, width, args.max_steps, args.max_samples, batch_tasks, device, args.deepest, args.rank, args.tau,
                              critic=critic, critic_weight=args.critic_weight, choose=args.choose)
            entry = {"run": label, "step": step, "split": args.split, "width": width, "rank": rank_name, "tau": args.tau, "max_steps": args.max_steps, "seconds": round(time.time() - start, 1), **result}
            report.append(entry)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=1) + "\n")
            by_depth = " ".join("-" if v is None else f"{v:.2f}" for v in result["transfer_by_depth"])
            print(f"{label} {args.split} {rank_name} width {width}: transfer {result['transfer']:.3f} unverified {result['transfer_unverified']:.3f} "
                  f"found_s1 {result['found_s1']:.3f} found_all {result['found_all']:.3f} s1_rejected {result['s1_rejected']:.3f} "
                  f"rescued {result['rescued']:.3f} query_wrong_verified {result['query_wrong_verified']:.3f} | by depth {by_depth} "
                  f"({entry['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
