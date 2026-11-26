#!/usr/bin/env python3
"""Does a code act as one function on states unlike those it was trained on? A ladder of states moving away from the
training distribution, and two ways to read each code on it.

States, all one object on the 15x15 grid, drawn with the generator of the run's primitive set:
    fresh     new objects from the training shape factory (pixel budget 4 to 15, box span up to 5)
    after_k   a fresh object after k random primitives of the set (each step must change the grid and keep the object
              on it); these are the intermediate states of programs of length k + 1
    large     new objects from a larger shape factory (pixel budget 16 to 29, box span up to 7), never seen in training

1. With the labels (a diagnostic): each primitive is mapped to the code that most often reproduced it exactly on the
   run's single-primitive eval tasks (as in oracle_program.py). On every state where the primitive applies and changes
   the grid, that code is run by the executor and its output is compared with the primitive's true output.
2. Without the labels: relations between codes that the model itself shows on fresh states. For every pair of codes the
   programmer uses (at least 0.5% of its choices in the run's last full evaluation), a pair is an inverse pair if running one and then the other returns the start grid, and a commuting
   pair if both orders give the same grid, each on at least 90% of fresh states where both codes change the grid. Each
   relation found on fresh states is then checked on every rung of the ladder. A code that is one function keeps its
   relations; a code that learned a mapping of training-like inputs loses them. For reference, the same relations are
   checked with the true primitives of the mapped codes on the same states (the label-free relations of the world).

    python scripts/babytheorist/code_function_ladder.py --primitives 24 --states 400 --out runs/queue/ladder_m24.json \
        runs/onestep_s_m24_hardem_pairs2_s42 [...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "babytheorist"))

import generate_many_primitives as gen  # noqa: E402
from oracle_program import primitive_codes  # noqa: E402
from support_search import code_table, run_programs  # noqa: E402
from training.evaluate import load_run  # noqa: E402

THRESHOLD = 0.9


def ladder(names: tuple[str, ...], states: int, depth: int, seed: int) -> dict[str, list]:
    g = gen.Generator(names, [seed, 0])
    large = gen.ShapeFactory(generators=(gen.solid, gen.bordered, gen.patterned), size_range=(16, 30), compactness=8.0, bordered_hollow_min_size=9,
                             bordered_hollow_random_inner_size=True, max_bbox_span=7)
    rungs: dict[str, list] = {"fresh": [], **{f"after_{k}": [] for k in range(1, depth + 1)}, "large": []}
    while len(rungs["fresh"]) < states:
        rungs["fresh"].append(g.scene())
    for k in range(1, depth + 1):
        while len(rungs[f"after_{k}"]) < states:
            shape, world = g.scene()
            seen = {world.tobytes()}
            for _ in range(k):
                for _ in range(50):
                    out = gen.step(shape, g.functions[int(g.rng.integers(len(g.functions)))])
                    if out is not None and out[1].tobytes() not in seen:
                        break
                else:
                    out = None
                if out is None:
                    break
                shape, world = out
                seen.add(world.tobytes())
            else:
                rungs[f"after_{k}"].append((shape, world))
    while len(rungs["large"]) < states:
        placed = gen.place_shapes([large.sample(g.rng, tuple(range(1, 10)))], world_size=gen.WORLD, gap=2, rng=g.rng)
        if placed is not None and gen.render(placed[1][0]) is not None:
            rungs["large"].append((placed[1][0], gen.render(placed[1][0])))
    return rungs


def used_codes(run: Path) -> list[int]:
    import glob
    for path in sorted(glob.glob(str(run / "eval" / "*full*_step*.json")), reverse=True):
        usage = json.loads(Path(path).read_text()).get("code_usage")
        if usage:
            total = sum(usage)
            return [c for c, u in enumerate(usage) if u >= 0.005 * total]
    return []


def truth(states: list, fn) -> tuple[list[int], np.ndarray]:
    """Indices of the states where the primitive applies and changes the grid, and its outputs there."""

    rows, outs = [], []
    for i, (shape, world) in enumerate(states):
        out = gen.step(shape, fn)
        if out is not None and not np.array_equal(out[1], world):
            rows.append(i)
            outs.append(out[1])
    return rows, (np.stack(outs) if outs else np.zeros((0, gen.WORLD, gen.WORLD), dtype=np.int8))


@torch.no_grad()
def relation_rates(model, table, grids: torch.Tensor, pairs: list[tuple[int, int]], codes: list[int]) -> dict:
    """For each code, its output on every grid; then for each pair the inverse and commute rates over grids both change."""

    n = grids.shape[0]
    once = {c: run_programs(model, table, grids, torch.full((n, 1), c, device=grids.device)) for c in codes}
    changed = {c: ~once[c].eq(grids).flatten(1).all(dim=1) for c in codes}
    out = {}
    for a, b in pairs:
        both = changed[a] & changed[b]
        if int(both.sum()) < 10:
            out[(a, b)] = (None, None, int(both.sum()))
            continue
        ab = run_programs(model, table, once[a][both], torch.full((int(both.sum()), 1), b, device=grids.device))
        back = ab.eq(grids[both]).flatten(1).all(dim=1)
        ba = run_programs(model, table, once[b][both], torch.full((int(both.sum()), 1), a, device=grids.device))
        out[(a, b)] = (float(back.float().mean()), float(ab.eq(ba).flatten(1).all(dim=1).float().mean()), int(both.sum()))
    return out


def world_rates(states: list, fa, fb) -> tuple[float | None, float | None]:
    """The same two relations with the true primitives, over states where both change the grid."""

    back = commute = total = 0
    for shape, world in states:
        a, b = gen.step(shape, fa), gen.step(shape, fb)
        if a is None or b is None or np.array_equal(a[1], world) or np.array_equal(b[1], world):
            continue
        total += 1
        ab, ba = gen.step(a[0], fb), gen.step(b[0], fa)
        back += int(ab is not None and np.array_equal(ab[1], world))
        commute += int(ab is not None and ba is not None and np.array_equal(ab[1], ba[1]))
    return (back / total, commute / total) if total >= 10 else (None, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--primitives", choices=sorted(gen.SETS), required=True)
    parser.add_argument("--states", type=int, default=400)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = gen.SETS[args.primitives]
    rungs = ladder(names, args.states, args.depth, args.seed)
    targets = {name: {rung: truth(states, gen.function_of(name)) for rung, states in rungs.items()} for name in names}
    report = json.loads(args.out.read_text()) if args.out.exists() else []
    for run in args.runs:
        mapping = primitive_codes(run)
        model, _, _, step = load_run(run, None, device)
        model.eval()
        table = code_table(model)
        accuracy = {}
        for name in names:
            if name not in mapping:
                continue
            accuracy[name] = {}
            for rung, states in rungs.items():
                rows, outs = targets[name][rung]
                if not rows:
                    accuracy[name][rung] = None
                    continue
                grids = torch.tensor(np.stack([states[i][1] for i in rows]), device=device, dtype=torch.long)
                pred = run_programs(model, table, grids, torch.full((len(rows), 1), mapping[name], device=device))
                accuracy[name][rung] = float(pred.eq(torch.tensor(outs, device=device, dtype=torch.long)).flatten(1).all(dim=1).float().mean())
        # relations the model shows on fresh states, among the codes the programmer uses (at least 0.5% of its choices
        # in the run's last full evaluation; no labels)
        codes = used_codes(run) or sorted(set(mapping.values()))
        pairs = [(a, b) for a in codes for b in codes if a < b]
        fresh = torch.tensor(np.stack([w for _, w in rungs["fresh"]]), device=device, dtype=torch.long)
        found = relation_rates(model, table, fresh, pairs, codes)
        inverse = [p for p, (back, _, n) in found.items() if back is not None and back >= THRESHOLD]
        commute = [p for p, (_, com, n) in found.items() if com is not None and com >= THRESHOLD]
        by_code = {c: [n for n, cc in mapping.items() if cc == c] for c in codes}  # for the world's reference only
        relations = {}
        for rung, states in rungs.items():
            grids = torch.tensor(np.stack([w for _, w in states]), device=device, dtype=torch.long)
            rates = relation_rates(model, table, grids, inverse + commute, codes) if rung != "fresh" else found
            kept_inverse = [rates[p][0] for p in inverse if rates[p][0] is not None]
            kept_commute = [rates[p][1] for p in commute if rates[p][1] is not None]
            world_inverse, world_commute = [], []
            for kind, plist, store in (("inverse", inverse, world_inverse), ("commute", commute, world_commute)):
                for a, b in plist:
                    if not by_code[a] or not by_code[b]:
                        continue
                    fa, fb = gen.function_of(by_code[a][0]), gen.function_of(by_code[b][0])
                    rate = world_rates(states, fa, fb)[0 if kind == "inverse" else 1]
                    if rate is not None:
                        store.append(rate)
            relations[rung] = {"inverse_model": float(np.mean(kept_inverse)) if kept_inverse else None, "inverse_world": float(np.mean(world_inverse)) if world_inverse else None,
                               "commute_model": float(np.mean(kept_commute)) if kept_commute else None, "commute_world": float(np.mean(world_commute)) if world_commute else None}
        mean = {rung: float(np.mean([accuracy[n][rung] for n in accuracy if accuracy[n][rung] is not None])) for rung in rungs}
        entry = {"run": run.name, "step": step, "primitives": args.primitives, "states": args.states, "mapping": mapping, "accuracy": accuracy, "mean_accuracy": mean,
                 "inverse_pairs": len(inverse), "commute_pairs": len(commute), "relations": relations}
        report = [r for r in report if r["run"] != run.name] + [entry]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(run.name, "accuracy", " ".join(f"{k} {v:.2f}" for k, v in mean.items()), flush=True)
        print(run.name, f"inverse pairs {len(inverse)} commute pairs {len(commute)}", " | ".join(
            f"{k} inv {v['inverse_model'] if v['inverse_model'] is None else round(v['inverse_model'], 2)}/{v['inverse_world'] if v['inverse_world'] is None else round(v['inverse_world'], 2)}"
            f" com {v['commute_model'] if v['commute_model'] is None else round(v['commute_model'], 2)}/{v['commute_world'] if v['commute_world'] is None else round(v['commute_world'], 2)}"
            for k, v in relations.items()), flush=True)


if __name__ == "__main__":
    main()
