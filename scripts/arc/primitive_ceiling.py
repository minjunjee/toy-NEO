#!/usr/bin/env python
"""Ceiling of ARC-AGI under the eight BabyTheorist base primitives.

No learning: brute-force search with the true primitive functions. For every
ARC task we look for one short program that maps every train input to its
train output, then check it on the test pairs.

Primitives (semantics mirrored from babytheorist/primitives.py and engine.py):
    rot90            np.rot90(grid, 1): 90 degrees counter-clockwise on the
                     object's bounding box, top-left corner kept in place.
    flip_h           np.flipud: upside-down inside the bounding box.
    flip_v           np.fliplr: left-right inside the bounding box.
    next_color       non-zero colours cycle c -> (c % 9) + 1, 0 stays 0.
    translate_*      the object moves by its OWN bounding-box height (up/down)
                     or width (left/right). Not by one cell.
Engine rule: a step whose result leaves the grid or overlaps a non-background
cell of another object is invalid and so is the whole program. The engine also
rejects a transform that leaves the object unchanged; that is a constraint of
the task generator, so the search here allows such steps by default and only
flags programs that contain one (`engine_valid`).

Vocabularies:
    A  whole grid as one object. rot90 and the flips act on the full grid
       array (the shape changes for non-square grids), next_color recolours
       every non-background cell, translate shifts all non-background content
       and is invalid if any cell would leave the grid. Program length 1..4.
    B  one object selected by index. Objects are connected components of the
       input (two segmentations), indexed in reading order of the top-left
       corner of their bounding box, and kept as objects for the whole
       program (no re-segmentation). A step is (object index, primitive).
       Program length 1..3. Attempted only when every train input has 1..4
       objects and every train output has the shape of its input.

Translate step size is run in two modes: `own_size` (the benchmark's rule;
for A the "own size" is the bounding box of all non-background content) and
`one_cell` (not the benchmark's rule, reported for comparison).

Selection rule: programs are enumerated by length, then lexicographically in
the fixed step order below. A task is SOLVED (strict) if the first program
that fits all train pairs also fits all test pairs, and solved (loose) if any
train-fitting program within the length limit fits all test pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

PRIMITIVES = (
    "rot90", "flip_h", "flip_v", "next_color",
    "translate_up", "translate_down", "translate_left", "translate_right",
)
DIRECTIONS = {
    "translate_up": (-1, 0),
    "translate_down": (1, 0),
    "translate_left": (0, -1),
    "translate_right": (0, 1),
}
TRANSLATE_MODES = ("own_size", "one_cell")
SEGMENTATIONS = ("same_colour_4conn", "any_colour_8conn")
MAX_LEN_WHOLE_GRID = 4
MAX_LEN_OBJECT = 3
MAX_OBJECTS = 4

ARC_SETS = {
    "arc1_training": "ARC-AGI/data/training",
    "arc1_evaluation": "ARC-AGI/data/evaluation",
    "arc2_training": "ARC-AGI-2/data/training",
    "arc2_evaluation": "ARC-AGI-2/data/evaluation",
}


# === Primitive functions ===================================================


def transform(grid: np.ndarray, prim: str) -> np.ndarray:
    """rot90 / flip_h / flip_v / next_color on a 2D array (bounding box or grid)."""
    if prim == "rot90":
        return np.rot90(grid, 1)
    if prim == "flip_h":
        return np.flipud(grid)
    if prim == "flip_v":
        return np.fliplr(grid)
    if prim == "next_color":
        out = grid.copy()
        mask = out != 0
        out[mask] = (out[mask] % 9) + 1
        return out
    raise ValueError(f"not a transform: {prim!r}")


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


# --- Vocabulary A: whole grid as one object --------------------------------


def whole_grid_step(grid, prim, translate_mode, noop_invalid=False):
    """Apply one primitive to the whole grid. Returns the new grid or None."""
    if prim not in DIRECTIONS:
        out = transform(grid, prim)
        if noop_invalid and _same(out, grid):
            return None
        return out
    rows = np.flatnonzero(grid.any(axis=1))
    cols = np.flatnonzero(grid.any(axis=0))
    if rows.size == 0:  # nothing to move
        return None if noop_invalid else grid
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
    dr, dc = DIRECTIONS[prim]
    if translate_mode == "own_size":
        dr *= r1 - r0 + 1
        dc *= c1 - c0 + 1
    n_rows, n_cols = grid.shape
    if r0 + dr < 0 or r1 + dr >= n_rows or c0 + dc < 0 or c1 + dc >= n_cols:
        return None  # border rule of the engine: out of bounds is invalid
    out = np.zeros_like(grid)
    out[r0 + dr:r1 + dr + 1, c0 + dc:c1 + dc + 1] = grid[r0:r1 + 1, c0:c1 + 1]
    return out


# --- Vocabulary B: one object selected by index ----------------------------

# A scene is (world_shape, objects); an object is (bbox_grid, row, col) with
# 0 = empty inside the bounding box, mirroring babytheorist.core.Shape.


def segment(grid: np.ndarray, segmentation: str):
    """Split a grid into objects, sorted in reading order of the bbox corner."""
    found = []
    if segmentation == "same_colour_4conn":
        for colour in np.unique(grid):
            if colour == 0:
                continue
            labels, _ = ndi.label(grid == colour)  # default structure: 4-conn
            found.append(labels)
    elif segmentation == "any_colour_8conn":
        labels, _ = ndi.label(grid != 0, structure=np.ones((3, 3), dtype=int))
        found.append(labels)
    else:
        raise ValueError(segmentation)
    objects = []
    for labels in found:
        for idx, sl in enumerate(ndi.find_objects(labels), start=1):
            if sl is None:
                continue
            box = np.where(labels[sl] == idx, grid[sl], 0).astype(np.int8)
            r, c = sl[0].start, sl[1].start
            first_col = int(np.flatnonzero(box[0])[0])  # tie-break, unique
            objects.append(((r, c, first_col), box, r, c))
    objects.sort(key=lambda o: o[0])
    return [(box, r, c) for _, box, r, c in objects]


def render(scene) -> np.ndarray:
    world_shape, objects = scene
    canvas = np.zeros(world_shape, dtype=np.int8)
    for box, r, c in objects:
        view = canvas[r:r + box.shape[0], c:c + box.shape[1]]
        mask = box != 0
        view[mask] = box[mask]
    return canvas


def object_step(scene, step, translate_mode, noop_invalid=False):
    """Apply (object index, primitive) to a scene. Returns a scene or None."""
    world_shape, objects = scene
    k, prim = step
    if k >= len(objects):
        return None  # the engine's "selector matches no shape"
    box, r, c = objects[k]
    if prim in DIRECTIONS:
        dr, dc = DIRECTIONS[prim]
        if translate_mode == "own_size":
            dr *= box.shape[0]
            dc *= box.shape[1]
        new_box, nr, nc = box, r + dr, c + dc
    else:
        new_box, nr, nc = transform(box, prim), r, c
        if noop_invalid and _same(new_box, box):
            return None
    rows, cols = np.nonzero(new_box)
    rows = rows + nr
    cols = cols + nc
    if (rows.min() < 0 or cols.min() < 0
            or rows.max() >= world_shape[0] or cols.max() >= world_shape[1]):
        return None  # out of bounds
    occupied = np.zeros(world_shape, dtype=bool)
    for i, (other, orow, ocol) in enumerate(objects):
        if i != k:
            occupied[orow:orow + other.shape[0], ocol:ocol + other.shape[1]] |= other != 0
    if occupied[rows, cols].any():
        return None  # collision with another object
    new_objects = list(objects)
    new_objects[k] = (new_box, nr, nc)
    return (world_shape, new_objects)


# === Search ================================================================


def search(train_states, train_targets, test_states, test_targets,
           steps, step_fn, key_fn, render_fn, max_len):
    """Breadth-first enumeration with de-duplication by joint effect.

    The key covers train AND test states, so two programs are merged only if
    they agree everywhere; this keeps both selection rules exact. Programs
    that are invalid on a train input are dropped.
    """
    n_train = len(train_states)
    targets = list(train_targets) + list(test_targets)
    start = list(train_states) + list(test_states)
    seen = {tuple(key_fn(s) for s in start)}
    frontier = [((), start)]
    first_fit = None        # (program, passes_tests)
    first_pass = None       # first train-fitting program that passes the tests
    n_fit = 0               # distinct-effect train-fitting programs
    for _ in range(max_len):
        next_frontier = []
        for program, states in frontier:
            for step in steps:
                new_states = []
                valid = True
                for i, s in enumerate(states):
                    ns = None if s is None else step_fn(s, step)
                    if ns is None and i < n_train:
                        valid = False
                        break
                    new_states.append(ns)
                if not valid:
                    continue
                key = tuple(key_fn(s) for s in new_states)
                if key in seen:
                    continue
                seen.add(key)
                new_program = program + (step,)
                next_frontier.append((new_program, new_states))
                if all(_same(render_fn(new_states[i]), targets[i]) for i in range(n_train)):
                    passes = all(
                        new_states[i] is not None and _same(render_fn(new_states[i]), targets[i])
                        for i in range(n_train, len(targets))
                    )
                    n_fit += 1
                    if first_fit is None:
                        first_fit = (new_program, passes)
                    if passes and first_pass is None:
                        first_pass = new_program
            if first_fit is not None and first_pass is not None:
                break
        if first_fit is not None and first_pass is not None:
            break
        frontier = next_frontier
    return first_fit, first_pass, n_fit


def _grid_key(grid):
    return None if grid is None else (grid.shape, grid.tobytes())


def _scene_key(scene):
    if scene is None:
        return None
    return tuple((r, c, box.shape, box.tobytes()) for box, r, c in scene[1])


def _run(state, program, step_fn):
    for step in program:
        if state is None:
            return None
        state = step_fn(state, step)
    return state


def _summarise(first_fit, first_pass, n_fit, pairs_states, strict_step_fn, fmt):
    """Turn a search result into a JSON-friendly record."""
    out = {"attempted": True, "train_fit": first_fit is not None,
           "solved_strict": False, "solved_loose": first_pass is not None,
           "n_fit_effects": n_fit}
    if first_fit is not None:
        program, passes = first_fit
        out["solved_strict"] = bool(passes)
        out["program"] = fmt(program)
        # Would the benchmark engine accept it (no step that changes nothing)?
        out["engine_valid"] = all(
            _run(s, program, strict_step_fn) is not None for s in pairs_states
        )
    if first_pass is not None and not out["solved_strict"]:
        out["loose_program"] = fmt(first_pass)
    return out


def solve_whole_grid(train, test, translate_mode):
    def step_fn(g, p):
        return whole_grid_step(g, p, translate_mode)

    def strict_fn(g, p):
        return whole_grid_step(g, p, translate_mode, noop_invalid=True)

    train_in = [x for x, _ in train]
    test_in = [x for x, _ in test]
    fit, passing, n_fit = search(
        train_in, [y for _, y in train], test_in, [y for _, y in test],
        PRIMITIVES, step_fn, _grid_key, lambda g: g, MAX_LEN_WHOLE_GRID,
    )
    return _summarise(fit, passing, n_fit, train_in + test_in, strict_fn, list)


def solve_objects(train, test, segmentation, translate_mode):
    if any(x.shape != y.shape for x, y in train):
        return {"attempted": False, "reason": "output_shape_differs"}
    train_scenes = [(x.shape, segment(x, segmentation)) for x, _ in train]
    counts = [len(s[1]) for s in train_scenes]
    if min(counts) < 1 or max(counts) > MAX_OBJECTS:
        return {"attempted": False, "reason": "object_count_outside_1_to_4"}
    test_scenes = [(x.shape, segment(x, segmentation)) for x, _ in test]
    steps = [(k, p) for k in range(max(counts)) for p in PRIMITIVES]

    def step_fn(s, st):
        return object_step(s, st, translate_mode)

    def strict_fn(s, st):
        return object_step(s, st, translate_mode, noop_invalid=True)

    fit, passing, n_fit = search(
        train_scenes, [y for _, y in train], test_scenes, [y for _, y in test],
        steps, step_fn, _scene_key, render, MAX_LEN_OBJECT,
    )
    # Object indices are reported 1-based, like the benchmark's by_index.
    return _summarise(fit, passing, n_fit, train_scenes + test_scenes, strict_fn,
                      lambda prog: [[k + 1, p] for k, p in prog])


def load_task(path):
    with open(path) as f:
        task = json.load(f)

    def pairs(key):
        return [(np.array(p["input"], dtype=np.int8), np.array(p["output"], dtype=np.int8))
                for p in task[key]]

    return pairs("train"), pairs("test")


def solve_task(job):
    set_name, path = job
    train, test = load_task(path)
    record = {"identity": all(_same(x, y) for x, y in train)}
    for mode in TRANSLATE_MODES:
        record[f"A_whole_grid/{mode}"] = solve_whole_grid(train, test, mode)
        for seg in SEGMENTATIONS:
            record[f"B_{seg}/{mode}"] = solve_objects(train, test, seg, mode)
    return set_name, Path(path).stem, record


# === Aggregation and reporting =============================================


def vocabulary_keys():
    keys = []
    for mode in TRANSLATE_MODES:
        keys.append(f"A_whole_grid/{mode}")
        keys.extend(f"B_{seg}/{mode}" for seg in SEGMENTATIONS)
    return keys


def aggregate(per_task):
    """per_task: {task_id: record} for one ARC set."""
    summary = {"n_tasks": len(per_task),
               "identity_tasks": sorted(t for t, r in per_task.items() if r["identity"])}
    for key in vocabulary_keys():
        rows = {t: r[key] for t, r in sorted(per_task.items())}
        tried = {t: r for t, r in rows.items() if r["attempted"]}
        fit = {t: r for t, r in tried.items() if r["train_fit"]}
        strict = {t: r for t, r in fit.items() if r["solved_strict"]}
        loose = {t: r for t, r in fit.items() if r["solved_loose"]}
        reasons = {}
        for r in rows.values():
            if not r["attempted"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        summary[key] = {
            "n_tasks": len(rows),
            "attempted": len(tried),
            "not_attempted_reasons": reasons,
            "train_fit": len(fit),
            "solved_strict": len(strict),
            "solved_loose": len(loose),
            "fits_train_only": len(fit) - len(strict),
            "fits_train_only_loose_rule": len(fit) - len(loose),
            "solved_strict_engine_valid": sum(r["engine_valid"] for r in strict.values()),
            "solved": [
                {"task": t, "program": r["program"], "length": len(r["program"]),
                 "engine_valid": r["engine_valid"]}
                for t, r in strict.items()
            ],
            "solved_loose_only": [
                {"task": t, "first_fit_program": r["program"], "passing_program": r["loose_program"]}
                for t, r in loose.items() if not r["solved_strict"]
            ],
            "fits_train_only_tasks": [
                {"task": t, "program": r["program"]}
                for t, r in fit.items() if not r["solved_strict"]
            ],
        }
    return summary


def print_table(results):
    header = (f"{'set':<16} {'vocabulary / translate':<36} {'tasks':>5} {'tried':>5} "
              f"{'fit':>4} {'strict':>6} {'loose':>5} {'train-only':>10} {'eng-valid':>9}")
    print(header)
    print("-" * len(header))
    for set_name, summary in results.items():
        for key in vocabulary_keys():
            s = summary[key]
            print(f"{set_name:<16} {key:<36} {s['n_tasks']:>5} {s['attempted']:>5} "
                  f"{s['train_fit']:>4} {s['solved_strict']:>6} {s['solved_loose']:>5} "
                  f"{s['fits_train_only']:>10} {s['solved_strict_engine_valid']:>9}")
        print()


def show_task(path, vocabulary, program):
    """Print every pair of a task next to the program's output (eye check)."""
    train, test = load_task(path)
    name, mode = vocabulary.split("/")
    for label, pairs in (("train", train), ("test", test)):
        for i, (x, y) in enumerate(pairs):
            if name == "A_whole_grid":
                got = _run(x, program, lambda g, p: whole_grid_step(g, p, mode))
            else:
                seg = name[len("B_"):]
                scene = (x.shape, segment(x, seg))
                steps = [(k - 1, p) for k, p in program]
                final = _run(scene, steps, lambda s, st: object_step(s, st, mode))
                got = None if final is None else render(final)
            ok = got is not None and _same(got, y)
            print(f"--- {label} pair {i}: program output {'==' if ok else '!='} target")
            for title, g in (("input", x), ("target", y), ("program output", got)):
                print(title)
                if g is None:
                    print("  (invalid)")
                    continue
                for row in g:
                    print("  " + "".join("." if v == 0 else str(int(v)) for v in row))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", default="data/external",
                        help="directory holding the ARC-AGI and ARC-AGI-2 clones")
    parser.add_argument("--out", default="docs/reports/arc-ceiling/ceiling.json")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="first N tasks per set (debug)")
    parser.add_argument("--show", nargs=3, metavar=("TASK_JSON", "VOCABULARY", "PROGRAM_JSON"),
                        help="print one task with a program applied, then exit")
    args = parser.parse_args()

    if args.show:
        show_task(args.show[0], args.show[1], json.loads(args.show[2]))
        return

    root = Path(args.data_root)
    jobs, missing = [], []
    for set_name, rel in ARC_SETS.items():
        files = sorted((root / rel).glob("*.json"))
        if not files:
            missing.append(set_name)
            continue
        jobs.extend((set_name, str(p)) for p in files[:args.limit])
    if missing:
        print(f"WARNING: no task files for {missing}", file=sys.stderr)

    started = time.time()
    per_set = {name: {} for name in ARC_SETS if name not in missing}
    with Pool(min(args.workers, 32)) as pool:
        for set_name, task_id, record in pool.imap_unordered(solve_task, jobs, chunksize=4):
            per_set[set_name][task_id] = record
    results = {name: aggregate(tasks) for name, tasks in per_set.items()}

    payload = {
        "meta": {
            "primitives_in_enumeration_order": list(PRIMITIVES),
            "object_step_order": "object index ascending, then primitive order",
            "max_program_length": {"A_whole_grid": MAX_LEN_WHOLE_GRID, "B": MAX_LEN_OBJECT},
            "object_count_range_for_B": [1, MAX_OBJECTS],
            "background_colour": 0,
            "translate_modes": {
                "own_size": "benchmark rule: move by the object's own bounding-box height or width",
                "one_cell": "not the benchmark rule: move by one cell",
            },
            "invalid_step": "out of bounds or overlap with another object makes the program invalid",
            "no_op_steps": "allowed in the search; engine_valid says whether the benchmark engine "
                           "would accept the program on every pair",
            "selection": "solved = first train-fitting program by (length, enumeration order) "
                         "passes all tests; solved_loose = any train-fitting program passes",
            "missing_sets": missing,
            "seconds": round(time.time() - started, 1),
        },
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)
    print_table(results)
    print(f"wrote {out_path} in {payload['meta']['seconds']} s")


if __name__ == "__main__":
    main()
