#!/usr/bin/env python3
"""Single-object tasks over more primitives than the benchmark has, in the benchmark's file layout.

BabyTheorist has 11 primitives that act on one object and its generator knows each of them by name, so it cannot be
given more. This generator uses the benchmark's own object shapes and its own primitive functions for the ones it has,
adds simple ones of the same kinds, and replaces the benchmark's symbolic program filter with a direct search:

  - a program is a sequence of primitives applied to the one object; a step must change the grid and keep the object
    inside it, and no grid may repeat along the way;
  - a pair is kept only if no shorter program over the same primitives turns its input into its output (breadth-first
    search over every shorter program);
  - an eval task has two support pairs and a query from the same program on three different objects. For programs of
    one or two primitives, every program of that length or shorter that reproduces the FIRST support pair must also give
    the query's answer, so transfer from one support pair is well defined; for longer programs only the shortest-program
    condition is checked.

Sets (each contains the smaller ones):
   8  the benchmark's base set: rot90, flip_h, flip_v, next_color, translate_up/down/left/right (by the object's own size)
  16  + shift_up/down/left/right (by one cell), rot180, rot270, transpose, prev_color
  24  + to_top/bottom/left/right (to the edge of the grid), anti_transpose, color_plus3, hollow, add_border
  32  + fill_bbox, invert_in_bbox, scale2, stretch_w, mirror_right, mirror_left, mirror_down, mirror_up
  52  + shift_<up|down|left|right>_<2..6>, moves by two to six cells (the one-cell moves are in 16)
  64  + color_plus2/4/5/6, shift_<up_left|up_right|down_left|down_right>_<1|2> (diagonal moves)
  shifts  a separate family for asking about amounts: shift_<up|down|left|right>_<1..6>, a move by that many cells

    python scripts/babytheorist/generate_many_primitives.py --primitives 32 --out data/babytheorist/many_p32 [--n-train 250000] [--n-eval 10000] [--workers 64]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BT_HOME = Path(os.environ.get("BT_HOME", "/home/minjun_dev/neo-vocabulary-restructuring/runs/external/babytheorist"))
sys.path.insert(0, str(BT_HOME))

from babytheorist.core import Shape  # noqa: E402
from babytheorist.primitives import PRIMITIVES as BENCHMARK  # noqa: E402
from babytheorist.shapes import ShapeFactory, bordered, patterned, place_shapes, solid  # noqa: E402

WORLD = 15
DEPTH_PROBS = {1: 0.30, 2: 0.50, 3: 0.12, 4: 0.08}  # the benchmark's level-1 depths, with everything from 4 up counted as 4


def tight(grid: np.ndarray, position: tuple[int, int]) -> Shape:
    rows, cols = np.nonzero(grid.any(axis=1))[0], np.nonzero(grid.any(axis=0))[0]
    if len(rows) == 0:
        raise ValueError("empty")
    return Shape(grid[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1], (position[0] + int(rows[0]), position[1] + int(cols[0])))


def recolor(step: int):
    def apply(s: Shape) -> Shape:
        grid = s.grid.copy()
        mask = grid != 0
        grid[mask] = (grid[mask] - 1 + step) % 9 + 1
        return Shape(grid, s.position)
    return apply


def shift(dr: int, dc: int):
    return lambda s: s.move_to((s.position[0] + dr, s.position[1] + dc))


def fill_bbox(s: Shape) -> Shape:
    if s.grid.all():
        raise ValueError("already full")
    return Shape(np.where(s.grid == 0, s.primary_color, s.grid), s.position)


def invert_in_bbox(s: Shape) -> Shape:
    return tight(np.where(s.grid == 0, s.primary_color, 0), s.position)


EXTRA = {
    "shift_up": shift(-1, 0), "shift_down": shift(1, 0), "shift_left": shift(0, -1), "shift_right": shift(0, 1),
    "rot180": lambda s: Shape(np.rot90(s.grid, 2), s.position), "rot270": lambda s: Shape(np.rot90(s.grid, 3), s.position),
    "transpose": lambda s: Shape(s.grid.T, s.position), "prev_color": recolor(-1),
    "to_top": lambda s: s.move_to((0, s.position[1])), "to_bottom": lambda s: s.move_to((WORLD - s.n_rows, s.position[1])),
    "to_left": lambda s: s.move_to((s.position[0], 0)), "to_right": lambda s: s.move_to((s.position[0], WORLD - s.n_cols)),
    "anti_transpose": lambda s: Shape(np.rot90(s.grid, 2).T, s.position), "color_plus3": recolor(3),
    "fill_bbox": fill_bbox, "invert_in_bbox": invert_in_bbox,
    "scale2": lambda s: Shape(np.kron(s.grid, np.ones((2, 2), dtype=np.int64)), s.position),
    "stretch_w": lambda s: Shape(np.repeat(s.grid, 2, axis=1), s.position),
    "mirror_right": lambda s: Shape(np.concatenate([s.grid, np.fliplr(s.grid)], axis=1), s.position),
    "mirror_left": lambda s: Shape(np.concatenate([np.fliplr(s.grid), s.grid], axis=1), (s.position[0], s.position[1] - s.n_cols)),
    "mirror_down": lambda s: Shape(np.concatenate([s.grid, np.flipud(s.grid)], axis=0), s.position),
    "mirror_up": lambda s: Shape(np.concatenate([np.flipud(s.grid), s.grid], axis=0), (s.position[0] - s.n_rows, s.position[1])),
}
SHIFT_DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
for _name, (_dr, _dc) in SHIFT_DIRECTIONS.items():
    for _k in range(1, 7):
        EXTRA[f"shift_{_name}_{_k}"] = shift(_k * _dr, _k * _dc)
SET_SHIFTS = tuple(f"shift_{name}_{k}" for name in SHIFT_DIRECTIONS for k in range(1, 7))
SET_8 = ("rot90", "flip_h", "flip_v", "next_color", "translate_up", "translate_down", "translate_left", "translate_right")
SET_16 = SET_8 + ("shift_up", "shift_down", "shift_left", "shift_right", "rot180", "rot270", "transpose", "prev_color")
SET_24 = SET_16 + ("to_top", "to_bottom", "to_left", "to_right", "anti_transpose", "color_plus3", "hollow", "add_border")
SET_32 = SET_24 + ("fill_bbox", "invert_in_bbox", "scale2", "stretch_w", "mirror_right", "mirror_left", "mirror_down", "mirror_up")
# two larger sets for asking how a model holds up as the environment grows; each contains the one before it
for _k in (2, 4, 5, 6):
    EXTRA[f"color_plus{_k}"] = recolor(_k)
for _name, (_dr, _dc) in {"up_left": (-1, -1), "up_right": (-1, 1), "down_left": (1, -1), "down_right": (1, 1)}.items():
    for _k in (1, 2):
        EXTRA[f"shift_{_name}_{_k}"] = shift(_k * _dr, _k * _dc)
SET_52 = SET_32 + tuple(f"shift_{name}_{k}" for name in SHIFT_DIRECTIONS for k in range(2, 7))   # shift_<direction> of SET_16 is the one-cell move
SET_64 = SET_52 + ("color_plus2", "color_plus4", "color_plus5", "color_plus6") + tuple(f"shift_{name}_{k}" for name in ("up_left", "up_right", "down_left", "down_right") for k in (1, 2))
SETS = {"8": SET_8, "16": SET_16, "24": SET_24, "32": SET_32, "52": SET_52, "64": SET_64, "shifts": SET_SHIFTS}


def function_of(name: str):
    return EXTRA[name] if name in EXTRA else BENCHMARK[name].fn


def render(s: Shape) -> np.ndarray | None:
    """The 15x15 grid with the object on it, or None when the object does not fit."""

    r, c = s.position
    if r < 0 or c < 0 or r + s.n_rows > WORLD or c + s.n_cols > WORLD or not s.grid.any():
        return None
    world = np.zeros((WORLD, WORLD), dtype=np.int8)
    world[r: r + s.n_rows, c: c + s.n_cols] = s.grid
    return world


def step(s: Shape, fn) -> tuple[Shape, np.ndarray] | None:
    try:
        after = fn(s)
    except (ValueError, IndexError):
        return None
    world = render(after)
    return None if world is None else (after, world)


def run(start: Shape, start_world: np.ndarray, functions: list) -> list[np.ndarray] | None:
    """Worlds after each step, or None if a step fails, changes nothing, or a grid repeats."""

    seen, worlds, s = {start_world.tobytes()}, [], start
    for fn in functions:
        out = step(s, fn)
        if out is None or out[1].tobytes() in seen:
            return None
        s = out[0]
        seen.add(out[1].tobytes())
        worlds.append(out[1])
    return worlds


def reachable_sooner(start: Shape, start_world: np.ndarray, target: bytes, functions: list, depth: int) -> bool:
    """True if some program shorter than `depth` turns the start grid into the target grid."""

    frontier, seen = [start], {start_world.tobytes()}
    for _ in range(depth - 1):
        following = []
        for s in frontier:
            for fn in functions:
                out = step(s, fn)
                if out is None:
                    continue
                key = out[1].tobytes()
                if key == target:
                    return True
                if key not in seen:
                    seen.add(key)
                    following.append(out[0])
        frontier = following
    return False


class Generator:
    def __init__(self, names: tuple[str, ...], seed, depths: dict[int, float] | None = None, without: tuple[str, ...] = (), start_steps: int = 0) -> None:
        self.names, self.functions = names, [function_of(n) for n in names]
        self.start_steps = start_steps  # a scene's object first goes through 0 to start_steps random primitives
        self.depths = depths or DEPTH_PROBS
        self.allowed = [i for i, n in enumerate(names) if n not in without]  # programs are drawn from these; the search for shorter programs uses all
        self.rng = np.random.default_rng(seed)
        self.factory = ShapeFactory(generators=(solid, bordered, patterned), size_range=(4, 16), compactness=8.0, bordered_hollow_min_size=9,
                                    bordered_hollow_random_inner_size=True, max_bbox_span=5)

    def scene(self) -> tuple[Shape, np.ndarray]:
        while True:
            placed = place_shapes([self.factory.sample(self.rng, tuple(range(1, 10)))], world_size=WORLD, gap=2, rng=self.rng)
            if placed is not None:
                shape = placed[1][0]
                world = render(shape)
                if world is not None:
                    return self.wander(shape, world) if self.start_steps else (shape, world)

    def wander(self, shape: Shape, world: np.ndarray) -> tuple[Shape, np.ndarray]:
        """The object after a random number (0 to start_steps) of random primitives, each changing the grid."""

        for _ in range(int(self.rng.integers(self.start_steps + 1))):
            for _ in range(50):
                out = step(shape, self.functions[int(self.rng.integers(len(self.functions)))])
                if out is not None and not np.array_equal(out[1], world):
                    shape, world = out
                    break
        return shape, world

    def pair(self, program: list[int]) -> tuple[Shape, np.ndarray, np.ndarray] | None:
        """A fresh object on which the program runs and is the shortest explanation of its own result."""

        functions = [self.functions[i] for i in program]
        for _ in range(40):
            shape, world = self.scene()
            worlds = run(shape, world, functions)
            if worlds is not None and not reachable_sooner(shape, world, worlds[-1].tobytes(), self.functions, len(program)):
                return shape, world, worlds[-1]
        return None

    def agrees(self, support: tuple, query: tuple, depth: int) -> bool:
        """Every program of at most `depth` primitives that reproduces the support pair also answers the query."""

        (s_shape, s_world, s_out), (q_shape, q_world, q_out) = support, query
        target, answer = s_out.tobytes(), q_out.tobytes()
        candidates = [[f] for f in self.functions] + ([[f, g] for f in self.functions for g in self.functions] if depth >= 2 else [])
        for functions in candidates:
            worlds = run(s_shape, s_world, functions)
            if worlds is None or worlds[-1].tobytes() != target:
                continue
            replay = run(q_shape, q_world, functions)
            if replay is None or replay[-1].tobytes() != answer:
                return False
        return True

    def task(self, pairs: int) -> tuple[list[int], list[tuple]] | None:
        """One task. The program is drawn first, uniformly over the primitives, and objects are then searched for it, so
        that a primitive that only suits some objects (a flip needs an object that is not symmetric) is as frequent as any
        other. Programs that are never the shortest explanation (two quarter turns, a move and its opposite) find no object
        and are dropped."""

        depth = int(self.rng.choice(list(self.depths), p=list(self.depths.values())))
        program = [int(self.allowed[i]) for i in self.rng.integers(len(self.allowed), size=depth)]
        for _ in range(8 if pairs > 1 else 1):
            made = []
            for _ in range(pairs):
                found = self.pair(program)
                if found is None:
                    return None
                made.append(found)
            if pairs == 1 or depth > 2 or self.agrees(made[0], made[-1], depth):
                return program, made
        return None


def work(job: tuple) -> dict:
    names, seed, count, pairs, depths, without, start_steps = job
    generator = Generator(names, seed, depths, without, start_steps)
    inputs, outputs, programs, attempts = [], [], [], 0
    while len(programs) < count:
        attempts += 1
        task = generator.task(pairs)
        if task is None:
            continue
        program, made = task
        inputs.append(np.stack([m[1] for m in made]))
        outputs.append(np.stack([m[2] for m in made]))
        programs.append(program)
    return {"inputs": np.stack(inputs), "outputs": np.stack(outputs), "programs": programs, "attempts": attempts}


def pair_hash(x: np.ndarray, y: np.ndarray) -> bytes:
    return hashlib.blake2b(x.tobytes() + y.tobytes(), digest_size=12).digest()


def generate(names: tuple[str, ...], count: int, pairs: int, seed: int, workers: int, exclude: set[bytes], depths: dict[int, float] | None = None,
             without: tuple[str, ...] = (), start_steps: int = 0) -> dict:
    chunk = max(50, min(2000, count // (workers * 4) or 50))
    inputs, outputs, programs, attempts, seen, next_seed = [], [], [], 0, set(exclude), 0
    with Pool(workers) as pool:
        while len(programs) < count:
            need = count - len(programs)
            jobs = [(names, [seed, next_seed + i], chunk, pairs, depths, without, start_steps) for i in range(max(workers, -(-need // chunk)))]
            next_seed += len(jobs)
            for result in pool.imap_unordered(work, jobs):
                attempts += result["attempts"]
                for x, y, p in zip(result["inputs"], result["outputs"], result["programs"]):
                    keys = [pair_hash(x[i], y[i]) for i in range(len(x))]
                    if any(k in seen for k in keys):
                        continue
                    seen.update(keys)
                    inputs.append(x); outputs.append(y); programs.append(p)  # noqa: E702
    inputs, outputs, programs = inputs[:count], outputs[:count], programs[:count]
    return {"inputs": np.stack(inputs), "outputs": np.stack(outputs), "programs": programs, "attempts": attempts, "hashes": seen}


def write(path: Path, names: tuple[str, ...], data: dict, pairs: int) -> None:
    count = len(data["programs"])
    suites = [json.dumps([{"selector": {"type": "by_index", "param": 1}, "primitive": names[i], "target": None} for i in p]) for p in data["programs"]]
    depths = np.array([len(p) for p in data["programs"]], dtype=np.int32)
    with h5py.File(path, "w") as f:
        f["inputs"], f["outputs"], f["depths"] = data["inputs"].astype(np.int8), data["outputs"].astype(np.int8), depths
        f["grid_sizes"] = np.full((count, 2), WORLD, dtype=np.int32)
        f["n_shapes"] = np.ones((count, pairs), dtype=np.int32)
        if pairs > 1:
            f["n_support"] = np.full(count, pairs - 1, dtype=np.int32)
        f.create_dataset("transformation_suites", data=suites, dtype=h5py.string_dtype())
    used = Counter(names[i] for p in data["programs"] for i in p)
    info = {"n_tasks": count, "pairs_per_task": pairs, "primitives": list(names), "depth_counts": {str(d): int((depths == d).sum()) for d in sorted(set(depths.tolist()))},
            "primitive_counts": {n: used.get(n, 0) for n in names}, "attempts": data["attempts"], "generator": "scripts/babytheorist/generate_many_primitives.py"}
    path.with_suffix(".info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"{path}: {count} tasks, depths {info['depth_counts']}, {data['attempts']} attempts", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primitives", choices=sorted(SETS), required=True)
    parser.add_argument("--depths", default="", help='depth distribution, e.g. "1" for single-primitive tasks only or "1:0.5,2:0.5"; default is the level-1 mix')
    parser.add_argument("--train-without", default="", help="comma-separated primitives left out of the train file (the eval file has all)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=250000)
    parser.add_argument("--n-eval", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-pairs", type=int, default=1, help="pairs per train task, all from the same program on different objects")
    parser.add_argument("--train-only", action="store_true", help="write the train file only")
    parser.add_argument("--exclude-from", type=Path, help="an eval file whose pairs the train file must not contain")
    parser.add_argument("--start-steps", type=int, default=0, help="train file only: each pair's object first goes through 0 to this many "
                        "random primitives, so inputs include the intermediate states of longer programs (the eval file is unchanged)")
    args = parser.parse_args()
    names = SETS[args.primitives]
    depths = None
    if args.depths:
        parts = [part.split(":") for part in args.depths.split(",")]
        weights = {int(p[0]): float(p[1]) if len(p) > 1 else 1.0 for p in parts}
        depths = {d: w / sum(weights.values()) for d, w in weights.items()}
    without = tuple(n for n in args.train_without.split(",") if n)
    unknown = [n for n in without if n not in names]
    if unknown:
        raise SystemExit(f"--train-without names primitives that are not in the set: {unknown}")
    args.out.mkdir(parents=True, exist_ok=True)
    exclude = set()
    if args.exclude_from:
        with h5py.File(args.exclude_from, "r") as f:
            ins, outs = f["inputs"][:], f["outputs"][:]
        exclude = {pair_hash(ins[i, j].astype(np.int8), outs[i, j].astype(np.int8)) for i in range(len(ins)) for j in range(ins.shape[1])}
    train = generate(names, args.n_train, args.train_pairs, args.seed, args.workers, exclude, depths, without, args.start_steps)
    write(args.out / "lv1_train.h5", names, train, args.train_pairs)
    if args.train_only:
        return
    evaluation = generate(names, args.n_eval, 3, args.seed + 1, args.workers, train["hashes"], depths)
    write(args.out / "lv1_eval.h5", names, evaluation, 3)


if __name__ == "__main__":
    main()
