#!/usr/bin/env python
"""Checks for primitive_ceiling.py against the BabyTheorist implementation.

Usage:
    python scripts/arc/check_primitive_ceiling.py /path/to/babytheorist_checkout

1. Each of the eight primitives on a small hand-made object: our function vs
   babytheorist.primitives (grid and position).
2. Random scenes on a square world: our object_step (own-size translate,
   no-op steps invalid) vs babytheorist.engine.apply_step with by_index.
3. Whole-grid steps, segmentation order and the search on hand-made cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import primitive_ceiling as pc  # noqa: E402


def check_single_primitives(bt):
    box = np.array([[1, 2, 0],
                    [0, 9, 0]], dtype=np.int8)
    pos = (4, 5)
    expected_delta = {"translate_up": (-2, 0), "translate_down": (2, 0),
                      "translate_left": (0, -3), "translate_right": (0, 3)}
    for prim in pc.PRIMITIVES:
        theirs = bt.PRIMITIVES[prim].fn(bt.Shape(box, pos))
        scene = ((12, 12), [(box, pos[0], pos[1])])
        ours = pc.object_step(scene, (0, prim), "own_size")
        new_box, r, c = ours[1][0]
        assert np.array_equal(new_box, theirs.grid), prim
        assert (r, c) == tuple(theirs.position), prim
        if prim in expected_delta:
            dr, dc = expected_delta[prim]
            assert (r, c) == (pos[0] + dr, pos[1] + dc), prim
        print(f"  {prim:<16} -> pos {(r, c)} grid {new_box.tolist()}")
    # Hand-written expectations, independent of the benchmark code.
    assert pc.transform(box, "rot90").tolist() == [[0, 0], [2, 9], [1, 0]]
    assert pc.transform(box, "flip_h").tolist() == [[0, 9, 0], [1, 2, 0]]
    assert pc.transform(box, "flip_v").tolist() == [[0, 2, 1], [0, 9, 0]]
    assert pc.transform(box, "next_color").tolist() == [[2, 3, 0], [0, 1, 0]]


def random_scene(rng, world, n_objects):
    """Non-overlapping random objects with tight bounding boxes, raster sorted."""
    occupied = np.zeros((world, world), dtype=bool)
    objects = []
    for _ in range(200):
        if len(objects) == n_objects:
            break
        h, w = (int(v) for v in rng.integers(1, 4, size=2))
        box = rng.integers(0, 10, size=(h, w)).astype(np.int8)
        box[0, rng.integers(w)] = rng.integers(1, 10)
        box[h - 1, rng.integers(w)] = rng.integers(1, 10)
        box[rng.integers(h), 0] = rng.integers(1, 10)
        box[rng.integers(h), w - 1] = rng.integers(1, 10)
        r, c = (int(v) for v in rng.integers(0, world - 2, size=2))
        if r + h > world or c + w > world:
            continue
        if (occupied[r:r + h, c:c + w] & (box != 0)).any():
            continue
        occupied[r:r + h, c:c + w] |= box != 0
        objects.append((box, r, c))
    objects.sort(key=lambda o: (o[1], o[2]))
    return objects


def check_engine(bt, n_scenes=3000, seed=0):
    rng = np.random.default_rng(seed)
    n_steps = n_valid = n_invalid = 0
    for _ in range(n_scenes):
        world = int(rng.integers(5, 13))
        objects = random_scene(rng, world, int(rng.integers(1, 5)))
        if not objects:
            continue
        ours = ((world, world), objects)
        theirs = [bt.Shape(box, (r, c)) for box, r, c in objects]
        for _ in range(4):
            k = int(rng.integers(0, 4))
            prim = pc.PRIMITIVES[int(rng.integers(len(pc.PRIMITIVES)))]
            step = bt.Step(bt.Selector("by_index", k + 1), prim)
            new_theirs = bt.apply_step(step, theirs, world)
            new_ours = pc.object_step(ours, (k, prim), "own_size", noop_invalid=True)
            n_steps += 1
            assert (new_theirs is None) == (new_ours is None), (prim, k, objects)
            if new_theirs is None:
                n_invalid += 1
                break
            n_valid += 1
            grid_theirs, _ = bt.apply_program([], new_theirs, world)
            assert np.array_equal(pc.render(new_ours), grid_theirs), (prim, k)
            for (box, r, c), shape in zip(new_ours[1], new_theirs):
                assert np.array_equal(box, shape.grid) and (r, c) == tuple(shape.position)
            ours, theirs = new_ours, new_theirs
    print(f"  {n_steps} random steps agree with the engine "
          f"({n_valid} valid, {n_invalid} invalid)")


def check_whole_grid():
    g = np.array([[0, 0, 0, 0],
                  [0, 3, 9, 0],
                  [0, 0, 0, 0]], dtype=np.int8)
    assert pc.whole_grid_step(g, "rot90", "own_size").shape == (4, 3)
    assert pc.whole_grid_step(g, "flip_v", "own_size").tolist() == [
        [0, 0, 0, 0], [0, 9, 3, 0], [0, 0, 0, 0]]
    assert pc.whole_grid_step(g, "next_color", "own_size").tolist() == [
        [0, 0, 0, 0], [0, 4, 1, 0], [0, 0, 0, 0]]
    # Content is 1 x 2: own-size moves one row up, or two columns sideways.
    assert pc.whole_grid_step(g, "translate_up", "own_size").tolist() == [
        [0, 3, 9, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert pc.whole_grid_step(g, "translate_right", "own_size") is None
    assert pc.whole_grid_step(g, "translate_right", "one_cell").tolist() == [
        [0, 0, 0, 0], [0, 0, 3, 9], [0, 0, 0, 0]]
    assert pc.whole_grid_step(
        pc.whole_grid_step(g, "translate_right", "one_cell"), "translate_right", "one_cell") is None


def check_segmentation_and_search():
    g = np.array([[0, 0, 0, 5, 5, 0],
                  [1, 2, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 7]], dtype=np.int8)
    same = pc.segment(g, "same_colour_4conn")
    assert [(r, c, box.tolist()) for box, r, c in same] == [
        (0, 3, [[5, 5]]), (1, 0, [[1], [1]]), (1, 1, [[2]]), (3, 5, [[7]])]
    anyc = pc.segment(g, "any_colour_8conn")
    assert [(r, c, box.tolist()) for box, r, c in anyc] == [
        (0, 3, [[5, 5]]), (1, 0, [[1, 2], [1, 0]]), (3, 5, [[7]])]
    assert np.array_equal(pc.render((g.shape, same)), g)
    assert np.array_equal(pc.render((g.shape, anyc)), g)

    # Search recovers a planted two-step object program and a whole-grid one.
    def planted(x):
        scene = (x.shape, pc.segment(x, "any_colour_8conn"))
        for step in ((1, "flip_v"), (0, "translate_down")):
            scene = pc.object_step(scene, step, "own_size")
        return pc.render(scene)

    x2 = np.roll(g, 0).copy()
    x2[1, 1] = 4
    pairs = [(g, planted(g)), (x2, planted(x2))]
    res = pc.solve_objects(pairs, pairs[:1], "any_colour_8conn", "own_size")
    assert res["solved_strict"] and res["program"] == [[1, "translate_down"], [2, "flip_v"]], res
    whole = [(g, np.rot90(pc.transform(g, "next_color"), 1))]
    res = pc.solve_whole_grid(whole, whole, "own_size")
    assert res["solved_strict"] and res["program"] == ["rot90", "next_color"], res


def main():
    checkout = sys.argv[1]
    sys.path.insert(0, checkout)
    import babytheorist as bt

    print("single primitives vs babytheorist.primitives")
    check_single_primitives(bt)
    print("object steps vs babytheorist.engine.apply_step")
    check_engine(bt)
    check_whole_grid()
    print("whole-grid steps: ok")
    check_segmentation_and_search()
    print("segmentation and search: ok")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
