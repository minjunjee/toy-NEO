#!/usr/bin/env python3
"""Print what a per-task sleep answered for one task, as text grids (pad cells shown as dots): the demonstration pairs, then
for each test the input, the answer, the most frequent guess, and where the guess differs from the answer.

    python scripts/arc/show_guess.py runs/x/dump/task012.npz
"""

from __future__ import annotations

import sys

import numpy as np


def text(grid: np.ndarray) -> list[str]:
    rows = [r for r in grid if (r != 10).any()]
    width = max((int((r != 10).sum()) for r in rows), default=0)
    return ["".join("." if v == 10 else str(int(v)) for v in r[:width]) for r in rows]


def side_by_side(blocks: list[tuple[str, list[str]]]) -> None:
    height = max(len(lines) for _, lines in blocks)
    widths = [max([len(title)] + [len(line) for line in lines]) for title, lines in blocks]
    print("   ".join(title.ljust(w) for (title, _), w in zip(blocks, widths)))
    for i in range(height):
        print("   ".join((lines[i] if i < len(lines) else "").ljust(w) for (_, lines), w in zip(blocks, widths)))
    print()


def main() -> None:
    for path in sys.argv[1:]:
        f = np.load(path)
        print("=" * 20, path)
        for k, (x, y) in enumerate(zip(f["demo_inputs"], f["demo_outputs"])):
            side_by_side([(f"demo {k} input", text(x)), ("output", text(y))])
        for k, (x, y, g) in enumerate(zip(f["test_inputs"], f["answers"], f["guesses"])):
            wrong = np.where((g != y) & (y != 10), 11, np.where(y == 10, 10, 12))
            marks = ["".join("." if v == 10 else ("X" if v == 11 else "-") for v in r[: int((r != 10).sum())]) for r in wrong if (r != 10).any()]
            same_shape = ((g != 10) == (y != 10)).all()
            side_by_side([(f"test {k} input", text(x)), ("answer", text(y)), ("guess", text(g)), ("wrong cells" if same_shape else "shape differs", marks if same_shape else [])])


if __name__ == "__main__":
    main()
