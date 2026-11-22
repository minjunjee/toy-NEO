#!/usr/bin/env python3
"""Figures for reading how models hold up as the environment grows. Everything is drawn against training steps or
against the number of primitives in the environment, never against compute.

  curves_by_size_<model>.png   one model, one line per environment size: accuracy on the training batch (every 50 steps,
                               smoothed) and transfer on held-out inputs (every 2,400 steps) against training steps
  curves_by_model.png          one panel per environment size, one line per model: the same two quantities
  steps_to_reach.png           training steps until the smoothed training accuracy first reaches 50%, 80%, 95%, against
                               the number of primitives (log axis), one line per model; "never" is marked at the top
  final_by_size.png            at the end of training: transfer by kind of task (one primitive, programs of 2, 3, 4
                               primitives, primitives left out of training) against the number of primitives, one line
                               per model; from scripts/babytheorist/scale_table.py
  by_depth.png                 one panel per environment size: transfer against the depth of the hidden program
  shift_family.png             the small shift family (24 primitives, four-cell moves left out): explained share against
                               steps for every condition of that experiment

A run that does not exist yet is skipped, so the script can be run again as runs finish.

    python scripts/babytheorist/scaling_figures.py [--out docs/reports/2026-11-22-scaling]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
SIZES = (8, 16, 24, 32, 52, 64)
MODELS = {  # label -> (color, {size: run})
    "lab NEO, one code": (SERIES[0], {8: "onestep_s_m8_wide_s42", 16: "onestep_s_m16_wide_s42", 24: "onestep_s_m24_wide_s42", 32: "onestep_s_m32_wide_s42",
                                      52: "ladder_lab_p52_s42", 64: "ladder_lab_p64_s42"}),
    "one code x count": (SERIES[1], {n: f"ladder_repeat_p{n}_s42" for n in SIZES}),
    "one code x count, no gradient between applications": (SERIES[3], {n: f"ladder_repeatcut_p{n}_s42" for n in SIZES}),
    "(kind, count), fixed-angle rotation": (SERIES[2], {n: f"ladder_counted_p{n}_s42" for n in SIZES}),
}
SHIFT_RUNS = {  # the small shift family: label -> (run, color, dashes)
    "lab NEO": ("onestep_k_flat_s42", SERIES[0], "-"), "rotation-tied codebook": ("onestep_k_rot_s42", SERIES[6], "-"),
    "rotation-tied + addition loss": ("onestep_k_rot_add_s42", SERIES[7], "-"), "free vectors + addition loss": ("onestep_k_free_add_s42", SERIES[4], "-"),
    "one code x count, 4 codes": ("repeat_c4_s42", SERIES[3], ":"), "one code x count, 6 codes": ("repeat_c6_s42", SERIES[3], "--"),
    "one code x count, 8 codes": ("repeat_c8_s42", SERIES[1], "--"), "one code x count, 12 codes": ("repeat_c12_s42", SERIES[1], "-"),
    "one code x count, 12 codes, no gradient between": ("repeat_c12_cut_s42", SERIES[1], ":"),
    "(kind, count), learned rotation (stopped)": ("counted_rot_s42", SERIES[2], ":"), "(kind, count), fixed-angle rotation": ("counted_circle_s42", SERIES[2], "-"),
    "(kind, count), free vectors": ("counted_free_s42", SERIES[5], "--"),
}


def read(run: str, split: str = "lv1") -> dict | None:
    path = REPO / "runs" / run / "train.jsonl"
    if not path.exists() or not path.stat().st_size:
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    train = [(r["step"], r["grid_acc"]) for r in rows if "grid_acc" in r]
    evals = [(r["step"], r[f"observe/{split}/explain_depth1"], r[f"observe/{split}/transfer_strict_depth1"]) for r in rows if f"observe/{split}/explain_depth1" in r]
    return {"train": train, "evals": evals, "done": (path.parent / "DONE").exists()} if train else None


def smooth(points: list[tuple], window: int = 10) -> tuple[list, list]:
    steps, values = [p[0] for p in points], [p[1] for p in points]
    out = [sum(values[max(0, i - window + 1): i + 1]) / len(values[max(0, i - window + 1): i + 1]) for i in range(len(values))]
    return steps, out


def style(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def percent(ax) -> None:
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])


def save(fig, out: Path, name: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(out / name, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out / name)


def ramp(color: str, count: int) -> list:
    """Light to dark steps of one hue, for an ordered quantity."""

    import matplotlib.colors as mc

    base = mc.to_rgb(color)
    return [tuple(1 - (1 - c) * (0.35 + 0.65 * i / max(1, count - 1)) for c in base) for i in range(count)]


def curves_by_size(out: Path) -> None:
    for label, (color, runs) in MODELS.items():
        have = [(n, read(r)) for n, r in runs.items()]
        have = [(n, d) for n, d in have if d]
        if not have:
            continue
        shades = dict(zip([n for n, _ in have], ramp(color, len(have))))
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for n, d in have:
            steps, values = smooth(d["train"])
            axes[0].plot(steps, values, color=shades[n], linewidth=1.8, label=f"{n} primitives")
            if d["evals"]:
                axes[1].plot([e[0] for e in d["evals"]], [e[2] for e in d["evals"]], color=shades[n], linewidth=1.8, marker="o", markersize=3.5, label=f"{n} primitives")
        style(axes[0], "training steps", "training pairs reproduced exactly", "On the training batch (smoothed over 500 steps)")
        style(axes[1], "training steps", "transfer, single-primitive tasks", "On new inputs: the code from one pair replayed on another")
        for ax in axes:
            percent(ax)
        axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
        fig.suptitle(f"{label}: one line per environment size", color=INK, fontsize=11, x=0.01, ha="left")
        save(fig, out, f"curves_by_size_{re.sub('[^a-z0-9]+', '_', label.lower()).strip('_')}.png")


def curves_by_model(out: Path) -> None:
    sizes = [n for n in SIZES if any(read(runs.get(n, "")) for _, runs in MODELS.values())]
    if not sizes:
        return
    fig, axes = plt.subplots(2, len(sizes), figsize=(3.3 * len(sizes), 5.6), squeeze=False, sharey=True)
    for column, n in enumerate(sizes):
        for label, (color, runs) in MODELS.items():
            d = read(runs.get(n, ""))
            if not d:
                continue
            steps, values = smooth(d["train"])
            axes[0][column].plot(steps, values, color=color, linewidth=1.8, label=label)
            if d["evals"]:
                axes[1][column].plot([e[0] for e in d["evals"]], [e[2] for e in d["evals"]], color=color, linewidth=1.8, marker="o", markersize=3, label=label)
        style(axes[0][column], "", "training pairs reproduced" if column == 0 else "", f"{n} primitives")
        style(axes[1][column], "training steps", "transfer, one primitive" if column == 0 else "")
        percent(axes[0][column])
        percent(axes[1][column])
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.suptitle("Learning curves at every environment size (top: training batch, bottom: transfer to new inputs)", color=INK, fontsize=11, x=0.01, ha="left")
    save(fig, out, "curves_by_model.png")


def steps_to_reach(out: Path) -> None:
    levels = (0.5, 0.8, 0.95)
    fig, axes = plt.subplots(1, len(levels), figsize=(4.0 * len(levels), 3.8), sharey=True)
    drawn = False
    for ax, level in zip(axes, levels):
        top = 26000
        for label, (color, runs) in MODELS.items():
            xs, ys, never = [], [], []
            for n in SIZES:
                d = read(runs.get(n, ""))
                if not d:
                    continue
                steps, values = smooth(d["train"])
                reached = next((s for s, v in zip(steps, values) if v >= level), None)
                if reached is None:
                    never.append(n)
                else:
                    xs.append(n)
                    ys.append(reached)
            if xs:
                ax.plot(xs, ys, color=color, linewidth=1.8, marker="o", markersize=5, label=label)
                drawn = True
            if never:
                ax.scatter(never, [top] * len(never), color=color, marker="x", s=36, linewidths=1.6)
        style(ax, "primitives in the environment (log scale)", "training steps" if level == levels[0] else "", f"until {int(level * 100)}% of training pairs are reproduced")
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(SIZES))
        ax.set_xticklabels([str(n) for n in SIZES])
        ax.set_ylim(0, top + 1500)
        ax.axhline(top, color=GRID, linewidth=0.8, linestyle="--")
        ax.text(SIZES[0], top + 250, "x = not reached in 24,000 steps", color=MUTED, fontsize=7.5, va="bottom")
    if drawn:
        axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="center left")
        fig.suptitle("How much training it takes as the environment grows (a straight line here is growth like a logarithm)", color=INK, fontsize=11, x=0.01, ha="left")
        save(fig, out, "steps_to_reach.png")
    plt.close(fig)


def scale_tables(out: Path) -> dict:
    """run name -> groups, from every scale_*.json next to the figures."""

    tables = {}
    for file in sorted(out.glob("scale_*.json")):
        for entry in json.loads(file.read_text()):
            tables[entry["run"]] = entry["groups"]
    return tables


def final_by_size(out: Path) -> None:
    tables = scale_tables(out)
    kinds = [("depth 1", "own_transfer", "one primitive\n(trained)"), ("depth 2", "stepwise_transfer", "programs of 2 primitives\n(never trained)"),
             ("depth 3", "stepwise_transfer", "programs of 3 primitives\n(never trained)"), ("depth 4", "stepwise_transfer", "programs of 4 primitives\n(never trained)"),
             ("depth 1, uses a primitive left out of training", "own_transfer", "one primitive that was\nleft out of training")]
    kinds = [k for k in kinds if any(runs.get(n) in tables and k[0] in tables[runs[n]] for _, runs in MODELS.values() for n in SIZES)]
    if not kinds:
        return
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.4 * len(kinds), 3.9), sharey=True, squeeze=False)
    axes = axes[0]
    drawn = False
    for ax, (group, key, title) in zip(axes, kinds):
        for label, (color, runs) in MODELS.items():
            points = [(n, tables[runs[n]][group][key]) for n in SIZES if runs.get(n) in tables and group in tables[runs[n]]]
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=1.8, marker="o", markersize=5, label=label)
                drawn = True
        style(ax, "primitives in the environment", "transfer" if ax is axes[0] else "", title)
        ax.title.set_fontsize(9)
        ax.set_xlim(7, 72)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(SIZES))
        ax.set_xticklabels([str(n) for n in SIZES])
        percent(ax)
    if drawn:
        axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
        fig.suptitle("At the end of training: transfer by kind of task as the environment grows", color=INK, fontsize=11, x=0.01, y=1.08, ha="left")
        save(fig, out, "final_by_size.png")
    plt.close(fig)


def by_depth(out: Path) -> None:
    tables = scale_tables(out)
    sizes = [n for n in SIZES if any(runs.get(n) in tables for _, runs in MODELS.values())]
    if not sizes:
        return
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.1 * len(sizes), 3.5), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], sizes):
        for label, (color, runs) in MODELS.items():
            groups = tables.get(runs.get(n, ""))
            if not groups:
                continue
            depths = [d for d in (1, 2, 3, 4) if f"depth {d}" in groups]
            ax.plot(depths, [max(groups[f"depth {d}"]["own_transfer"], groups[f"depth {d}"]["stepwise_transfer"]) if d == 1 else groups[f"depth {d}"]["stepwise_transfer"] for d in depths],
                    color=color, linewidth=1.8, marker="o", markersize=5, label=label)
        style(ax, "primitives in the hidden program", "transfer" if ax is axes[0][0] else "", f"{n} primitives")
        ax.set_xticks([1, 2, 3, 4])
        percent(ax)
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.suptitle("Transfer against the length of the hidden program (trained on single primitives only)", color=INK, fontsize=11, x=0.01, ha="left")
    save(fig, out, "by_depth.png")


def per_primitive(run: str) -> dict | None:
    """primitive -> transfer on its single-primitive eval tasks, from the run's last full evaluation."""

    path = REPO / "runs" / run / "train.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if "one_step_tasks" in line]
    rows = [r for r in rows if "observe_full/lv1/one_step_tasks" in r]
    if not rows:
        return None
    return {name: v["transfer_strict"] / v["tasks"] for name, v in rows[-1]["observe_full/lv1/one_step_tasks"].items() if v["tasks"]}


def blocks() -> list[tuple[str, tuple]]:
    """The primitives that each size adds to the one before it, in order."""

    sys.path.insert(0, str(REPO / "scripts" / "babytheorist"))
    import generate_many_primitives as generator

    out, before = [], ()
    for n in SIZES:
        names = generator.SETS[str(n)]
        out.append((f"added at {n}" if before else f"first {n}", tuple(x for x in names if x not in before and x not in HELD_OUT)))
        before = names
    return out


HELD_OUT = ("shift_up_4", "shift_down_4", "shift_left_4", "shift_right_4")   # left out of the train files of 52 and 64: not in any block
BATCH = 128


def trained_count(n: int) -> int:
    """How many primitives the train file of the environment of size n has (the four-cell moves are left out from 52 on)."""

    return n - len(HELD_OUT) if n >= 52 else n


def old_primitives(out: Path) -> None:
    """Does adding primitives to the environment hurt the ones that were there? Transfer on each block of primitives,
    measured in every environment that contains it."""

    groups = blocks()
    models = [(label, color, runs) for label, (color, runs) in MODELS.items() if any(per_primitive(runs.get(n, "")) for n in SIZES)]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.9), sharey=True, squeeze=False)
    for ax, (label, color, runs) in zip(axes[0], models):
        shades = ramp(color, len(groups))
        for (name, members), shade in zip(groups, shades):
            points = []
            for n in SIZES:
                table = per_primitive(runs.get(n, ""))
                if table and all(m in table for m in members):
                    points.append((n, sum(table[m] for m in members) / len(members)))
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], color=shade, linewidth=1.8, marker="o", markersize=5, label=f"primitives {name}")
        style(ax, "primitives in the environment", "transfer on that block's single-primitive tasks" if ax is axes[0][0] else "", label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(SIZES))
        ax.set_xticklabels([str(n) for n in SIZES])
        ax.set_xlim(7, 72)
        percent(ax)
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower left")
    fig.suptitle("What happens to the primitives already there when more are added (a flat line: adding did not hurt them)", color=INK, fontsize=11, x=0.01, y=1.04, ha="left")
    save(fig, out, "old_primitives.png")


GROW_MODELS = {"lab NEO, one code": (SERIES[0], "lab"), "one code x count": (SERIES[1], "repeat"),
               "one code x count, no gradient between applications": (SERIES[3], "repeatcut"),
               "(kind, count), fixed-angle rotation (Lie algebra)": (SERIES[2], "counted"),
               "one network per code": (SERIES[4], "expert"),
               "shared executor + per-code low-rank change, shared part slow": (SERIES[6], "lora"),
               "lab NEO, control: the environment stays at 8": (MUTED, "labsame")}
STAGE_STEPS = 6000


def growing_points(model: str) -> list[tuple[int, int, dict]]:
    """(environment size, steps since the growing run began, primitive -> transfer) for every light evaluation of a
    growing-environment chain, in order."""

    points = []
    for index, n in enumerate(SIZES):
        folder = REPO / "runs" / f"grow_{model}_p{n}_s42" / "eval"
        for file in sorted(folder.glob("observe_lv1_step*.json")) if folder.exists() else []:
            d = json.loads(file.read_text())
            table = d.get("one_step_tasks")
            if table:
                points.append((n, index * STAGE_STEPS + int(file.stem.split("step")[1]), {k: v["transfer_strict"] / v["tasks"] for k, v in table.items() if v["tasks"]}))
    return points


def growing(out: Path) -> None:
    """One model learning while the environment grows: transfer on each block of primitives against the steps since the
    start, the environment growing at the dashed lines; and how fast each newly added block is learned."""

    groups = blocks()
    have = [(label, color, model, growing_points(model)) for label, (color, model) in GROW_MODELS.items()]
    have = [h for h in have if h[3]]
    if not have:
        return
    fig, axes = plt.subplots(len(have), 1, figsize=(12, 3.3 * len(have)), sharex=True, squeeze=False)
    for ax, (label, color, model, points) in zip(axes[:, 0], have):
        shades = ramp(color, len(groups))
        for (name, members), shade in zip(groups, shades):
            line = [(t, sum(table[m] for m in members) / len(members)) for n, t, table in points if all(m in table for m in members)]
            if line:
                ax.plot([p[0] for p in line], [p[1] for p in line], color=shade, linewidth=1.8, marker="o", markersize=3, label=f"primitives {name}")
        for index, n in enumerate(SIZES):
            ax.axvline(index * STAGE_STEPS, color=MUTED, linewidth=0.8, linestyle="--")
            ax.text(index * STAGE_STEPS + 120, 1.0, f"{n}", color=MUTED, fontsize=8, va="top")
        style(ax, "training steps since the environment had 8 primitives" if ax is axes[-1, 0] else "", "transfer, one primitive", label)
        percent(ax)
        ax.set_xlim(0, len(SIZES) * STAGE_STEPS)
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower right", ncol=2)
    fig.suptitle("An environment that grows (dashed lines: the number of primitives grows to the number shown) while one model keeps learning",
                 color=INK, fontsize=11, x=0.01, y=0.995, ha="left")
    fig.subplots_adjust(top=0.92, hspace=0.35)
    save(fig, out, "growing.png")

    levels = (0.5, 0.8)
    fig, axes = plt.subplots(1, len(levels) + 2, figsize=(20, 3.8))
    for label, color, model, points in have:
        examples = []
        for index, (n, (name, members)) in enumerate(zip(SIZES, groups)):
            stage = [(t - index * STAGE_STEPS, sum(table[m] for m in members) / len(members)) for size, t, table in points if size == n and all(m in table for m in members)]
            reached = next((t for t, v in stage if v >= 0.5), None)
            if reached is not None and index > 0:
                examples.append((n, reached * BATCH / trained_count(n)))   # training pairs of each new primitive seen until then
        if examples:
            axes[2].plot([e[0] for e in examples], [e[1] / 1000 for e in examples], color=color, linewidth=1.8, marker="o", markersize=5, label=label)
        for ax, level in zip(axes, levels):
            xs, ys, never = [], [], []
            for index, (n, (name, members)) in enumerate(zip(SIZES, groups)):
                stage = [(t - index * STAGE_STEPS, sum(table[m] for m in members) / len(members)) for size, t, table in points if size == n and all(m in table for m in members)]
                if not stage:
                    continue
                reached = next((t for t, v in stage if v >= level), None)
                if reached is None and stage[-1][0] >= STAGE_STEPS:   # "not reached" only once the stage has ended
                    never.append(n)
                elif reached is None:
                    continue
                else:
                    xs.append(n)
                    ys.append(reached)
            if xs:
                ax.plot(xs, ys, color=color, linewidth=1.8, marker="o", markersize=5, label=label)
            if never:
                ax.scatter(never, [STAGE_STEPS + 300] * len(never), color=color, marker="x", s=36, linewidths=1.6)
        kept = []
        for index, n in enumerate(SIZES):
            stage = [(t, table) for size, t, table in points if size == n]
            if stage and stage[-1][0] >= (index + 1) * STAGE_STEPS:   # completed stages only
                members = groups[0][1]
                kept.append((n, sum(stage[-1][1][m] for m in members) / len(members)))
        if kept:
            axes[-1].plot([k[0] for k in kept], [k[1] for k in kept], color=color, linewidth=1.8, marker="o", markersize=5, label=label)
    for ax, level in zip(axes, levels):
        style(ax, "primitives in the environment after it grew", "steps after the growth" if ax is axes[0] else "", f"until the new primitives reach {int(level * 100)}% transfer")
        ax.set_ylim(0, STAGE_STEPS + 800)
        ax.text(SIZES[0], STAGE_STEPS + 450, "x = not reached in this stage", color=MUTED, fontsize=7.5)
    style(axes[2], "primitives in the environment after it grew", "thousand pairs of each new primitive", "pairs of each new primitive seen until 50%")
    axes[2].set_ylim(0, None)
    style(axes[-1], "primitives in the environment", "transfer", "the first 8 primitives at the end of each stage")
    percent(axes[-1])
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(SIZES))
        ax.set_xticklabels([str(n) for n in SIZES])
        ax.set_xlim(7, 72)
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    fig.suptitle("How fast the newly added primitives are learned, and whether the first ones are kept", color=INK, fontsize=11, x=0.01, y=1.03, ha="left")
    save(fig, out, "growing_speed.png")


HELDOUT_RUNS = {  # run -> (label, color, dashes)
    "counted_circle_s42": ("(kind, count), fixed-angle rotation (Lie algebra)", SERIES[2], "-"),
    "counted_rot_s42": ("(kind, count), learned rotation (stopped at 7,200)", SERIES[2], ":"),
    "counted_free_s42": ("(kind, count), free vectors", SERIES[5], "--"),
    "repeat_c12_cut_s42": ("one code x count, no gradient between", SERIES[1], "-"),
    "repeat_c12_g0_s42": ("one code x count, grounding off", SERIES[1], "--"),
}


def heldout_curve(out: Path) -> None:
    path = out / "heldout_curve.json"
    if not path.exists():
        return
    report = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, group, title in ((axes[0], "held out", "Four-cell moves, never in training"), (axes[1], "trained", "The 20 shifts that were in training")):
        for run, (label, color, dashes) in HELDOUT_RUNS.items():
            points = report.get(run)
            if points:
                ax.plot([p["step"] for p in points], [p[f"{group} transfer"] for p in points], color=color, linestyle=dashes, linewidth=1.8, marker="o", markersize=4, label=label)
        style(ax, "training steps", "transfer" if ax is axes[0] else "", title)
        percent(ax)
        ax.set_xlim(0, 25000)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center", bbox_to_anchor=(-0.08, -0.16), ncol=2)
    fig.suptitle("Shift family: can a model name an amount it never saw? (the programmer's own program, measured at each saved checkpoint)",
                 color=INK, fontsize=11, x=0.01, y=1.03, ha="left")
    save(fig, out, "heldout_curve.png")


LENGTH_RUNS = [  # (label, [(run, step offset)], color, dashes)
    ("one step, then programs of up to 2 on depth <= 2, then up to 4 on depth <= 4", [("fastneo_3k_len2_single_a_lv1_s42", 0), ("fastneo_3k_len4_single_a_lv1_s42", 3000)], SERIES[2], "-"),
    ("the same with learning rate 3e-5 and no grounding term", [("fastneo_3k_len2_lr3e5_g0_single_a_lv1_s42", 0), ("fastneo_3k_len4_lr3e5_g0_single_a_lv1_s42", 3000)], SERIES[2], "--"),
    ("one step, then programs of up to 2 on pairs of every depth", [("fastneo_3k_len2_alldata_single_a_lv1_s42", 0)], SERIES[3], "-"),
    ("one step, then programs of up to 5 on depth <= 2", [("fastneo_3k_len5_depth2_single_a_lv1_s42", 0)], SERIES[4], "-"),
    ("one step, then straight to programs of up to 5 on every depth", [("fastneo_6k_warm_all_single_a_lv1_s42", 0)], SERIES[7], "-"),
    ("the same with learning rate 3e-5 and no grounding term", [("fastneo_3k_warm_all_g0_lr3e5_single_a_lv1_s42", 0)], SERIES[7], "--"),
]


def program_length(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    drawn = False
    for label, parts, color, dashes in LENGTH_RUNS:
        xs, ys = [0], [0.832]
        for run, offset in parts:
            path = REPO / "runs" / run / "train.jsonl"
            if not path.exists():
                continue
            for row in (json.loads(line) for line in path.read_text().splitlines()):
                if "observe/lv1/transfer_strict" in row:
                    xs.append(offset + row["step"])
                    ys.append(row["observe/lv1/transfer_strict"])
        if len(xs) > 1:
            ax.plot(xs, ys, color=color, linestyle=dashes, linewidth=1.8, marker="o", markersize=3, label=label)
            drawn = True
    ax.axhline(0.832, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(150, 0.845, "the one-step model it starts from: 83%", color=MUTED, fontsize=8)
    ax.axvline(3000, color=GRID, linewidth=1.0)
    style(ax, "training steps after the one-step model", "transfer, level 1, all depths (5-step rollout, grid only)", "")
    percent(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.suptitle("Longer programs, gradually or at once (8 primitives, seed 42)", color=INK, fontsize=11, x=0.01, y=1.0, ha="left")
    if drawn:
        save(fig, out, "program_length.png")
    plt.close(fig)


def shift_family(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), sharey=True)
    groups = (("Codebook changes, the executor applied once", ("lab NEO", "rotation-tied codebook", "rotation-tied + addition loss", "free vectors + addition loss")),
              ("One code applied a counted number of times", ("one code x count, 4 codes", "one code x count, 6 codes", "one code x count, 8 codes", "one code x count, 12 codes",
                                                              "one code x count, 12 codes, no gradient between")),
              ("The programmer names (kind, count); the count is the number of applications",
               ("(kind, count), learned rotation (stopped)", "(kind, count), fixed-angle rotation", "(kind, count), free vectors")))
    for ax, (title, labels) in zip(axes, groups):
        for label in labels:
            run, color, dashes = SHIFT_RUNS[label]
            d = read(run)
            if d and d["evals"]:
                ax.plot([e[0] for e in d["evals"]], [e[1] for e in d["evals"]], color=color, linewidth=1.8, linestyle=dashes, marker="o", markersize=3.5, label=label)
        ax.axhline(20 / 24, color=MUTED, linewidth=0.9, linestyle="--")
        style(ax, "training steps", "eval pairs explained (all 24 primitives)" if ax is axes[0] else "", title)
        percent(ax)
        ax.set_xlim(0, 25000)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
        ax.title.set_fontsize(9.5)
    axes[0].text(24800, 20 / 24 + 0.02, "dashed: all 20 trained primitives explained, none of the 4 left out", color=MUTED, fontsize=7.5, ha="right")
    fig.suptitle("Shift family, four-cell moves left out of training: above the dashed line means the left-out amount is being solved", color=INK, fontsize=11, x=0.01, ha="left")
    save(fig, out, "shift_family.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "reports" / "2026-11-22-scaling")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    for draw in (curves_by_size, curves_by_model, steps_to_reach, final_by_size, by_depth, old_primitives, growing, heldout_curve, program_length, shift_family):
        draw(args.out)
    _ = math


if __name__ == "__main__":
    main()
