#!/usr/bin/env python3
"""Collect every number the autoresearch report deck draws into one json file.

The deck generator lives in the website repository (tools/slides-autoresearch-report) and must not read this repository
at build time, so the numbers are copied there as a snapshot. Rerun this and copy the file when runs finish.

    python scripts/babytheorist/export_deck_data.py [--out docs/reports/2026-11-22-scaling/deck_data.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
SCALING = REPO / "docs" / "reports" / "2026-11-22-scaling"
ORIENT = ("rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "anti_transpose")
SIZES = (8, 16, 24, 32, 52, 64)
HELD_OUT = ("shift_up_4", "shift_down_4", "shift_left_4", "shift_right_4")


def rows(run: str) -> list[dict]:
    path = RUNS / run / "train.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def light(run: str, key: str = "transfer_strict_depth1") -> list[list[float]]:
    return [[r["step"], round(r[f"observe/lv1/{key}"], 4)] for r in rows(run) if f"observe/lv1/{key}" in r]


def train_curve(run: str, every: int = 250, window: int = 10) -> list[list[float]]:
    points = [(r["step"], r["grid_acc"]) for r in rows(run) if "grid_acc" in r and "elapsed" in r]
    out = []
    for i, (step, _) in enumerate(points):
        if step % every == 0:
            chunk = [v for _, v in points[max(0, i - window + 1): i + 1]]
            out.append([step, round(sum(chunk) / len(chunk), 4)])
    return out


def per_primitive(run: str) -> dict | None:
    full = [r for r in rows(run) if "observe_full/lv1/one_step_tasks" in r]
    if not full:
        return None
    return {n: round(v["transfer_strict"] / v["tasks"], 4) for n, v in full[-1]["observe_full/lv1/one_step_tasks"].items() if v["tasks"]}


def blocks() -> list[tuple[str, list[str]]]:
    sys.path.insert(0, str(REPO / "scripts" / "babytheorist"))
    import generate_many_primitives as generator

    out, before = [], ()
    for n in SIZES:
        names = generator.SETS[str(n)]
        out.append((str(n), [x for x in names if x not in before and x not in HELD_OUT]))
        before = names
    return out


def growing(model: str) -> list[dict]:
    points = []
    for index, n in enumerate(SIZES):
        for file in sorted((RUNS / f"grow_{model}_p{n}_s42" / "eval").glob("observe_lv1_step*.json")):
            table = json.loads(file.read_text()).get("one_step_tasks")
            if table:
                points.append({"size": n, "t": index * 6000 + int(file.stem.split("_step")[1]),
                               "p": {k: round(v["transfer_strict"] / v["tasks"], 4) for k, v in table.items() if v["tasks"]}})
    return points


def long_tables(run: str) -> list[dict]:
    out = []
    for file in sorted((RUNS / run / "eval").glob("observe_lv1_step*.json")):
        table = json.loads(file.read_text()).get("one_step_tasks")
        if table:
            out.append({"step": int(file.stem.split("_step")[1]), "p": {k: round(v["transfer_strict"] / v["tasks"], 4) for k, v in table.items() if v["tasks"]}})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=SCALING / "deck_data.json")
    args = parser.parse_args()
    data = {}
    # the multi-step problem and the one-step learner (earlier report, 2026-11-22-fast-search.md)
    data["warm_start"] = {"jump5": light("fastneo_6k_warm_all_single_a_lv1_s42", "transfer_strict"),
                          "jump5_low_lr": light("fastneo_3k_warm_all_g0_lr3e5_single_a_lv1_s42", "transfer_strict")}
    data["length"] = {name: light(run, "transfer_strict") for name, run in {
        "len2_s42": "fastneo_3k_len2_lr3e5_g0_single_a_lv1_s42", "len4_s42": "fastneo_3k_len4_lr3e5_g0_single_a_lv1_s42",
        "len2_s43": "fastneo_3k_len2_lr3e5_g0_single_a_lv1_s43", "len4_s43": "fastneo_3k_len4_lr3e5_g0_single_a_lv1_s43",
        "len2_s44": "fastneo_3k_len2_lr3e5_g0_single_a_lv1_s44", "len4_s44": "fastneo_3k_len4_lr3e5_g0_single_a_lv1_s44",
        "len2_alldata": "fastneo_3k_len2_alldata_single_a_lv1_s42", "len5_depth2": "fastneo_3k_len5_depth2_single_a_lv1_s42"}.items()}
    # amounts: the shift family with four-cell moves left out
    data["heldout_curve"] = json.loads((SCALING / "heldout_curve.json").read_text()) if (SCALING / "heldout_curve.json").exists() else {}
    data["amount_curves"] = {run: light(run, "explain_depth1") for run in (
        "onestep_k_flat_s42", "onestep_k_rot_s42", "onestep_k_rot_add_s42", "onestep_k_free_add_s42", "repeat_c12_s42", "repeat_c12_cut_s42",
        "counted_circle_s42", "counted_circle_s43", "counted_circle_s44", "counted_free_s42", "counted_free_s43", "counted_free_s44")}
    # the ladder of environment sizes
    lab = {8: "onestep_s_m8_wide_s42", 16: "onestep_s_m16_wide_s42", 24: "onestep_s_m24_wide_s42", 32: "onestep_s_m32_wide_s42", 52: "ladder_lab_p52_s42", 64: "ladder_lab_p64_s42"}
    data["ladder"] = {"train": {str(n): train_curve(r) for n, r in lab.items()}, "transfer": {str(n): light(r) for n, r in lab.items()},
                      "per_primitive": {str(n): per_primitive(r) for n, r in lab.items()}}
    scale = {}
    for file in SCALING.glob("scale_*.json"):
        for entry in json.loads(file.read_text()):
            scale[entry["run"]] = {g: {"own": round(v["own_transfer"], 4), "stepwise": round(v["stepwise_transfer"], 4), "tasks": v["tasks"]} for g, v in entry["groups"].items()}
    data["depth"] = {str(n): scale.get(r) for n, r in lab.items() if r in scale}
    data["blocks"] = blocks()
    # the growing environment
    data["growing"] = {m: growing(m) for m in ("lab", "repeat", "repeatcut", "lora", "counted", "labsame")}
    # the orientation family
    data["orientation_runs"] = {run: per_primitive(run) for run in (
        "onestep_s_m24_wide_s42", "onestep_s_m24_labprog_s42", "onestep_s_m24_deepprog_s42", "onestep_m24_more_shared_s42", "onestep_m24_more_split_s42",
        "onestep_lora_m24_s42", "onestep_split2_m24_s42", "onestep_m24_more_compete4_s42", "onestep_m24_more_compete8_s42", "onestep_m24_split_children_s42",
        "onestep_m24_split_random_s42", "onestep_m24_sleep4_s42", "onestep_s_m24_oracle_s42", "onestep_s_m24gen_wide_s42", "onestep_s_m8_wide_s42",
        "onestep_s_m32_wide_48k_s42", "onestep_s_m32_wide_x_s42", "onestep_m24_em_s42", "onestep_m24_em_hot_s42", "onestep_m24_alternate_s42")}
    data["long"] = {run: long_tables(run) for run in ("onestep_s_m24_long200k_s42", "onestep_s_m24_long200k_x_s42")}
    args.out.write_text(json.dumps(data))
    print(args.out, args.out.stat().st_size)


if __name__ == "__main__":
    main()
