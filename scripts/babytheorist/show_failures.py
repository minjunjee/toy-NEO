#!/usr/bin/env python3
"""Draw tasks a trained NEO gets wrong, to be looked at by a person.

For each failed task one row: the first support pair, the query input, the query answer, the model's prediction, and the
cells where the prediction differs. The title of a row gives the hidden program (read from the file for this picture
only), the number of codes the model extracted from the support pair, and the codes.

    python scripts/babytheorist/show_failures.py --run runs/<run> --split lv1 --count 40 --out docs/reports/<name>/failures.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from training.evaluate import load_run  # noqa: E402

# the colour convention of ARC-style grids: 0 is the background
PALETTE = ListedColormap(["#111111", "#1e93ff", "#f93c31", "#4fcc30", "#ffdc00", "#999999", "#e53aa3", "#ff851b", "#87d8f1", "#921231"])
COLUMNS = ("support input", "support output", "query input", "query answer", "prediction", "wrong cells")


def short(program: str) -> str:
    return " > ".join(step["primitive"] for step in json.loads(program))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--max-depth", type=int, help="only tasks whose hidden program is at most this deep")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data, config, step = load_run(args.run, args.ckpt, device)
    model.eval()
    infer = next((dict(b.get("infer", {})) for b in config["evals"] if b["type"] == "babytheorist_observe" and b["split"] == args.split), {})
    rows, seen, wrong = [], 0, 0
    for batch in data.eval_loader(args.split, 250):
        moved = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model.infer(moved["support_x"], moved["support_y"], moved["query_x"], None, **infer)
        failed = ~out["pred"].eq(moved["query_y"]).flatten(1).all(dim=1)
        seen += len(failed)
        wrong += int(failed.sum())
        for row in torch.nonzero(failed).flatten().tolist():
            if len(rows) >= args.count or (args.max_depth and int(batch["depth"][row]) > args.max_depth):
                continue
            codes = [int(i.view(len(failed), -1)[row, 0]) for i in out["indices"]][: int(out["n_extracted"][row])] if out.get("indices") and out["indices"][0] is not None else []
            rows.append({"grids": [batch[k][row] for k in ("support_x", "support_y", "query_x", "query_y")] + [out["pred"][row].cpu()],
                         "title": f"task {int(batch['index'][row])}   hidden program: {short(batch['transitions'][row])}   extracted {int(out['n_extracted'][row])} codes {codes}"})
        if len(rows) >= args.count:
            break
    figure, axes = plt.subplots(len(rows), len(COLUMNS), figsize=(1.9 * len(COLUMNS), 2.05 * len(rows)), squeeze=False)
    for r, item in enumerate(rows):
        grids = item["grids"] + [(item["grids"][4] != item["grids"][3]).long() * 2]  # wrong cells in red on black
        for c, grid in enumerate(grids):
            ax = axes[r, c]
            ax.imshow(grid.numpy(), cmap=PALETTE, vmin=0, vmax=9, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(COLUMNS[c], fontsize=8, pad=3)
        axes[r, 0].text(0.0, 1.22 if r == 0 else 1.05, item["title"], transform=axes[r, 0].transAxes, ha="left", va="bottom", fontsize=7)
    figure.text(0.01, 1.0, f"{args.run.name}, step {step}: {wrong} of the first {seen} tasks wrong; {len(rows)} shown (the query answer is never shown to the model)", ha="left", va="bottom", fontsize=8)
    figure.subplots_adjust(hspace=0.32, wspace=0.08, top=0.985, bottom=0.005, left=0.01, right=0.99)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"{len(rows)} failed tasks -> {args.out}")


if __name__ == "__main__":
    main()
