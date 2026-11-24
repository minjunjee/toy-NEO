#!/usr/bin/env python3
"""Architecture figure of the long-term NEO for the paper: wake (verified program search), learning codes from tasks
(two-pair hard EM), and sleep (dreaming and a label-free gate). Writes docs/figures/neo_architecture.{pdf,svg,png}.

Full text width of a two-column paper (178 mm). Fonts: Nimbus Sans converted to TrueType (family NimbusSansTT) and
embedded as Type 42, so the PDF holds no Type 3 font. After drawing, every label is checked to lie inside its box and
no two labels overlap; problems are printed.

    python scripts/figures/neo_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

FONTS = Path("/data/minjun_dev/toy-NEO/fonts")
for style in ("Regular", "Bold", "Italic"):
    font_manager.fontManager.addfont(str(FONTS / f"NimbusSansTT-{style}.ttf"))
plt.rcParams.update({
    "font.family": "NimbusSansTT", "font.size": 6.0, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "mathtext.fontset": "custom", "mathtext.rm": "NimbusSansTT", "mathtext.it": "NimbusSansTT:italic", "mathtext.bf": "NimbusSansTT:bold",
})

W, H = 178.0, 82.0  # mm
INK, MUTED, RULE = "#262626", "#6b6b6b", "#d4d4d4"
STYLE = {  # fill, edge
    "perception": ("#f0f0f0", "#7d7d7d"),
    "programmer": ("#dde8f7", "#2f5d9b"),
    "executor": ("#fbe7d4", "#b0621c"),
    "codebook": ("#e2f0e4", "#3c7d4d"),
    "critic": ("#ece4f5", "#6a4e97"),
    "gate": ("#ffffff", "#3a3a3a"),
}
OBJECT = "#c4473f"

fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")
CONTAINERS: list[tuple[tuple[float, float, float, float], list]] = []  # box extent and the labels that must stay inside
LABELS: list = []


def text(x, y, s, size=5.8, color=INK, ha="center", va="center", weight="normal", rotation=0):
    artist = ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, fontweight=weight, rotation=rotation, zorder=6, linespacing=1.18)
    LABELS.append(artist)
    return artist


def box(x, y, w, h, kind, title=None, sub=None, dashed=False, title_size=6.4, sub_size=5.6, radius=1.0, gap=None):
    fill, edge = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", fc=fill, ec=edge, lw=0.75,
                                ls=(0, (2.0, 1.3)) if dashed else "-", zorder=2))
    inside = []
    if title and sub:
        gap = gap if gap is not None else h * 0.17
        inside.append(text(x + w / 2, y + h / 2 + gap, title, size=title_size, weight="bold"))
        inside.append(text(x + w / 2, y + h / 2 - gap, sub, size=sub_size))
    elif title:
        inside.append(text(x + w / 2, y + h / 2, title, size=title_size, weight="bold"))
    CONTAINERS.append(((x, y, w, h), inside))
    return inside


def arrow(p, q, color=INK, dashed=False, lw=0.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=0.32,head_width=0.16", mutation_scale=10, color=color, lw=lw,
                                 ls=(0, (2.2, 1.4)) if dashed else "-", connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
                                 joinstyle="miter", capstyle="butt", zorder=5))


def grid(x, y, cells, size=5.0, n=5, color=OBJECT, unknown=False):
    step = size / n
    ax.add_patch(Rectangle((x, y), size, size, fc="white", ec="#9a9a9a", lw=0.45, zorder=3))
    for i in range(1, n):
        ax.plot([x + i * step] * 2, [y, y + size], color="#e4e4e4", lw=0.28, zorder=3)
        ax.plot([x, x + size], [y + i * step] * 2, color="#e4e4e4", lw=0.28, zorder=3)
    for r, c in cells:
        ax.add_patch(Rectangle((x + c * step, y + size - (r + 1) * step), step, step, fc=color, ec="none", zorder=4))
    if unknown:
        ax.text(x + size / 2, y + size / 2, "?", ha="center", va="center", fontsize=7, color=MUTED, fontweight="bold", zorder=6)


def token(x, y, label, size=3.2):
    fill, edge = STYLE["codebook"]
    ax.add_patch(FancyBboxPatch((x, y), size, size, boxstyle="round,pad=0,rounding_size=0.45", fc=fill, ec=edge, lw=0.6, zorder=4))
    CONTAINERS.append(((x, y, size, size), [text(x + size / 2, y + size / 2, label, size=5.2)]))


def rule(x, y0, y1):
    ax.plot([x, x], [y0, y1], color=RULE, lw=0.6, zorder=0)


L_SHAPE = [(1, 1), (2, 1), (2, 2)]
L_TURN = [(1, 2), (1, 1), (2, 1)]
BAR = [(1, 3), (2, 3), (3, 3), (3, 2)]
BAR_TURN = [(2, 1), (2, 2), (2, 3), (1, 1)]
Q_SHAPE = [(1, 1), (1, 2), (2, 2), (3, 2)]
Q_TURN = [(2, 1), (2, 2), (1, 2), (1, 3)]

# ------------------------------------------------------------------ (a) wake
text(2, 79.5, "(a) Wake: verified program search", size=7.2, weight="bold", ha="left")
text(9.75, 73.0, "support pairs", color=MUTED)
grid(2, 63.5, L_SHAPE)
arrow((7.4, 66.0), (11.1, 66.0), color=MUTED, lw=0.55)
grid(12.5, 63.5, L_TURN)
grid(2, 54.5, BAR)
arrow((7.4, 57.0), (11.1, 57.0), color=MUTED, lw=0.55)
grid(12.5, 54.5, BAR_TURN)
text(9.75, 48.0, "query", color=MUTED)
grid(2, 39.5, Q_SHAPE)
arrow((7.4, 42.0), (11.1, 42.0), color=MUTED, lw=0.55)
grid(12.5, 39.5, [], unknown=True)

box(20.5, 35, 5.6, 38, "perception")
CONTAINERS[-1][1].append(text(23.3, 54.0, "state encoder E", size=6.2, weight="bold", rotation=90))
for yy in (66.0, 57.0, 42.0):
    arrow((17.7, yy), (20.5, yy), color=MUTED, lw=0.55)

ax.add_patch(FancyBboxPatch((28.5, 25.5), 47.5, 50.5, boxstyle="round,pad=0,rounding_size=1.4", fc="none", ec="#a2a2a2", lw=0.6,
                            ls=(0, (3, 2)), zorder=1))
text(52.25, 73.2, r"program search: one code per step, up to $L$ steps", color=MUTED)
box(30.5, 56.5, 20, 11, "programmer", r"programmer $\pi$", r"$(s_k,\, s_y) \mapsto a_k$")
box(54.0, 56.5, 20, 11, "perception", "state decoder D", r"$s_{k+1} \mapsto x_{k+1}$")
box(30.5, 39.5, 20, 13.5, "codebook")
CONTAINERS[-1][1].append(text(40.5, 50.3, "codebook C", size=6.4, weight="bold"))
for r in range(3):
    for c in range(5):
        ax.add_patch(Rectangle((33.0 + c * 3.0, 40.9 + r * 2.2), 2.4, 1.7, fc="#3c7d4d" if (r, c) == (1, 2) else "#cfe6d3", ec="#3c7d4d",
                               lw=0.35, zorder=4))
box(54.0, 39.5, 20, 13.5, "executor", r"executor $f$", r"$(s_k,\, c) \mapsto s_{k+1}$")
arrow((40.5, 56.5), (40.5, 53.0))
text(41.6, 54.75, "top-B", size=5.2, color=MUTED, ha="left")
arrow((50.5, 46.25), (54.0, 46.25))
text(52.25, 47.7, r"$c$", size=5.8)
arrow((64.0, 53.0), (64.0, 56.5))
arrow((54.0, 62.0), (50.5, 62.0))
text(52.25, 69.3, r"next step: $x_{k+1} \rightarrow$ E $\rightarrow s_{k+1}$", size=5.2, color=MUTED)
box(30.5, 27.5, 20, 9, "critic", r"critic $h$", "steps left")
box(54.0, 27.5, 20, 9, "gate", "keep the best B", r"cost $-\log \pi + \lambda h$", title_size=6.0, sub_size=5.4)
arrow((50.5, 32.0), (54.0, 32.0), color=STYLE["critic"][1])

box(78.5, 55.5, 12.5, 12.5, "gate", "verify on", r"all supports", dashed=True, title_size=6.0, sub_size=6.0, gap=1.7)
arrow((76.0, 61.75), (78.5, 61.75))
text(84.75, 51.6, "program z", color=MUTED)
token(79.2, 46.0, r"$c_3$")
token(83.15, 46.0, r"$c_7$")
token(87.1, 46.0, r"$c_3$")
arrow((84.75, 55.5), (84.75, 53.3), lw=0.6)
arrow((84.75, 45.8), (84.75, 38.6), lw=0.6)
text(85.5, 42.2, "on the\nquery", size=5.2, color=MUTED, ha="left")
grid(82.25, 32.6, Q_TURN)
text(84.75, 29.6, r"$\hat{y}_q$", size=6.2)

# legend
ax.plot([2.5, 7.0], [17.5, 17.5], color=INK, lw=0.75)
text(8.0, 17.5, "data flow", ha="left")
ax.plot([22.0, 26.5], [17.5, 17.5], color=INK, lw=0.75, ls=(0, (2.2, 1.4)))
text(27.5, 17.5, "training signal", ha="left")
for i, (kind, label) in enumerate((("perception", "encoder, decoder"), ("programmer", "programmer"), ("codebook", "codebook"),
                                   ("executor", "executor"), ("critic", "critic"))):
    fill, edge = STYLE[kind]
    xx = (2.5, 25.0, 42.0, 57.5, 72.5)[i]
    ax.add_patch(FancyBboxPatch((xx, 11.0), 3.0, 2.0, boxstyle="round,pad=0,rounding_size=0.35", fc=fill, ec=edge, lw=0.55, zorder=3))
    text(xx + 4.0, 12.0, label, ha="left")

# ------------------------------------------------------------------ (b) learning codes
rule(93.5, 5, 80.5)
text(96, 79.5, "(b) Learning codes", size=7.2, weight="bold", ha="left")
text(112.5, 73.0, "a task: two pairs, one hidden program", color=MUTED)
grid(97, 63.5, L_SHAPE)
arrow((102.4, 66.0), (106.1, 66.0), color=MUTED, lw=0.55)
grid(107.5, 63.5, L_TURN)
grid(97, 54.5, BAR)
arrow((102.4, 57.0), (106.1, 57.0), color=MUTED, lw=0.55)
grid(107.5, 54.5, BAR_TURN)
text(122.5, 61.5, "different\nobjects", size=5.4, color=MUTED)
box(96.0, 41.5, 33, 8, "executor", "E step: every code, both pairs", title_size=5.6)
arrow((112.5, 53.8), (112.5, 49.5))
heights = [7.0, 5.4, 8.2, 1.3, 6.2, 7.6, 4.9, 8.0]
for i, hgt in enumerate(heights):
    ax.add_patch(Rectangle((98.6 + i * 3.65, 26.0), 2.5, hgt, fc="#3c7d4d" if i == 3 else "#d5d5d5", ec="none", zorder=3))
ax.plot([97.6, 127.6], [26.0, 26.0], color="#8a8a8a", lw=0.45, zorder=3)
text(112.5, 37.4, "loss summed over both pairs, per code", size=5.4, color=MUTED)
arrow((112.5, 41.5), (112.5, 39.0), lw=0.55, color=MUTED)
text(99.85 + 3 * 3.65, 23.6, r"$c^{*}$", size=6.0, color=STYLE["codebook"][1], weight="bold")
box(96.5, 8.5, 15, 10.5, "executor", "M step", r"train $f$ on $c^{*}$")
box(113.5, 8.5, 15, 10.5, "programmer", r"$\pi$ imitates", r"the choice $c^{*}$")
arrow((109.6, 21.6), (104.0, 19.0), color=STYLE["executor"][1], dashed=True)
arrow((111.4, 21.6), (121.0, 19.0), color=STYLE["programmer"][1], dashed=True)
text(112.5, 4.6, "unused codes get a slowly growing bonus", size=5.2, color=MUTED)

# ------------------------------------------------------------------ (c) sleep
rule(132.5, 5, 80.5)
text(135, 79.5, "(c) Sleep: dreams and a gate", size=7.2, weight="bold", ha="left")
text(155.5, 73.0, r"a dreamed program ($f$, C frozen)", color=MUTED)
xs = [135.5, 146.2, 156.9, 167.6]
shapes = [L_SHAPE, L_TURN, [(1, 3), (1, 2), (2, 2)], [(2, 3), (2, 2), (3, 2)]]
for i, (gx, cells) in enumerate(zip(xs, shapes)):
    grid(gx, 63.5, cells)
    if i < 3:
        arrow((gx + 5.3, 66.0), (gx + 10.4, 66.0), color=STYLE["executor"][1], lw=0.55)
        token(gx + 6.35, 67.2, ["a", "b", "c"][i], size=2.9)
text(xs[0] + 2.5, 61.3, "real input", size=5.2, color=MUTED)
text(xs[3] + 2.5, 61.3, "target", size=5.2, color=MUTED)
box(135, 42.5, 19.5, 12.5, "programmer", r"programmer $\pi$", "next code toward\na far target", sub_size=5.3, gap=2.3)
box(157.5, 42.5, 19.5, 12.5, "critic", r"critic $h$", "steps left", sub_size=5.6, gap=2.0)
arrow((147.0, 59.8), (145.0, 55.0), color=STYLE["programmer"][1], dashed=True)
arrow((163.0, 59.8), (166.5, 55.0), color=STYLE["critic"][1], dashed=True)
box(135, 28.5, 19.5, 9.5, "codebook", "real pairs", "one code reproduces", title_size=5.9, sub_size=5.2)
arrow((144.75, 38.0), (144.75, 42.5), color=STYLE["programmer"][1], dashed=True)
box(140.0, 7.5, 32.5, 14.5, "gate", "label-free gate", "keep an update only if held-out\ntasks of two pairs are\nstill reproduced", dashed=True,
    title_size=6.0, sub_size=5.2, gap=3.6)
arrow((146.0, 28.5), (151.5, 22.0), color=MUTED, lw=0.55)
arrow((167.25, 42.5), (162.0, 22.0), color=MUTED, lw=0.55)
text(156.25, 3.6, "the wake phase uses the kept snapshot", size=5.2, color=MUTED)

# ------------------------------------------------------------------ checks: labels inside boxes, no two labels overlap
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inverse = ax.transData.inverted()


def extent(artist):
    bb = artist.get_window_extent(renderer)
    (x0, y0), (x1, y1) = inverse.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return x0, y0, x1, y1


problems = []
for (x, y, w, h), inside in CONTAINERS:
    for artist in inside:
        x0, y0, x1, y1 = extent(artist)
        margin = 0.15 if w < 4 else 0.5
        if x0 < x + margin or x1 > x + w - margin or y0 < y + margin * 0.8 or y1 > y + h - margin * 0.8:
            problems.append(f"outside its box: {artist.get_text()!r}")
for i, a in enumerate(LABELS):
    ax0, ay0, ax1, ay1 = extent(a)
    if ax0 < 0.5 or ax1 > W - 0.5 or ay0 < 0.5 or ay1 > H - 0.5:
        problems.append(f"off the figure: {a.get_text()!r}")
    for b in LABELS[i + 1:]:
        bx0, by0, bx1, by1 = extent(b)
        if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
            problems.append(f"overlap: {a.get_text()!r} and {b.get_text()!r}")
print("\n".join(problems) if problems else "no label problems")

out = Path(__file__).resolve().parents[2] / "docs" / "figures"
out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "svg"):
    fig.savefig(out / f"neo_architecture.{ext}")
with plt.rc_context({"svg.fonttype": "path"}):  # text drawn as outlines: looks the same where the font is not installed
    fig.savefig(out / "neo_architecture_outlined.svg")
fig.savefig(out / "neo_architecture.png", dpi=300)
print(out / "neo_architecture.pdf")
