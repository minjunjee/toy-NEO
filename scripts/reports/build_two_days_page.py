#!/usr/bin/env python3
"""The two-day report (docs/reports/2026-11-24-two-days.md) as one page with charts, for publishing as an artifact.

Charts are inline SVG drawn from the result files (no image of a chart): their text and marks take the page's colour
tokens, so they read in light and dark, and every mark has a native tooltip with its value. The architecture figure is
embedded as an image. Writes docs/reports/2026-11-24-two-days-page.html (page content only; the artifact host adds the
document skeleton).

    python scripts/reports/build_two_days_page.py
"""

from __future__ import annotations

import base64
import glob
import json
import re
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "babytheorist"))
from orientation_table import rows as orientation_rows  # noqa: E402

SWAP = ["rot90", "rot270", "transpose", "anti_transpose"]


def load(path):
    return json.loads((REPO / path).read_text())


# ------------------------------------------------------------------ svg helpers
class Chart:
    """A plot area in a viewBox; x and y map data to pixels. Text uses classes styled by the page's tokens."""

    def __init__(self, w=560, h=300, left=52, right=118, top=20, bottom=46, x=(0, 1), y=(0, 1)):
        self.w, self.h, self.l, self.r, self.t, self.b = w, h, left, right, top, bottom
        self.xr, self.yr = x, y
        self.parts = []
        self.ends = []  # direct labels at line ends, spread apart when drawn

    def X(self, v):
        a, b = self.xr
        return self.l + (v - a) / (b - a) * (self.w - self.l - self.r)

    def Y(self, v):
        a, b = self.yr
        return self.h - self.b - (v - a) / (b - a) * (self.h - self.t - self.b)

    def yaxis(self, ticks, fmt=lambda v: f"{round(v * 100)}%", label=None):
        for t in ticks:
            y = self.Y(t)
            self.parts.append(f'<line class="grid" x1="{self.l}" x2="{self.w - self.r}" y1="{y:.1f}" y2="{y:.1f}"/>')
            self.parts.append(f'<text class="tick" x="{self.l - 8}" y="{y + 4:.1f}" text-anchor="end">{fmt(t)}</text>')
        if label:
            self.parts.append(f'<text class="axlabel" x="{self.l}" y="{self.t - 6}" text-anchor="start">{escape(label)}</text>')

    def xaxis(self, ticks, labels=None, label=None):
        y0 = self.h - self.b
        self.parts.append(f'<line class="axis" x1="{self.l}" x2="{self.w - self.r}" y1="{y0}" y2="{y0}"/>')
        for i, t in enumerate(ticks):
            x = self.X(t)
            text = labels[i] if labels else str(t)
            self.parts.append(f'<text class="tick" x="{x:.1f}" y="{y0 + 16}" text-anchor="middle">{escape(str(text))}</text>')
        if label:
            self.parts.append(f'<text class="axlabel" x="{(self.l + self.w - self.r) / 2:.1f}" y="{self.h - 6}" text-anchor="middle">{escape(label)}</text>')

    def line(self, pts, cls, name, dashed=False, end_label=True, label_dy=0):
        pts = [(x, y) for x, y in pts if y is not None]
        d = " ".join(f"{'M' if i == 0 else 'L'}{self.X(x):.1f},{self.Y(y):.1f}" for i, (x, y) in enumerate(pts))
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.parts.append(f'<path class="ln {cls}" d="{d}"{dash}/>')
        for x, y in pts:
            self.parts.append(f'<circle class="pt {cls}" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="3.2"><title>{escape(name)}, {x}: {y * 100:.0f}%</title></circle>')
        if end_label and pts:
            x, y = pts[-1]
            self.ends.append([self.Y(y) + 4.5, self.X(x) + 9, name, cls])

    def svg(self, title):
        ends = sorted(self.ends)
        for i in range(1, len(ends)):  # at least 15 px between direct labels, pushed downward
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 15)
        for y, x, name, cls in ends:
            self.parts.append(f'<text class="direct {cls}t" x="{x:.1f}" y="{y:.1f}">{escape(name)}</text>')
        return (f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{escape(title)}" class="chart">'
                + "".join(self.parts) + "</svg>")


def figure(svg, caption, number):
    return f'<figure class="fig"><div class="figbox">{svg}</div><figcaption><b>그림 {number}.</b> {caption}</figcaption></figure>'


def bars(groups, series, y=(0, 1), w=560, h=290, label=None, right=16, legend=True, value_labels=False):
    """groups: list of (group label, [value per series]); series: list of (name, class)."""

    c = Chart(w=w, h=h, right=right, x=(0, len(groups)), y=y, bottom=52 if any("\n" in g for g, _ in groups) else 40)
    c.yaxis([t / 100 for t in range(0, 101, 25)], label=label)
    gw = (c.w - c.l - c.r) / len(groups)
    bw = min(22.0, gw * 0.8 / len(series))
    for gi, (glabel, values) in enumerate(groups):
        x0 = c.l + gi * gw + (gw - bw * len(series)) / 2
        for si, (name, cls) in enumerate(series):
            v = values[si]
            if v is None:
                continue
            x = x0 + si * bw
            yv = c.Y(v)
            c.parts.append(f'<rect class="bar {cls}" x="{x + 1:.1f}" y="{yv:.1f}" width="{bw - 2:.1f}" height="{c.Y(0) - yv:.1f}" rx="2">'
                           f'<title>{escape(glabel.replace(chr(10), " "))}, {escape(name)}: {v * 100:.0f}%</title></rect>')
            if value_labels:
                c.parts.append(f'<text class="vlabel" x="{x + bw / 2:.1f}" y="{yv - 4:.1f}" text-anchor="middle">{v * 100:.0f}</text>')
        for li, part in enumerate(glabel.split("\n")):
            c.parts.append(f'<text class="tick" x="{c.l + gi * gw + gw / 2:.1f}" y="{c.Y(0) + 16 + li * 13:.1f}" text-anchor="middle">{escape(part)}</text>')
    c.parts.append(f'<line class="axis" x1="{c.l}" x2="{c.w - c.r}" y1="{c.Y(0)}" y2="{c.Y(0)}"/>')
    svg = c.svg(label or "bars")
    if legend:
        svg = legend_html(series) + svg
    return svg


def legend_html(series, dashed=()):
    items = "".join(f'<span class="key"><i class="sw {cls}{" dash" if name in dashed else ""}"></i>{escape(name)}</span>' for name, cls in series)
    return f'<div class="legend">{items}</div>'


# ------------------------------------------------------------------ data and charts
def chart_search():
    s = load("docs/reports/2026-11-23-search/search_summary.json")
    lengths = list(range(1, 11))
    c = Chart(x=(1, 10), y=(0, 1))
    c.yaxis([0, .25, .5, .75, 1.0], label="transfer, 여섯 seed 평균")
    c.xaxis(lengths, label="숨은 program의 길이")
    for w, cls, dy in ((1, "c5", 0), (4, "c2", 0), (16, "c1", 0)):
        c.line([(d, s["by_length"][str(d)][str(w)]["transfer"]) for d in lengths], cls, f"폭 {w}", label_dy=dy)
    return figure(c.svg("search by length"), "다시 학습하지 않고 탐색의 폭만 넓혔을 때 숨은 길이별 transfer. 8개 primitive, 한 단계 모델, 레벨 1, 5, 7의 평가 과제를 길이별로 모았다.", 1)


def chart_ceiling():
    ceil = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for f in glob.glob(str(REPO / "runs/queue/oracle_program_lv*.json")):
        for e in json.load(open(f)):
            for d, (t, q) in enumerate(zip(e["tasks_by_depth"], e["query_by_depth"])):
                if t and q is not None:
                    ceil[e["run"]][d + 1][0] += q * t
                    ceil[e["run"]][d + 1][1] += t
    s = load("docs/reports/2026-11-23-search/search_summary.json")["per_run"]
    order = ["onestep_t_w_12k_s45", "onestep_t_w_12k_s42", "onestep_t_w_12k_s47", "onestep_t_w_12k_s44", "onestep_t_w_12k_s43", "onestep_t_w_12k_s46"]
    groups = []
    for run in order:
        c8 = ceil[run][8]
        groups.append((f"seed {run[-2:]}", [c8[0] / c8[1], s[run]["16"]["8"]["transfer"], s[run]["1"]["8"]["transfer"]]))
    svg = bars(groups, [("정답 program을 그대로 실행", "c3"), ("탐색 폭 16", "c1"), ("greedy", "c5")], label="길이 8의 query 정답률")
    return figure(svg, "executor만 시험한 상한과 programmer가 고른 program. 정답 program을 primitive마다 가장 자주 쓰인 code로 적어 실행하면, code가 깨끗한 seed는 길이 8에서도 92%에서 97%를 맞힌다.", 2)


def chart_dream():
    before = load("docs/reports/2026-11-23-dream/dream_summary.json")["by_width"]
    mix = load("docs/reports/2026-11-24-dream-mix/dream_summary.json")["by_width"]
    lengths = list(range(1, 11))
    panels = []
    for w, title in (("1", "greedy"), ("16", "폭 16")):
        c = Chart(w=320, h=260, right=14, left=44, x=(1, 10), y=(0, 1))
        c.yaxis([0, .25, .5, .75, 1.0], label=title)
        c.xaxis([1, 4, 7, 10], label="숨은 길이")
        c.line([(d, before[w][str(d)]["before"]) for d in lengths], "c5", "꿈 학습 전", end_label=False)
        c.line([(d, before[w][str(d)]["after"]) for d in lengths], "c2", "꿈만", end_label=False, dashed=True)
        c.line([(d, mix[w][str(d)]["after"]) for d in lengths], "c1", "꿈과 실제 pair", end_label=False)
        panels.append(c.svg(title))
    legend = legend_html([("꿈 학습 전", "c5"), ("꿈만", "c2"), ("꿈과 검증된 실제 pair", "c1")], dashed=("꿈만",))
    return figure(legend + f'<div class="pair">{panels[0]}{panels[1]}</div>',
                  "꿈 학습 전후의 길이별 transfer, 여섯 seed 평균. 꿈만 하면 greedy의 짧은 program이 나빠지고, 검증된 실제 pair를 섞으면 그 손실이 사라지면서 긴 program도 더 오른다.", 3)


def chart_critic():
    def collect(pattern, rank):
        out = {}
        for f in glob.glob(str(REPO / pattern)):
            for r in json.load(open(f)):
                if r.get("rank", "policy") == rank:
                    out[(r["run"][-3:], r["split"], r["width"])] = r["transfer"]
        return out
    base = collect("runs/queue/dream_mix_onestep_search_lv*_gpu*.json", "policy")
    crit = collect("runs/queue/critic_search_lv*_s4*.json", "policy+critic1")
    groups = []
    for seed in ("s45", "s42"):
        for split in ("lv5", "lv7"):
            for w in (4, 16):
                k = (seed, split, w)
                if k in base and k in crit:
                    groups.append((f"seed {seed[1:]}\n레벨 {split[2:]}, 폭 {w}", [base[k], crit[k]]))
    svg = bars(groups, [("programmer의 선호로만", "c5"), ("남은 거리 예측기를 더함", "c4")], y=(0.5, 1.0), label="레벨 전체 transfer", value_labels=True)
    svg = svg.replace('<text class="tick"', '<text class="tick"', 1)
    return figure(svg, "남은 거리 예측기를 탐색 순위에 더했을 때. 폭이 좁을 때 효과가 크다. 세로축은 50%에서 시작한다.", 4)


def chart_vote():
    rows = {}
    for f in ("runs/queue/critic_search_lv7_s45.json", "runs/queue/critic_vote_lv7_s45.json"):
        for r in load(f):
            if r["width"] == 16 and r["run"].endswith("s45"):
                rows[r.get("rank", "policy")] = r["transfer_by_depth"]
    depths = list(range(2, 11))
    c = Chart(x=(2, 10), y=(0.4, 1.0), right=150)
    c.yaxis([.4, .6, .8, 1.0], label="transfer, 폭 16, 레벨 7, seed 45")
    c.xaxis(depths, label="숨은 program의 길이")
    lines = (("policy+critic1", "c5", "처음 통과한 program", False, 0),
             ("policy+critic1+vote", "c1", "모두 투표", False, 0),
             ("policy+critic1+vote_shortest", "c2", "가장 짧은 것만 투표", True, 0),
             ("policy+critic1+vote_weighted", "c4", "짧을수록 큰 표", False, 0))
    for key, cls, name, dashed, dy in lines:
        if key in rows:
            c.line([(d, rows[key][d - 1]) for d in depths], cls, name, dashed=dashed, label_dy=dy)
    return figure(c.svg("answer voting"), "support를 모두 통과한 program이 여럿일 때 query 답을 고르는 방법. 남은 거리 예측기를 쓴 같은 탐색에서 고르는 단계만 다르다. "
                  "모두 투표하면 길이 6 이상이 오르고 길이 2와 3이 조금 내려가며, 가장 짧은 program만 투표하면 짧은 과제의 손실이 사라진다. 세로축은 40%에서 시작한다.", 5)


def chart_m24():
    rows = {}
    for f in glob.glob(str(REPO / "runs/queue/m24_search_*.json")):
        for r in json.load(open(f)):
            if r["width"] == 16:
                rows[r["run"]] = r["transfer_by_depth"]
    groups = []
    for label, b, a in (("한 pair 랩, seed 42", "onestep_s_m24_wide_s42", "dream_mix_m24_wide_s42"),
                        ("두 pair hard EM\nseed 42", "onestep_s_m24_hardem_pairs2_s42", "dream_mix_m24_hardem_pairs2_s42"),
                        ("두 pair hard EM\nseed 45", "onestep_s_m24_hardem_pairs2_s45", "dream_mix_m24_hardem_pairs2_s45"),
                        ("두 pair hard EM\nseed 46", "onestep_s_m24_hardem_pairs2_s46", "dream_mix_m24_hardem_pairs2_s46")):
        if b in rows and a in rows:
            groups.append((label, [rows[b][3], rows[a][3]]))
    svg = bars(groups, [("꿈 학습 전", "c5"), ("꿈과 실제 pair", "c1")], label="24개 환경, 숨은 길이 4, 폭 16", value_labels=True)
    return figure(svg, "24개 환경에서 code를 먼저 깨끗하게 세운 모델에 꿈 학습을 얹었을 때. code가 섞인 한 pair 랩 모델에서는 거의 효과가 없다.", 6)


def chart_conditions():
    data = load("docs/reports/2026-11-23-em/orientation_by_condition.json")
    labels = {"lab": "랩, pair 하나", "soft EM": "soft EM, pair 하나", "hard EM": "hard EM, pair 하나", "lab, two pairs": "랩, pair 둘",
              "hard EM, two pairs": "hard EM, pair 둘"}
    panels = []
    for key, title in (("swap", "가로세로를 바꾸는 넷"), ("others", "나머지 17개")):
        n = len(data)
        c = Chart(w=320, h=270, right=10, left=44, bottom=64, x=(0, n), y=(0, 1))
        c.yaxis([0, .25, .5, .75, 1.0], label=title)
        gw = (c.w - c.l - c.r) / n
        if key == "swap":
            c.parts.append(f'<line class="gate" x1="{c.l}" x2="{c.w - c.r}" y1="{c.Y(0.6):.1f}" y2="{c.Y(0.6):.1f}"/>'
                           f'<text class="gatet" x="{c.l + 4}" y="{c.Y(0.6) - 6:.1f}" text-anchor="start">기준 60%</text>')
        for i, (cond, v) in enumerate(data.items()):
            cx = c.l + gw * (i + 0.5)
            mean, seeds = v[key]["mean"], v[key]["seeds"]
            cls = "c1" if "two" in cond else "c5"
            c.parts.append(f'<rect class="bar {cls} soft" x="{cx - gw * 0.3:.1f}" y="{c.Y(mean):.1f}" width="{gw * 0.6:.1f}" height="{c.Y(0) - c.Y(mean):.1f}" rx="2">'
                           f'<title>{escape(labels[cond])}: 평균 {mean * 100:.0f}%</title></rect>')
            for j, sv in enumerate(seeds):
                c.parts.append(f'<circle class="seed" cx="{cx + (j - (len(seeds) - 1) / 2) * 3.2:.1f}" cy="{c.Y(sv):.1f}" r="2.6"><title>{escape(labels[cond])}, seed {j + 1}: {sv * 100:.0f}%</title></circle>')
            c.parts.append(f'<text class="vlabel" x="{cx:.1f}" y="{c.Y(0) - 7:.1f}" text-anchor="middle">{mean * 100:.0f}</text>')
            for li, part in enumerate(labels[cond].split(", ")):
                c.parts.append(f'<text class="tick" x="{cx:.1f}" y="{c.Y(0) + 15 + li * 12:.1f}" text-anchor="middle">{escape(part)}</text>')
        c.parts.append(f'<line class="axis" x1="{c.l}" x2="{c.w - c.r}" y1="{c.Y(0)}" y2="{c.Y(0)}"/>')
        panels.append(c.svg(title))
    return figure(f'<div class="pair">{panels[0]}{panels[1]}</div>',
                  "24개 환경, 조건별 결과. 막대는 seed 평균, 점은 seed 하나씩이다. pair 둘로 된 과제의 hard EM만 기준선 근처까지 올라오고, 한 seed는 96%다.", 7)


def chart_validation():
    v = {r["run"]: r["both_pairs"] for r in load("runs/queue/pair_validation_m24.json")}
    pts = []
    for run, score in v.items():
        last = orientation_rows(REPO / "runs" / run)
        swap = [r for r in last if r["full"]][-1]["swap"]
        pts.append((run, score, swap))
    c = Chart(w=560, h=300, right=24, x=(0.24, 0.38), y=(0.25, 1.0))
    c.yaxis([.25, .5, .75, 1.0], label="가로세로를 바꾸는 넷")
    c.xaxis([0.25, 0.28, 0.31, 0.34, 0.37], labels=["0.25", "0.28", "0.31", "0.34", "0.37"], label="라벨 없는 검증 점수: 고른 code가 두 pair를 모두 재현한 비율")
    for run, score, swap in pts:
        cls = "c2" if "anneal" in run else "c1"
        name = ("온도를 내리는 형태" if "anneal" in run else "기본 형태") + f", seed {run[-2:]}"
        c.parts.append(f'<circle class="pt {cls}" cx="{c.X(score):.1f}" cy="{c.Y(swap):.1f}" r="4.5"><title>{escape(name)}: 점수 {score:.3f}, 넷 {swap * 100:.0f}%</title></circle>')
    best = max(pts, key=lambda p: p[1])
    c.parts.append(f'<text class="direct c1t" x="{c.X(best[1]) - 8:.1f}" y="{c.Y(best[2]) + 4:.1f}" text-anchor="end">점수 최고 run: 96%</text>')
    legend = legend_html([("hard EM 기본 형태", "c1"), ("온도를 내리는 형태", "c2")])
    return figure(legend + c.svg("validation"), "두 pair hard EM 열 run의 라벨 없는 검증 점수와 가로세로 넷. 순위 상관은 0.98이고, 점수가 가장 높은 run이 가장 잘 풀린 run이다.", 8)


def chart_scale():
    def last_swap(run):
        rows = orientation_rows(REPO / "runs" / run)
        return rows[-1]["swap"] if rows else None
    em = load("docs/reports/2026-11-23-em/orientation_by_condition.json")
    anneal24 = sum(last_swap(f"onestep_s_m24_hardem_pairs2_anneal_s{s}") for s in (42, 43, 44, 45, 46)) / 5
    series = [
        ("랩, pair 하나", "c5", [(24, em["lab"]["swap"]["mean"]), (32, last_swap("onestep_s_m32_wide_s42")), (52, last_swap("ladder_lab_p52_s42")), (64, last_swap("ladder_lab_p64_s42"))]),
        ("랩, pair 둘", "c3", [(24, em["lab, two pairs"]["swap"]["mean"]), (32, last_swap("onestep_s_m32_lab_pairs2_s42")), (52, last_swap("onestep_s_m52_lab_pairs2_s42")), (64, last_swap("onestep_s_m64_lab_pairs2_s42"))]),
        ("hard EM, pair 둘", "c1", [(24, em["hard EM, two pairs"]["swap"]["mean"]), (32, last_swap("onestep_s_m32_hardem_pairs2_s42")), (52, last_swap("onestep_s_m52_hardem_pairs2_s42")), (64, last_swap("onestep_s_m64_hardem_pairs2_s42"))]),
        ("온도를 내리는 hard EM", "c2", [(24, anneal24), (52, last_swap("onestep_s_m52_hardem_pairs2_anneal_s42")), (64, last_swap("onestep_s_m64_hardem_pairs2_anneal_s42"))]),
    ]
    c = Chart(x=(22, 66), y=(0, 1), right=158)
    c.yaxis([0, .25, .5, .75, 1.0], label="가로세로를 바꾸는 넷")
    c.xaxis([24, 32, 52, 64], label="환경의 primitive 수")
    dys = {"랩, pair 하나": 4, "랩, pair 둘": 0, "hard EM, pair 둘": 8, "온도를 내리는 hard EM": -6}
    for name, cls, pts in series:
        c.line(pts, cls, name, dashed="온도" in name, label_dy=dys[name])
    return figure(c.svg("scale"), "환경 크기에 따른 가로세로 넷. 24개는 seed 셋에서 다섯의 평균, 32개부터는 seed 42 하나다. 큰 환경에서는 두 hard EM 형태 가운데 하나가 넷을 완전히 푼다.", 9)


def architecture():
    data = base64.b64encode((REPO / "docs/figures/neo_architecture.png").read_bytes()).decode()
    return (f'<figure class="fig arch"><div class="figbox white"><img src="data:image/png;base64,{data}" alt="NEO 구조도"></div>'
            f'<figcaption><b>그림 10.</b> 이번 결과를 합친 NEO. (a) 깨어 있을 때 support로 검증하며 program을 찾고, (b) 같은 program의 pair 둘로 code를 배우고, '
            f'(c) 잠에서 얼린 executor로 꿈을 꾸어 programmer와 남은 거리 예측기를 고친다. 논문용 원본은 docs/figures/neo_architecture.pdf다.</figcaption></figure>')


# ------------------------------------------------------------------ page
CSS = """
:root{--bg:#ffffff;--surface:#f5f7fa;--fg:#1b1f26;--muted:#5b6472;--rule:#e3e7ee;--line:#c9d0da;
--c1:#2f6db5;--c2:#c46d1c;--c3:#3a8a55;--c4:#7556ad;--c5:#8b929c;--accent:#2f6db5;
--sans:"IBM Plex Sans KR","Noto Sans KR",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14171c;--surface:#1c2027;--fg:#e8ebf0;--muted:#9aa3b0;--rule:#2b313a;--line:#3a414c;
--c1:#6fa3e6;--c2:#e39a52;--c3:#6cc08a;--c4:#a68ae0;--c5:#8f97a3;--accent:#6fa3e6;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#14171c;--surface:#1c2027;--fg:#e8ebf0;--muted:#9aa3b0;--rule:#2b313a;--line:#3a414c;
--c1:#6fa3e6;--c2:#e39a52;--c3:#6cc08a;--c4:#a68ae0;--c5:#8f97a3;--accent:#6fa3e6;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:16px/1.78 var(--sans);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:760px;margin:0 auto;padding-block:40px 80px}
h1{font-size:30px;line-height:1.32;margin:0 0 10px;font-weight:700;text-wrap:balance;letter-spacing:-0.01em}
h2{font-size:22px;margin:56px 0 12px;padding-top:18px;border-top:1px solid var(--rule);text-wrap:balance}
h3{font-size:18px;margin:34px 0 8px;text-wrap:balance}
p,li{text-wrap:pretty}
.meta{color:var(--muted);font-size:14px;margin-bottom:28px}
code{font:13.5px/1.5 var(--mono);background:var(--surface);padding:1px 5px;border-radius:4px;overflow-wrap:anywhere}
.tablewrap{overflow-x:auto;margin:18px 0}
table{border-collapse:collapse;width:100%;font-size:14.5px;line-height:1.5;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
th{color:var(--muted);font-weight:600;border-bottom:1.5px solid var(--fg)}
.fig{margin:26px 0 30px}
.figbox{background:var(--surface);border-radius:10px;padding:14px 12px 8px}
.figbox.white{background:#ffffff;padding:10px}
.figbox img{display:block;width:100%;height:auto}
figcaption{color:var(--muted);font-size:14px;line-height:1.6;margin-top:10px}
figcaption b{color:var(--fg)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media (max-width:560px){.pair{grid-template-columns:1fr}}
.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .grid{stroke:var(--rule);stroke-width:1}
.chart .axis{stroke:var(--line);stroke-width:1}
.chart .tick{fill:var(--muted);font:13px var(--sans)}
.chart .axlabel{fill:var(--muted);font:600 13.5px var(--sans)}
.chart .vlabel{fill:var(--fg);font:600 12.5px var(--mono)}
.chart .direct{font:600 13.5px var(--sans)}
.chart .ln{fill:none;stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round}
.chart .pt{stroke:var(--surface);stroke-width:1.5}
.chart .gate{stroke:var(--c2);stroke-width:1.2;stroke-dasharray:5 4}
.chart .gatet{fill:var(--c2);font:600 12.5px var(--sans)}
.chart .seed{fill:var(--fg);stroke:var(--surface);stroke-width:1}
.chart .bar.soft{opacity:.38}
.c1{stroke:var(--c1);fill:var(--c1)}.c2{stroke:var(--c2);fill:var(--c2)}.c3{stroke:var(--c3);fill:var(--c3)}.c4{stroke:var(--c4);fill:var(--c4)}.c5{stroke:var(--c5);fill:var(--c5)}
.chart .ln.c1,.chart .ln.c2,.chart .ln.c3,.chart .ln.c4,.chart .ln.c5{fill:none}
.chart .bar{stroke:none}
.c1t{fill:var(--c1)}.c2t{fill:var(--c2)}.c3t{fill:var(--c3)}.c4t{fill:var(--c4)}.c5t{fill:var(--c5)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--muted);margin:0 4px 6px}
.key{display:inline-flex;align-items:center;gap:7px}
.sw{display:inline-block;width:18px;height:10px;border-radius:2px}
.sw.c1{background:var(--c1)}.sw.c2{background:var(--c2)}.sw.c3{background:var(--c3)}.sw.c4{background:var(--c4)}.sw.c5{background:var(--c5)}
.sw.dash{height:0;border-top:2.5px dashed var(--c2);background:none;border-radius:0}
.lead{font-size:17px;line-height:1.75}
"""


def main() -> None:
    text = (REPO / "docs/reports/2026-11-24-two-days.md").read_text()
    figs = {
        "### 3.2": chart_search(), "### 3.3": chart_ceiling(), "### 3.5": chart_dream(), "### 3.6": chart_critic(), "### 3.7": chart_vote(),
        "## 4. ": chart_m24(), "### 4.3": chart_conditions(), "### 4.5": chart_validation(), "## 5. ": chart_scale(),
    }
    for heading, html in figs.items():  # each chart goes right before the next section's heading
        i = text.index(heading)
        text = text[:i] + f"\n<!--FIG {heading}-->\n\n" + text[i:]
    md = markdown.Markdown(extensions=["tables", "fenced_code"])
    body = md.convert(text)
    for heading, html in figs.items():
        body = body.replace(f"<!--FIG {heading}-->", html)
    body = re.sub(r"<table>", '<div class="tablewrap"><table>', body).replace("</table>", "</table></div>")
    body = body.replace('<h2>6. 한계와 다음 일</h2>', architecture() + '<h2>6. 한계와 다음 일</h2>')
    title = re.search(r"<h1>(.*?)</h1>", body).group(1)
    body = re.sub(r"<p>(2026-11-24 14시 KST, Claude Code가 썼다\..*?)</p>", r'<p class="meta">\1</p>', body, count=1, flags=re.S)
    page = f"""<title>섞이는 code와 짧은 계획</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<main>
{body}
</main>
"""
    out = REPO / "docs/reports/2026-11-24-two-days-page.html"
    out.write_text(page)
    print(out, len(page) // 1024, "KB")


if __name__ == "__main__":
    main()
