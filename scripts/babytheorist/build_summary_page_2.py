#!/usr/bin/env python3
"""Builds docs/reports/2026-11-22-summary.html: the Korean summary of the night of 2026-11-22 with inline SVG figures.

Figure text is English, prose is Korean. The numbers are written in by hand from docs/reports/2026-11-22-fast-search.md
and the files it names; the page head, the styles and the task figure come from build_summary_page.py.

    python scripts/babytheorist/build_summary_page_2.py "<time>" "<paragraph on what is still running>"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_summary_page import PAGE as FIRST_PAGE  # noqa: E402
from build_summary_page import task_figure  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "reports" / "2026-11-22-summary.html"
HEAD = FIRST_PAGE[: FIRST_PAGE.index("<main>")]


def depth_figure() -> str:
    """Transfer against the length of the hidden program, level 3 eval tasks, grid-only handover."""

    depths = [1, 2, 3, 4, 5, 6, 7, 8]
    series = [("trained on one step only", "soft", False, [100, 86, 68, 57, 47, 30, 18, 19]),
              ("lab config, trained on several steps", "hard", False, [50, 38, 25, 18, 11, 12, 9, 2])]
    x0, x1, y0, y1 = 56, 640, 250, 30
    fx = lambda d: x0 + (x1 - x0) * (d - 1) / 7   # noqa: E731
    vy = lambda v: y0 + (y1 - y0) * v / 100       # noqa: E731
    s = ['<svg viewBox="0 0 740 300" role="img" aria-label="Transfer against the length of the hidden program for a model trained on one step and a model trained on several">']
    for v in (0, 25, 50, 75, 100):
        s.append(f'<line x1="{x0}" y1="{vy(v)}" x2="{x1}" y2="{vy(v)}" class="gridline"/><text x="{x0 - 8}" y="{vy(v) + 4}" class="t2 end">{v}%</text>')
    for d in depths:
        s.append(f'<text x="{fx(d)}" y="{y0 + 18}" class="t2 mid">{d}</text>')
    s.append(f'<text x="{(x0 + x1) / 2}" y="{y0 + 38}" class="t2 mid">number of primitives in the hidden program</text>')
    s.append(f'<text x="{x0}" y="16" class="t2">share of tasks transferred to a new input</text>')
    for name, rule, dashed, values in series:
        points = [(fx(d), vy(v)) for d, v in zip(depths, values)]
        s.append(f'<path d="M{" L".join(f"{x:.1f} {y:.1f}" for x, y in points)}" class="line {rule}{" dash" if dashed else ""}"/>')
        for x, y in points:
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="dot {rule}"/>')
    s.append(f'<text x="{fx(4) + 24}" y="{vy(82)}" class="t softt b">trained on one step only</text>')
    s.append(f'<text x="{fx(4) + 24}" y="{vy(82) + 16}" class="t2">56% over all level-3 tasks</text>')
    s.append(f'<text x="{fx(1) + 12}" y="{vy(10)}" class="t hardt b">lab config, several steps</text>')
    s.append(f'<text x="{fx(1) + 12}" y="{vy(10) + 16}" class="t2">17% over all level-3 tasks</text>')
    s.append('</svg>')
    return "".join(s)


def bars(rows: list[tuple[str, str, float]], title: str, label: str) -> str:
    """Horizontal bars, one series. rows: (name, note, value in percent)."""

    left, width, top, rh = 250, 380, 44, 34
    height = top + rh * len(rows) + 30
    s = [f'<svg viewBox="0 0 740 {height}" role="img" aria-label="{label}">', f'<text x="{left}" y="22" class="t b">{title}</text>']
    for tick in (0, 50, 100):
        x = left + width * tick / 100
        s.append(f'<line x1="{x}" y1="{top - 6}" x2="{x}" y2="{top + rh * len(rows) - 8}" class="{"axis" if tick == 0 else "gridline"}"/>')
        s.append(f'<text x="{x}" y="{top + rh * len(rows) + 8}" class="t2 mid">{tick}%</text>')
    for i, (name, note, value) in enumerate(rows):
        y = top + i * rh
        s.append(f'<text x="{left - 12}" y="{y + (10 if note else 15)}" class="t end b">{name}</text>')
        if note:
            s.append(f'<text x="{left - 12}" y="{y + 24}" class="t2 end">{note}</text>')
        w = width * value / 100
        s.append(f'<path d="M{left} {y + 2} H{left + w - 3} a3 3 0 0 1 3 3 V{y + 17} a3 3 0 0 1 -3 3 H{left} Z" class="softf"/>')
        s.append(f'<text x="{left + w + 6}" y="{y + 16}" class="t mono">{value:.0f}%</text>')
    s.append('</svg>')
    return "".join(s)


def headline_figure() -> str:
    return bars([("lab config", "several steps, 36,000 steps", 40),
                 ("searched parts", "several steps, grid handed over", 38),
                 ("searched parts", "several steps, probabilities handed over", 12),
                 ("trained on one step only", "searched parts, saw no composition", 83)],
                "transferred to a new input, all of level 1", "Transfer on level 1 for models trained on several steps and a model trained on one step")


def codes_figure() -> str:
    return bars([("8 codes", "too few are found", 40), ("10 codes", "", 97), ("12 codes", "", 93), ("16 codes", "", 90), ("32 codes", "", 84)],
                "one step, 8 primitives: transferred to a new input", "Transfer on single-primitive tasks against the size of the codebook")


BODY = r"""<main>
<header>
<div class="eyebrow">toy-NEO · 2026-11-22 새벽 작업 · @@NOW@@ 기준</div>
<h1>NEO가 못하는 이유는 부품이 아니라 여러 단계를 한꺼번에 배우는 방식이었다</h1>
<p class="lead">어제 정리한 내용에서 많이 바뀌었다. 부품을 직접 찾아 학습을 빠르게 만들었고, 그 과정에서 원인이 한 곳으로 좁혀졌다. 어제의 정리는 <code>docs/reports/2026-11-21-summary.html</code>에 그대로 있다.</p>
</header>

<ol class="first">
<li>primitive 하나짜리 학습 쌍으로 program 한 단계만 학습한 모델이, 한 번도 본 적 없는 여러 단계 과제를 transfer 83%로 푼다.</li>
<li>여러 단계로 학습한 모델은 랩 설정이든, 부품을 고치든, 단계 사이의 통로를 막든 40%에서 멈춘다.</li>
<li>그러니 NEO의 부품은 primitive를 배우고 이어 쓸 수 있다. 여러 단계를 한꺼번에 학습하는 동안 code가 primitive가 아닌 다른 것이 된다.</li>
</ol>

<figure>
<div class="scroll">@@FIG_HEADLINE@@</div>
<figcaption>레벨 1의 평가 과제 10,000개. 예시 쌍 하나에서 뽑은 program을 새 입력에 실행하고, 정답은 보여 주지 않는다. 위 둘과 맨 아래는 단계 사이로 격자만 넘겨서 쟀다.</figcaption>
</figure>

<h2><span class="n">1</span>무엇을 재고 있나</h2>
<p>과제는 BabyTheorist의 레벨 1이다. 15칸 격자 위의 object 하나를 숨은 program이 바꾼다. 숨은 program은 primitive 여덟 가운데 몇 개를 이은 것이다. 색 바꾸기, 위아래 뒤집기, 좌우 뒤집기, 회전, 네 방향 이동이다. NEO의 programmer는 예시 쌍을 보고 codebook에서 code를 하나씩 골라 program을 쓰고, executor가 그 code들을 차례로 실행한다.</p>
<figure>
<div class="scroll">@@FIG_TASK@@</div>
<figcaption>과제 하나에서 재는 것 둘. 첫째가 재현, 둘째가 transfer다. 격자와 code 번호는 설명을 위한 예시다.</figcaption>
</figure>
<p>재현은 예시 쌍에서 쓴 program이 그 예시의 출력을 다시 만드는 과제의 비율이다. transfer는 그 program을 새 입력에 실행했을 때 정답과 격자 전체가 같은 과제의 비율이다. 재현은 높은데 transfer가 낮으면 모델이 그 예시에만 맞는 설명을 쓴 것이다. 숫자는 모두 직접 돌려서 쟀다.</p>

<h2><span class="n">2</span>먼저 속도를 올렸다</h2>
<p>학습 하나에 6시간이 걸려서는 아무것도 찾을 수 없었다. 학습 한 스텝을 프로파일해 보니, GPU는 계속 일하는데 행렬곱은 시간의 15%에서 20%뿐이었다. 작은 모델이 token 451개짜리 열을 다섯 단계씩 풀어 돌리면서 작은 kernel을 스텝마다 만 번 가까이 부르고 있었다.</p>
<div class="tablewrap"><table>
<tr><th>바꾼 것</th><th class="num">학습 한 스텝</th><th class="num">초당 쌍</th></tr>
<tr><td>랩 설정 그대로</td><td class="num">152ms</td><td class="num">840</td></tr>
<tr><td>programmer와 executor를 컴파일</td><td class="num">88ms</td><td class="num">1,456</td></tr>
<tr><td>컴파일에 더해, programmer가 칸마다 token 하나만 받게 함</td><td class="num">44ms</td><td class="num">2,898</td></tr>
</table></div>
<p>컴파일은 결과를 바꾸지 않는다. 첫 스텝의 loss가 소수 넷째 자리까지 같다. 랩의 programmer는 지금 state와 목표를 칸마다 token 둘로 늘어놓는데, 새 programmer는 칸마다 둘을 이어 붙여 token 하나로 만든다. token이 451개에서 226개가 된다.</p>
<p>그리고 싼 판을 만들었다. primitive 하나짜리 학습 쌍만 쓰고 program을 한 단계로 제한한 학습이다. 계산이 다섯 단계짜리의 5분의 1이고 하나에 5분쯤이다. 이 판으로 지금까지 조건 80개쯤을 돌렸다.</p>

<h2><span class="n">3</span>싼 판에서 찾은 것</h2>
<p>싼 판은 한 가지만 묻는다. 부품이 primitive 여덟을 code 하나씩으로 배울 수 있는가. 랩 설정 그대로는 못 한다. 재현이 seed에 따라 53%와 17%다. 이유는 code를 못 찾아서였다. 랩의 programmer를 쓰면 executor를 키우든 학습률을 바꾸든 처음 1,800스텝 동안 code를 하나만 쓴다.</p>
<p>세 가지를 바꾸면 풀린다. programmer를 위의 새것으로 바꾸고 폭을 256으로 넓힌다. 평가에 쓰는 가중치 평균을 끈다. batch를 512에서 128로 줄이고 스텝을 네 배로 늘린다. 계산량은 같다. 이러면 재현이 99%가 된다.</p>
<p>transfer는 codebook의 크기가 정한다.</p>
<figure>
<div class="scroll">@@FIG_CODES@@</div>
<figcaption>primitive 여덟, program 한 단계. primitive 하나짜리 평가 과제 2,774개. 재현은 code 10개 이상에서 모두 99%다.</figcaption>
</figure>
<p>code가 남으면 모델은 primitive 하나를 code 여러 개로 쪼갠다. 회전을 보면 가로로 긴 object를 돌리는 code와 세로로 긴 object를 돌리는 code가 따로 생긴다. 예시의 object와 새 입력의 object가 다른 쪽이면 틀린다. 남는 code가 적으면 쪼갤 여유가 없어서 transfer가 오른다. 다만 딱 8개만 주면 code를 다 찾지 못한다.</p>
<p>어제 넣었던 잠과 같은 종류의 동작도 여기서 다시 쟀다. 안 쓰이는 code를 다시 뿌려 주면 재현은 빨리 오르지만 code 16개가 전부 쓰이면서 뒤집기가 다섯에서 일곱 개로 쪼개지고, transfer는 89%에서 70%로 내려간다.</p>

<h2><span class="n">4</span>여러 단계로 학습하면 무엇을 해도 40%다</h2>
<p>싼 판에서 고른 부품을 다섯 단계짜리 본 모델에 넣었다. 두 가지 일이 생겼다.</p>
<p>단계 사이로 칸마다의 색 확률을 넘기는 랩의 방식으로 학습하면, 재현은 6,000스텝에 76%로 빨리 오르는데 transfer는 12%다. 이 모델은 primitive 하나짜리 과제를 전부 code 두 개짜리 program으로 설명한다. code 하나가 연산이 아니라, code 두 개가 합쳐져 변환 하나를 가리키는 주소다. 주소의 앞자리는 격자에 보이지 않는 확률에 실려 다음 단계로 넘어간다. 단계 사이로 격자만 넘기게 하고 재면 재현이 76%에서 2%가 된다.</p>
<p>그래서 처음부터 격자만 넘기며 학습했다. 이번에는 code가 주소가 될 수 없다. 재현 84%, transfer 38%가 나온다. 랩 설정이 36,000스텝에 닿은 자리에 3분의 1의 스텝으로 닿지만, 같은 자리다. 이 모델의 program을 보면 이동조차 깔끔하지 않다. 아래로 이동 과제의 20%가 "거의 맞는 code 하나에 고치는 code 하나"로 설명되고, 뒤집기 둘은 같은 code 넷을 나눠 쓴다. 마지막 격자만 맞으면 loss가 0이라, code가 얼추 하는 단계와 땜질하는 단계로 나뉘어도 학습은 막지 않는다.</p>

<h2><span class="n">5</span>한 단계로만 배운 모델은 이어 쓸 줄 안다</h2>
<p>싼 판의 모델을 추가 학습 없이 그대로 가져왔다. program을 다섯 단계까지 쓰게 하고, 단계 사이로 격자만 넘기며 레벨 1 전체를 풀게 했다. 이 모델은 primitive 둘을 이은 과제를 한 번도 본 적이 없다.</p>
<div class="tablewrap"><table>
<tr><th>모델</th><th class="num">재현</th><th class="num">transfer</th><th>숨은 program 길이 1, 2, 3, 4의 transfer</th></tr>
<tr><td>랩 설정, 여러 단계로 36,000스텝</td><td class="num">83%</td><td class="num">40%</td><td>57%, 39%, 28%, 21%</td></tr>
<tr><td>고친 부품, 여러 단계로 10,800스텝</td><td class="num">84%</td><td class="num">38%</td><td>57%, 35%, 27%, 22%</td></tr>
<tr class="hl"><td>한 단계로만 학습, code 16개</td><td class="num">87%</td><td class="num">83%</td><td>97%, 87%, 69%, 62%</td></tr>
<tr><td>한 단계로만 학습, code 10개</td><td class="num">81%</td><td class="num">77%</td><td>97%, 81%, 56%, 43%</td></tr>
</table></div>
<p>programmer가 목표를 보고 첫 primitive의 code를 고른다. executor가 깨끗한 격자를 낸다. 그 격자를 본 programmer가 다음 code를 고른다. 따로 가르치지 않았는데 이렇게 이어 간다. 더 긴 program에서도 같다.</p>
<figure>
<div class="scroll">@@FIG_DEPTH@@</div>
<figcaption>레벨 3의 평가 과제 10,000개. 숨은 program의 길이는 4가 가장 많다. program을 10단계까지 허용하고 격자만 넘겼다. 두 모델 모두 레벨 1의 데이터로만 학습했다.</figcaption>
</figure>
<div class="fix">
<h3>이 결과를 읽을 때 조심할 것</h3>
<p>한 단계 학습은 숨은 program의 길이를 읽어서 primitive 하나짜리 학습 쌍만 골랐다. 벤치마크는 그 정보를 주지 않으므로 이 83%는 벤치마크 점수가 아니다. 원인을 가르는 진단이다. 길이를 읽지 않고 전체 학습 쌍에 한 단계만 허용해서 학습해도 같은 것이 나오는지는 지금 돌리고 있다.</p>
</div>
<p>이 결과가 어제까지의 그림을 바꾼다. NEO가 40%에서 멈추는 이유를 codebook의 크기, 잠, 단계 사이의 통로에서 찾았는데, 그것들은 곁가지였다. 부품은 할 수 있다. 예시 하나를 여러 code로 한꺼번에 설명하도록 학습시키면, 설명만 되면 되는 목표 아래에서 code가 primitive가 아닌 것으로 굳는다.</p>

<h2><span class="n">6</span>primitive를 늘리면 어디가 무너지나</h2>
<p>BabyTheorist가 가진 primitive는 object 하나짜리로 11개가 전부라서, 같은 object 모양을 쓰고 primitive를 32개까지 늘린 데이터를 직접 만들었다. 한 칸 이동, 격자 가장자리로 붙이기, 180도와 270도 회전, 전치, 색을 거꾸로 돌리기, 두 배로 키우기, 거울처럼 이어 붙이기 같은 것이다. 싼 판으로 돌렸고, 아래는 primitive 하나짜리 과제의 값이다.</p>
<div class="tablewrap"><table>
<tr><th>primitive 수</th><th class="num">재현</th><th class="num">transfer</th><th>code 하나로 90% 넘게 옮겨 가는 primitive</th></tr>
<tr><td>8</td><td class="num">99%</td><td class="num">96%</td><td>8개 가운데 7개</td></tr>
<tr><td>11, BabyTheorist의 것 전부</td><td class="num">100%</td><td class="num">98%</td><td>10개 가운데 9개</td></tr>
<tr><td>16</td><td class="num">81~84%</td><td class="num">59~60%</td><td>16개 가운데 6~8개</td></tr>
<tr><td>24</td><td class="num">82%</td><td class="num">55%</td><td>24개 가운데 9개</td></tr>
<tr><td>32</td><td class="num">88%</td><td class="num">66%</td><td>32개 가운데 16개</td></tr>
</table></div>
<p>11개까지는 거의 다 된다. 16개부터 무너지고, 무너지는 종류가 정해져 있다.</p>
<p>회전과 뒤집기 계열은 16개 이상에서 transfer가 8%에서 24%다. 재현도 30%에서 78%로 낮고 primitive 하나에 code가 네 개에서 일곱 개 붙는다.</p>
<p>이동 계열은 더 흥미롭게 무너진다. primitive 8개일 때는 "자기 높이만큼 위로"가 code 하나였고 100% 옮겨 갔다. 한 칸 이동을 더한 16개에서도 그대로 100%다. 격자의 맨 위까지 붙이는 이동을 더한 24개에서 28%로 떨어진다. 이 모델의 code를 직접 보니 위로 가는 code가 primitive가 아니라 칸 수로 나뉘어 있다. 위로 2칸인 과제는 "자기 높이만큼"이든 "맨 위까지"든 같은 code 하나가 설명하고, 3칸은 다른 code, 4칸은 또 다른 code다. 두 primitive를 함께 설명하는 더 단순한 기준이 "위로 k칸"이기 때문으로 보인다. 그런데 k는 object의 크기와 위치에 따라 달라지므로 새 입력에서는 틀린다. 예시 하나만 보면 "위로 3칸"과 "자기 높이인 3칸만큼 위로"는 구별되지 않는다. 모델이 어느 쪽으로 일반화하는지는 데이터에 어떤 primitive가 섞여 있느냐에 따라 달라졌다.</p>
<p>모양을 바꾸는 primitive는 잘 배운다. 속 비우기, 테두리 두르기, 두 배로 키우기, 거울 이어 붙이기는 88%에서 100%다. 16개 이상의 줄은 code를 primitive 수의 두 배로 준 조건이고, 24개와 32개는 안 쓰는 code를 다시 뿌리는 조건만 끝났다. code를 빠듯하게 준 조건은 아직 돌고 있다.</p>

<h2><span class="n">7</span>지금 도는 것과 다음</h2>
<p>@@RUNNING@@</p>
<p>다음에 볼 것은 셋이다. 길이를 읽지 않는 한 단계 학습이 통하는가. 통하면 그것이 벤치마크의 규칙 안에서 쓸 수 있는 학습 방법의 출발점이다. 여러 단계 학습에서 code가 정확히 언제 망가지는가. 한 단계로 배운 모델에서 시작해 여러 단계 학습을 이어 가면 깨끗한 code가 유지되는지 보면 된다. 그리고 회전과 뒤집기가 왜 code 하나를 얻지 못하는가.</p>
<p>디스크는 <code>/home</code>이 있는 SSD가 58GB, <code>/data</code>가 435GB 남아 있다. 여러 개를 까는 실험은 체크포인트를 하나만 남기고 있다.</p>

<h2><span class="n">8</span>이 문서에서 쓴 말</h2>
<dl>
<dt>재현</dt><dd>예시 쌍에서 쓴 program을 그 예시의 입력에 실행했을 때 예시의 출력이 그대로 나오는 과제의 비율.</dd>
<dt>transfer</dt><dd>그 program을 새 입력에 실행했을 때 정답과 격자 전체가 같은 과제의 비율. 정답은 보여 주지 않는다.</dd>
<dt>여러 단계로 학습</dt><dd>랩 NEO의 학습 방식. 학습 쌍 하나를 code 다섯 개까지 이은 program으로 설명하게 하고, 마지막 격자가 맞는지로 loss를 준다.</dd>
<dt>한 단계로만 학습</dt><dd>program을 code 하나로 제한하고 primitive 하나짜리 학습 쌍만 쓴 학습. 싼 판이라고 부른 것이다.</dd>
<dt>격자만 넘기기</dt><dd>program의 한 단계가 끝나면 칸마다 확률이 가장 높은 색만 남긴 격자를 다음 단계의 출발점으로 삼는 것. 랩 설정은 확률을 그대로 넘긴다.</dd>
</dl>
<p class="lead" style="font-size:14px">숫자의 원본과 자세한 과정은 레포의 <code>docs/reports/2026-11-22-fast-search.md</code>에 있다. 브랜치는 <code>code/baseline-observation</code>이다.</p>
</main>
</body>
</html>
"""


def main() -> None:
    now, running = sys.argv[1:3]
    head = HEAD.replace("<title>toy-NEO 실험 정리</title>", "<title>toy-NEO 실험 정리</title>")
    page = head + (BODY.replace("@@FIG_HEADLINE@@", headline_figure()).replace("@@FIG_TASK@@", task_figure()).replace("@@FIG_CODES@@", codes_figure())
                   .replace("@@FIG_DEPTH@@", depth_figure()).replace("@@NOW@@", now).replace("@@RUNNING@@", running))
    assert "@@" not in page
    OUT.write_text(page)
    print(OUT, len(page))


if __name__ == "__main__":
    main()
