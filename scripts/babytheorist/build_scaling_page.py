#!/usr/bin/env python3
"""Builds the page of scaling figures: docs/reports/2026-11-22-scaling.html, and a copy without the document wrapper for
publishing as an artifact (second argument). Prose is Korean, figure text is English. The figures are the PNG files that
scripts/babytheorist/scaling_figures.py writes into docs/reports/2026-11-22-scaling/; a figure that does not exist yet is
left out with a note. The reading notes are in NOTES below and are rewritten by hand as results come in.

    python scripts/babytheorist/build_scaling_page.py "<time>" [<artifact copy path>]
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIGURES = REPO / "docs" / "reports" / "2026-11-22-scaling"
OUT = REPO / "docs" / "reports" / "2026-11-22-scaling.html"

STYLE = """
:root { --bg: #f6f7f9; --surface: #ffffff; --ink: #151a21; --ink-2: #505a68; --rule: #dde2e8; --note: #eef3fa; --accent: #2a78d6;
  --sans: "IBM Plex Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", system-ui, sans-serif; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #111418; --surface: #191d23; --ink: #e9ecf1; --ink-2: #a9b2bf; --rule: #2b323c; --note: #1a222e; --accent: #7fb2f0; color-scheme: dark; } }
:root[data-theme="dark"] { --bg: #111418; --surface: #191d23; --ink: #e9ecf1; --ink-2: #a9b2bf; --rule: #2b323c; --note: #1a222e; --accent: #7fb2f0; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 16px; line-height: 1.75; margin: 0; }
main { max-width: 1040px; margin: 0 auto; padding-block: 40px 80px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1 { font-size: 28px; line-height: 1.3; margin: 0 0 6px; font-weight: 600; text-wrap: balance; }
h2 { font-size: 20px; line-height: 1.4; margin: 48px 0 10px; font-weight: 600; text-wrap: balance; }
p { margin: 0 0 14px; max-width: 46em; }
.when { color: var(--ink-2); font-size: 14px; margin-bottom: 22px; }
.lead { background: var(--note); border-left: 3px solid var(--accent); padding: 14px 18px; border-radius: 4px; }
.lead p:last-child { margin-bottom: 0; }
figure { margin: 18px 0 8px; background: #fcfcfb; border: 1px solid var(--rule); border-radius: 8px; padding: 10px; overflow-x: auto; }
figure img { display: block; width: 100%; min-width: 640px; height: auto; }
.read { color: var(--ink-2); font-size: 14.5px; }
.missing { color: var(--ink-2); border: 1px dashed var(--rule); border-radius: 8px; padding: 14px 18px; font-size: 14.5px; }
ul { margin: 0 0 14px; padding-left: 22px; max-width: 46em; }
"""

# (heading, figure file, how to read it, what it shows); the last item is a list of paragraphs
NOTES = [
    ("모델을 나란히 놓은 학습 곡선", "curves_by_model.png",
     "열 하나가 환경 크기 하나이고 선 하나가 모델 하나다. 위는 학습 batch에서 pair를 정확히 재현한 비율, 아래는 다른 입력으로 옮겼을 때의 transfer다. 가로축은 학습 step이다.",
     ["32개와 64개 환경에서 code 하나를 횟수만큼 실행하는 모델의 곡선이 랩 NEO보다 늦게 오른다. 64개 환경의 7,000 step에서 학습 batch 정답률이 랩 NEO 75%, 이 모델 60%다. 작은 이동 묶음에서는 이 모델이 더 빨랐으므로, 섞인 묶음에서는 횟수가 필요 없는 primitive가 대부분이라 이 모델의 구조가 짐이 되는 것으로 본다. 추측이다.",
      "랩 NEO의 64개 환경 transfer가 12,000 step에 76%로, 16개와 24개 환경의 58%와 62%보다 높다. 52개와 64개 묶음에 새로 들어간 것이 대부분 2칸에서 6칸 이동과 대각선 이동처럼 executor가 쉽게 배우는 것이기 때문이다. primitive 수만으로는 환경의 어려움을 잴 수 없다는 뜻이고, 이 축은 다시 짜야 한다."]),
    ("Lie algebra codebook은 본 적 없는 양을 말한다", "heldout_curve.png",
     "네 방향 1칸에서 6칸 이동 가운데 4칸 이동을 학습에서 뺀 작은 환경이다. 저장된 checkpoint마다 왼쪽은 4칸 이동, 오른쪽은 학습한 20개의 transfer를 쟀다. program은 모델이 스스로 적은 것 그대로다.",
     ["초록 실선이 사용자가 제안한 모델이다. programmer가 pair를 보고 codebook에서 종류와 양을 함께 고르고, executor는 그 종류의 단위 code를 양만큼 실행한다. codebook의 항목은 c(g,t) = exp(t·A_g)·b_g이고 회전 각을 고정해 둔 것이다. 19,200 step에서 4칸 이동의 transfer가 93%다. 양 4의 항목은 학습 중 한 번도 골라진 적이 없는데, 양 3과 양 5 사이에 놓여 있어서 programmer가 처음 보는 4칸 pair에서 그 자리를 가리켰다.",
      "같은 모델에서 항목을 서로 관계없는 벡터로 둔 대조가 초록 점선이다. 학습한 20개는 78%까지 가지만 4칸 이동은 끝까지 0%다. 둘 사이의 차이는 항목이 회전으로 묶여 있는지 하나뿐이므로, 4칸을 푼 것은 Lie algebra 구조 덕분이다.",
      "회전 각을 학습하게 둔 형태는 회전이 줄어 항목이 모이면서 무너졌다. 점선 짧은 것이 그것이다.",
      "주황 선은 code 하나를 횟수만큼 실행하는 모델이다. 4,800 step부터 100%로 더 빠르고 안정적이다. 다만 이 모델은 횟수를 말하지 않는다. support pair에서 1번부터 6번까지 실행해 보고 정답과 같아지는 횟수를 고른다. 시험 때 답과 견주는 탐색을 하는 셈이다. Lie algebra 모델은 pair를 한 번 보고 횟수를 바로 말한다.",
      "seed 42 하나의 결과이고 seed 43과 44를 돌리고 있다. 위쪽 방향은 이 seed에서 종류 둘로 갈라져서 4칸이 그 틈에 빠졌다."]),
    ("환경이 커지면 이미 있던 primitive가 망가진다", "old_primitives.png",
     "선 하나가 primitive 묶음 하나다. 가장 연한 선은 처음 8개, 그다음은 16개 환경에서 새로 들어온 8개, 이런 식이다. 가로축은 그 묶음이 들어 있는 환경의 크기이고 세로축은 그 묶음의 primitive 하나짜리 과제의 transfer다. 수평이면 환경이 커져도 그 묶음이 영향을 받지 않은 것이다.",
     ["랩 NEO에서 처음 8개의 transfer는 8개 환경에서 96%, 16개 환경에서 60%다. 같은 primitive, 같은 학습량인데 옆에 primitive 8개가 더 있다는 것만으로 3분의 1을 잃었다. 그 뒤 24개, 32개로 커질 때는 53%에서 61% 사이로 더 떨어지지 않는다.",
      "16개 환경에서 새로 들어온 것은 180도와 270도 회전, 전치, 한 칸 이동, 색 되돌리기다. 처음 8개의 회전과 뒤집기와 비슷한 일을 하는 것들이고, 앞 보고서에서 회전과 뒤집기가 16개부터 무너진다는 것을 쟀다. 비슷한 primitive가 들어오면 기존 것과 섞여 버리는 것으로 본다. 확장성의 첫 실패 지점이 여기다.",
      "32개 환경에서 새로 들어온 bbox 채우기, 2배 확대, 거울 같은 것들은 98%다. 서로 겹치지 않는 primitive는 더해도 문제가 없다."]),
    ("점점 커지는 환경", "growing.png",
     "모델 하나가 계속 학습하는 동안 환경이 8, 16, 24, 32, 52, 64개로 커진다. 점선에서 환경이 커진다. 각 단계는 6,000 step이고 그 단계의 전체 묶음으로 학습한다. 선 하나가 primitive 묶음 하나다.",
     ["첫 단계인 8개 환경은 6,000 step 동안 랩 NEO가 54%까지만 갔다. codebook이 처음부터 code 128개라 남는 code가 많고, 앞에서 쟀듯 남는 code가 많으면 primitive가 여러 code로 나뉘어 transfer가 낮다. 16개로 커진 뒤에도 처음 8개는 떨어지지 않고 69%까지 계속 오르고, 새로 들어온 8개는 1,000 step 만에 59%다. 크기마다 따로 학습했을 때 처음 8개가 96%에서 60%로 떨어진 것과 다르다. 아직 두 단계째라 더 지켜본다."]),
    ("새로 들어온 primitive를 배우는 속도", "growing_speed.png",
     "왼쪽 둘은 환경이 커진 뒤 새로 들어온 묶음의 transfer가 50%와 80%에 닿기까지 걸린 step이다. 가로축은 로그 눈금이다. 선이 수평이거나 완만하게 오르면 환경이 커져도 새 것을 쉽게 배우는 것이다. 오른쪽은 처음 8개의 transfer를 각 단계 끝에서 잰 것이다.",
     ["돌고 있다."]),
    ("랩 NEO, 환경 크기마다 한 선", "curves_by_size_lab_neo_one_code.png",
     "선 하나가 환경 크기 하나다. 색이 진할수록 큰 환경이다.",
     ["primitive가 8개든 32개든 학습 batch 곡선이 5,000 step 근처까지 거의 겹친다. 갈리는 것은 그 뒤에 닿는 높이다. 8개는 99%, 16개부터는 80%에서 92%에서 멈춘다. 앞 보고서에서 끝까지 배우지 못하는 것이 회전과 뒤집기 계열이라는 것을 쟀다."]),
    ("일정 수준에 닿는 데 걸린 학습량", "steps_to_reach.png",
     "가로축은 환경의 primitive 수이고 로그 눈금이다. 세로축은 학습 batch 정답률이 처음 그 수준에 닿은 step이다. 선이 곧으면 로그처럼 느는 것이고 수평이면 크기와 무관한 것이다. x 표시는 24,000 step 안에 닿지 못했다는 뜻이다.",
     ["랩 NEO가 50%에 닿는 step은 네 크기에서 4,200에서 4,600 사이로 수평이다. 80%는 10,000에서 15,000 사이이고, 95%는 8개 환경만 닿았다."]),
    ("숨은 program이 길어질 때", "by_depth.png",
     "판 하나가 환경 크기 하나다. 가로축은 숨은 program의 primitive 수이고 세로축은 transfer다. 모든 모델은 primitive 하나짜리 pair로만 학습했으므로 2개 이상은 전부 본 적 없는 과제다.",
     ["랩 NEO는 8개 환경에서 깊이 1, 2, 3, 4의 transfer가 97%, 84%, 62%, 35%다. 16개 이상에서는 깊이 2가 33% 근처, 깊이 3이 11%에서 16%, 깊이 4가 4%에서 7%다. 깊이가 하나 늘 때마다 대략 절반이 된다."]),
    ("과제 종류별로 본 끝 성능", "final_by_size.png",
     "판 하나가 과제 종류 하나다. 가로축은 환경의 primitive 수다. 선이 오른쪽으로 갈수록 덜 떨어지는 모델이 확장성이 좋은 모델이다.",
     ["랩 NEO는 네 판 모두 8개에서 16개로 갈 때 떨어지고 그 뒤로는 평평하다. 다른 모델의 점은 run이 끝나면 들어온다."]),
    ("code 하나를 횟수만큼 실행하는 모델, 환경 크기마다 한 선", "curves_by_size_one_code_x_count.png",
     "위의 랩 NEO 그림과 같은 방식이다.", ["32개와 64개 환경만 돌고 있다. 작은 환경의 run은 큐에 있다."]),
    ("같은 모델에서 실행 사이 gradient를 끊은 것", "curves_by_size_one_code_x_count_no_gradient_between_applications.png",
     "횟수 t의 loss가 t번째 실행만 학습시키고 앞의 실행에는 gradient가 가지 않는다. 랩 NEO의 reencode_hard와 같은 처리다.",
     ["작은 이동 묶음에서 이 변형은 4,800 step에 100%에 닿아 끝까지 유지했다. gradient를 흘리는 원래 형태보다 빨랐고 흔들리지 않았다. 큰 환경에서도 그런지를 보려고 넣었다."]),
    ("작은 이동 묶음에서 본 것", "shift_family.png",
     "네 방향 1칸에서 6칸 이동 24개 가운데 네 칸 이동을 학습에서 뺀 환경이다. 세로축은 24개 전체의 설명률이다. 점선은 학습한 20개를 다 맞히고 뺀 4개를 못 맞힐 때의 높이다. 점선 위로 올라간 선만 학습하지 않은 양을 풀고 있다.",
     ["왼쪽은 codebook만 바꾸고 executor를 한 번 실행하는 모델들이다. Lie algebra로 묶은 codebook을 포함해 어느 것도 점선을 넘지 못한다.",
      "가운데는 code 하나를 횟수만큼 실행하는 모델이다. code 6개와 12개가 끝에 100%이고, 실행 사이 gradient를 끊은 것은 4,800 step부터 내내 100%다. 위로 4칸은 위로 1칸 code를 네 번으로 적혔다.",
      "오른쪽은 programmer가 종류와 횟수를 함께 말하는 모델, 곧 Lie algebra codebook의 양을 실행 횟수로 쓴 모델이다. 세 형태 모두 아직 점선에 못 미치고 출렁인다. 회전을 학습하게 둔 형태는 회전이 줄어 한 종류의 항목 여섯 개가 20도 안에 모여서 멈췄다. 회전 각을 고정한 형태와 항목을 관계없는 벡터로 둔 대조가 돌고 있다."]),
]


def image(name: str) -> str | None:
    path = FIGURES / name
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def body(now: str) -> str:
    parts = ['<main>', '<h1>환경이 커질 때 모델이 어떻게 버티는가</h1>', f'<div class="when">{now} 기준. Claude Code가 그렸다. run이 끝나는 대로 그림이 채워진다.</div>',
             '<div class="lead"><p>가로축은 학습 step이거나 환경의 primitive 수다. 연산량은 쓰지 않는다. 환경은 primitive 8, 16, 24, 32, 52, 64개 묶음이고 앞의 것이 뒤의 것에 들어 있다. '
             '모든 모델은 primitive 하나짜리 pair로만 학습한다.</p><p>선의 이름은 그림 안에 영어로 적었다. lab NEO, one code는 랩 NEO에서 program을 code 한 개로 묶은 것이다. '
             'one code x count는 code 하나를 횟수만큼 연달아 실행하는 모델이다. rotation-tied (kind, count)는 Lie algebra codebook의 양이 곧 실행 횟수인 모델이다.</p></div>']
    for heading, name, how, shows in NOTES:
        parts.append(f"<h2>{heading}</h2>")
        data = image(name)
        if data is None:
            parts.append('<div class="missing">이 그림은 아직 없다. 필요한 run이 끝나면 그려진다.</div>')
            parts.append(f'<p class="read">읽는 법. {how}</p>')
            continue
        parts.append(f'<figure><img src="{data}" alt="{heading}"></figure>')
        parts.append(f'<p class="read">읽는 법. {how}</p>')
        parts.extend(f"<p>{text}</p>" for text in shows)
    parts.append("</main>")
    return "\n".join(parts)


def main() -> None:
    now = sys.argv[1]
    fonts = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap">'
    title = "<title>toy-NEO 확장성 그래프</title>"
    content = body(now)
    OUT.write_text(f'<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n{title}\n{fonts}\n'
                   f"<style>{STYLE}</style>\n</head>\n<body>\n{content}\n</body>\n</html>\n")
    print(OUT, OUT.stat().st_size)
    if len(sys.argv) > 2:
        copy = Path(sys.argv[2])
        copy.write_text(f'{title}\n<meta name="claude:viewport-fit-sides" content="cover">\n{fonts}\n<style>{STYLE}</style>\n{content}\n')
        print(copy, copy.stat().st_size)


if __name__ == "__main__":
    main()
