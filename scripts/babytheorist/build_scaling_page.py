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
    ("랩 NEO는 환경이 커져도 초반 속도는 같고, 닿는 높이가 달라진다", "curves_by_size_lab_NEO.png",
     "선 하나가 환경 크기 하나다. 왼쪽은 학습 batch에서 pair를 정확히 재현한 비율이고 오른쪽은 다른 입력으로 옮겼을 때의 transfer다. 가로축은 학습 step이다.",
     ["primitive가 8개든 32개든 왼쪽 곡선은 5,000 step 근처까지 거의 겹친다. 절반을 배우는 데 걸리는 학습량은 환경 크기와 무관했다.",
      "갈리는 곳은 그 뒤다. 8개에서는 99%까지 가는데 16개부터는 80%에서 92% 사이에서 멈춘다. 오른쪽의 transfer는 8개가 96%, 나머지가 58%에서 74%다.",
      "그러니 랩 NEO가 큰 환경에서 잃는 것은 속도가 아니다. 끝까지 배우지 못하는 primitive가 생긴다. 앞 보고서에서 그것이 회전과 뒤집기 계열이라는 것을 쟀다. seed 하나의 결과다."]),
    ("일정 수준에 닿는 데 걸린 학습량", "steps_to_reach.png",
     "가로축은 환경의 primitive 수이고 로그 눈금이다. 세로축은 학습 batch 정답률이 처음 그 수준에 닿은 step이다. 이 그림에서 선이 곧으면 로그처럼 느는 것이고, 수평이면 크기와 무관한 것이다. x 표시는 24,000 step 안에 닿지 못했다는 뜻이다.",
     ["50%에 닿는 step은 네 크기에서 4,200에서 4,600 사이로 수평이다. 80%는 10,000에서 15,000 사이다. 95%는 8개 환경만 닿았다.",
      "새 모델 둘과 52개, 64개 환경의 점은 run이 끝나는 대로 이 그림에 들어간다."]),
    ("숨은 program이 길어질 때", "by_depth.png",
     "판 하나가 환경 크기 하나다. 가로축은 숨은 program의 primitive 수이고 세로축은 transfer다. 모든 모델은 primitive 하나짜리 pair로만 학습했으므로 2개 이상은 전부 본 적 없는 과제다.",
     ["랩 NEO는 primitive 8개 환경에서 깊이 1, 2, 3, 4의 transfer가 97%, 84%, 62%, 35%다. 16개 이상에서는 깊이 2가 33% 근처, 깊이 3이 11%에서 16%, 깊이 4가 4%에서 7%다.",
      "환경이 8개에서 16개로 커질 때 한 번 크게 꺾이고, 그 뒤 32개까지는 비슷한 선이 유지된다. 깊이가 하나 늘 때마다 대략 절반이 된다."]),
    ("과제 종류별로 본 끝 성능", "final_by_size.png",
     "판 하나가 과제 종류 하나다. 가로축은 환경의 primitive 수다. 선이 오른쪽으로 갈수록 덜 떨어지는 모델이 확장성이 좋은 모델이다.",
     ["지금은 랩 NEO의 선만 있다. 네 판 모두 8개에서 16개로 갈 때 떨어지고 그 뒤로는 평평하다. 16개 묶음에서 새로 들어오는 것은 한 칸 이동, 180도와 270도 회전, 전치, 색을 하나 되돌리기다.",
      "새 모델 둘의 선과 52개, 64개 환경의 점이 들어오면, 이 그림에서 어느 선이 오른쪽으로 덜 내려가는지를 본다. 학습에서 뺀 primitive의 판은 52개와 64개 환경에서만 생긴다."]),
    ("모델을 나란히 놓은 학습 곡선", "curves_by_model.png",
     "열 하나가 환경 크기 하나이고 선 하나가 모델 하나다. 위는 학습 batch, 아래는 transfer다.",
     ["지금은 랩 NEO만 그려져 있다. 새 모델의 run이 끝나는 대로 선이 추가된다."]),
    ("작은 이동 묶음에서 본 것", "shift_family.png",
     "네 방향 1칸에서 6칸 이동 24개 가운데 네 칸 이동을 학습에서 뺀 환경이다. 세로축은 24개 전체의 설명률이다. 점선은 학습한 20개를 다 맞히고 뺀 4개를 못 맞힐 때의 높이다. 점선 위로 올라간 선만 학습하지 않은 양을 풀고 있다.",
     ["왼쪽은 codebook만 바꾸고 executor를 한 번 실행하는 모델들이다. 어느 것도 점선을 넘지 못한다. 덧셈을 loss로 건 둘은 오르내리거나 0으로 무너진다.",
      "오른쪽은 양을 실행 횟수로 적는 모델들이다. code 12개는 7,200 step부터 끝까지 100%다. 6개는 100%에 닿았다가 떨어지고 다시 올라온다. 8개는 오르내리고 4개는 못 배운다.",
      "이 환경은 작아서 여기서의 숫자 자체는 중요하지 않다. 읽을 것은 선의 모양이다. 횟수로 적는 모델은 답에 닿지만 학습이 그 답을 안정적으로 지키지 못한다."]),
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
