"""Experiments of 2026-11-23 afternoon to 2026-11-24: search, the oracle ceiling, EM, dream training, two-pair tasks."""

from .base import ASSIGN, DATA, EXTRA, INFER, ONE_STEP, TWO_PAIR_EM

OVERVIEW = "project-overview.md"
ABLATIONS = "ablations.md"

SEARCH = ("support로 검증하는 탐색. 단계마다 programmer 출력에 가까운 code 여러 개를 executor로 실행하고, programmer의 선호를 누적한 점수로 상위 몇 갈래만 남긴다. "
          "support 둘을 모두 재현하는 첫 program을 query에 쓴다. 남기는 갈래 수가 폭이고 폭 1은 탐색 없는 이어 쓰기다")
DREAM = ("꿈 학습. 한 단계 모델의 끝에서 programmer만 6,000스텝 더 학습한다. programmer가 고르는 빈도대로 code를 1개에서 8개 뽑아 얼린 executor로 실행하고, "
         "마지막 격자를 목표로 삼아 중간 격자마다 다음에 실행한 code를 맞히게 한다. 나머지 부품은 얼린다")
DREAM_MIX = DREAM + ". 위치의 절반은 실제 학습 pair로 채운다. pair마다 모든 code를 실행해 한 code가 정확히 재현할 때만 남기고 그 code를 목표로 삼는다"

EXPERIMENTS = [
    {
        "title": "support로 검증하는 program 탐색",
        "date": "2026-11-23",
        "source": "`docs/project-overview.md` 5.13절, 도구는 `scripts/babytheorist/support_search.py`",
        "question": "한 단계 모델을 이어 쓰면 숨은 길이 8에서 transfer가 7%다. 다시 학습하지 않고 탐색만 바꿔 막을 수 있는가.",
        "did": "8개 환경에서 한 단계로만 학습한 seed 여섯을 그대로 두고 평가 때의 탐색만 바꿨다. 레벨 1, 5, 7의 평가 과제 2,000개씩을 최대 10단계, 폭 1, 4, 8, 16으로 풀었다. "
               "처음에는 갈래를 support 출력과의 칸별 오차로 줄 세웠는데 폭 4가 폭 1보다 나빠서 programmer의 선호를 누적한 점수로 바꿨다.",
        "diff": {**ONE_STEP, INFER: SEARCH + ". 최대 10단계"},
        "items": [
            ("table", OVERVIEW, "| 숨은 길이 | 폭 1 | 폭 4 | 폭 8 | 폭 16 |", "탐색의 폭에 따른 길이별 transfer, seed 여섯 평균",
             "모델은 앞의 한 단계 모델 그대로이고 다시 학습하지 않았다. 평가 때의 추론이 랩과 다르다. 랩은 탐색하지 않는다."),
            ("fig", "reports/2026-11-23-search/search_by_length.png", "탐색의 폭에 따른 길이별 transfer", "위 표와 같은 측정이다."),
            ("fig", "reports/2026-11-23-search/search_checks.png", "둘째 support로 걸러지는 비율과, support 둘을 통과하고도 query에서 틀리는 비율", "위 표와 같은 측정이다. support로 program을 거르는 단계는 랩 NEO에 없다."),
        ],
        "result": "길이 5에서 8 사이는 탐색만으로 2배에서 3배가 된다. code가 깨끗한 seed는 길이 5에서 8을 모은 transfer가 25% 근처에서 47에서 55%로 오르고, code가 갈라진 seed는 0에서 23%다. "
                  "두 support를 다 통과하고도 query에서 틀리는 비율은 길이 5 이후 20에서 30%다. 길이 8에서는 폭 16이어도 과제의 70%쯤이 두 support를 통과하는 program을 찾지 못한다.",
        "insights": [
            ("revise", "length_decay", "긴 program에서 떨어지는 큰 부분은 탐색 없이 이어 쓰기 때문이고, 탐색을 넓히면 2배에서 3배가 된다",
             "한 단계 모델을 탐색 없이 이어 쓰면 길이가 늘수록 transfer가 급히 준다. 그 큰 부분은 한 번 틀리면 되돌리지 못하는 이어 쓰기 탓이다. support로 검증하는 탐색을 폭 16으로 넓히면 다시 학습하지 않고도 길이 8의 transfer가 7%에서 22%가 된다. "
             "갈래의 순위는 support 출력과의 오차가 아니라 programmer의 선호를 누적한 점수로 매겨야 한다.",
             "8개 환경 seed 여섯에서 폭 1, 4, 8, 16을 나란히 쟀다. 오차로 줄 세우면 폭 4가 폭 1보다 나빴다."),
            ("new", "false_pass", "plan", "긴 program에서는 support를 통과한 틀린 program보다 찾지 못하는 것이 더 큰 손실이다",
             "support 둘을 통과하고도 query에서 틀리는 program이 길이 5 이후 20에서 30% 있다. 그러나 더 큰 손실은 찾지 못하는 것이다. 길이 8에서는 폭 16이어도 과제의 70%쯤이 두 support를 통과하는 program을 찾지 못한다."),
        ],
    },
    {
        "title": "정답 program을 그대로 실행한 상한",
        "date": "2026-11-23",
        "source": "`docs/project-overview.md` 5.14절, 도구는 `scripts/babytheorist/oracle_program.py`",
        "question": "길이 8에서 탐색이 program을 찾지 못하는 이유가 executor의 누적 오차인가, programmer가 맞는 code를 후보에 올리지 못하는 것인가.",
        "did": "primitive마다 깊이 1 과제에서 가장 자주 정확히 재현한 code를 그 primitive의 code로 정했다. 숨은 program을 그 code의 열로 바꿔 query 입력에 그대로 실행하고 단계 사이에는 격자만 넘겼다. 같은 seed 여섯, 같은 평가 과제다.",
        "diff": {**ONE_STEP, INFER: "진단이다. 숨은 program을 primitive별 code의 열로 바꿔 query에 그대로 실행한다. 숨은 program을 읽는다"},
        "items": [
            ("table", OVERVIEW, "| seed | 정답 program 실행 | 탐색 폭 16 | greedy |", "길이 8의 query 정답률, 8개 환경",
             "모델은 한 단계 모델 그대로다. 첫 열은 숨은 program을 읽는 진단이고, 둘째 열은 앞 실험의 탐색, 셋째 열은 탐색 없는 이어 쓰기다."),
        ],
        "result": "code가 깨끗한 seed의 executor는 여덟 단계를 이어 실행해도 92에서 97%를 맞힌다. 같은 executor로 programmer가 고른 program은 탐색 폭 16에서 30에서 42%다. code가 갈라진 seed는 이 상한부터 낮다.",
        "insights": [
            ("new", "exec_chain", "exec", "code가 깨끗하면 executor는 여덟 단계를 이어 실행해도 거의 틀리지 않는다",
             "8개 환경에서 code가 깨끗한 seed의 executor는 정답 program을 여덟 단계 이어 실행해도 92에서 97%를 맞힌다. 한 단계로만 학습했는데도 그렇다."),
            ("new", "bottleneck", "plan", "긴 program의 병목은 executor가 아니라 programmer의 계획이다",
             "긴 program에서 떨어지는 것은 executor가 아니라 programmer가 여러 단계 떨어진 목표를 보고 첫 code를 고르지 못하는 데서 온다. 길이 8에서 정답 program의 상한이 92에서 97%인데 탐색은 30에서 42%다. programmer는 학습 때 한 단계 떨어진 목표만 봤다."),
        ],
    },
    {
        "title": "code NMI를 평가에 넣기",
        "date": "2026-11-23",
        "source": "`docs/project-overview.md` 5.15절",
        "question": "transfer 하나로는 code가 primitive와 어떻게 대응하는지 보이지 않는다. 회전 계열 문제의 판정에 쓸 숫자가 있는가.",
        "did": "평가가 매번 code NMI를 내게 했다. 깊이 1 과제에서 primitive와 고른 code의 정규화 상호정보량이다. primitive마다 자기 code를 하나씩 쓰면 1이고, primitive끼리 code를 나눠 쓰거나 한 primitive가 여러 code로 흩어지면 낮다.",
        "diff_note": "모델과 학습은 앞 실험들의 것 그대로이고 재는 숫자만 더했다. 숨은 primitive는 채점에만 쓴다.",
        "diff": {},
        "items": [
            ("inline", """| run | 회전 계열 일곱만의 code NMI |
|---|---|
| 8개 환경 한 단계 모델 | 0.86 |
| primitive마다 정답 code를 준 진단 | 0.67 |
| 200,000스텝 학습 | 0.37 |
| 굳은 뒤에 시작한 soft EM | 0.16, 0.20 |
| 24개 환경 한 단계, 부품과 code 고르는 방식은 랩 그대로 | 0.14 |""",
             "이미 끝난 run에서 잰 회전 계열의 code NMI",
             "앞 실험의 run들에 새 지표를 잰 것이다. 각 run의 설정은 그 실험에 적은 대로다."),
        ],
        "result": "회전 계열 일곱만의 NMI는 transfer와 같은 순서로 갈린다. 전체 primitive로 잰 NMI는 0.73에서 0.95로 덜 갈려서, 판정에는 회전 계열만의 NMI를 쓴다.",
        "insights": [
            ("new", "nmi", "measure", "회전 계열만의 code NMI가 code 배정의 상태를 transfer와 같은 순서로 보여 준다",
             "primitive와 고른 code의 정규화 상호정보량을 회전 계열 일곱만으로 재면 transfer와 같은 순서로 갈린다. 8개 환경 0.86, 24개 환경 랩 방식 0.14다. 전체 primitive로 재면 덜 갈린다."),
        ],
    },
    {
        "title": "처음부터 학습하는 hard EM과 soft EM, pair 하나",
        "date": "2026-11-23",
        "source": "`docs/project-overview.md` 5.16절, 모델은 `src/models/neo_em.py`",
        "question": "앞선 soft EM은 넷이 이미 한 code로 굳은 run의 끝에서 시작했다. 처음부터 EM으로 code를 배정하면 섞인 code가 풀리는가.",
        "did": "24개 환경에서 두 형태를 처음부터 24,000스텝, seed 셋으로 돌렸다. 둘 다 pair마다 codebook의 모든 code를 executor로 실행해 재현 오차를 잰다. "
               "판정 기준은 미리 정한 대로 가로세로를 바꾸는 넷의 seed 평균 60%다.",
        "diff": {**ONE_STEP, DATA: "24개 환경. 숨은 길이를 읽어 primitive 하나짜리 pair만 쓴다",
                 ASSIGN: "처음부터 EM. pair마다 모든 code를 실행해 재현 오차를 잰다. soft EM은 오차로 책임 비율을 정해 상위 네 code로 학습하며 온도를 10에서 0.1로 내리고, hard EM은 오차가 가장 작은 code 하나로만 학습한다. "
                         "programmer는 그 결과를 맞히게 학습한다. 아무도 고르지 않는 code는 오차를 깎아 주는 보너스로 구제한다"},
        "items": [
            ("table", OVERVIEW, "| 조건 | 가로세로를 바꾸는 넷 | 회전 계열 일곱 | 나머지 17개 | 회전 계열 NMI |", "pair 하나로 처음부터 학습한 세 조건, seed 셋 평균",
             "첫 줄은 부품과 code를 고르는 방식이 랩과 같은 한 단계 학습이다. 아래 두 줄은 code를 고르는 방식만 EM이다. 셋 다 pair 하나짜리 과제다."),
        ],
        "result": "둘 다 기준에 못 미친다. hard EM은 회전 계열 NMI를 두 배로 올렸지만 transfer가 거의 그대로이고 나머지 17개를 66%로 깎았다. 나머지 17개는 support를 98에서 100% 재현하는데 transfer는 69%이고 primitive 하나에 code가 평균 2.8개 붙는다.",
        "insights": [
            ("revise", "no_force", "pair 하나만 보는 학습 신호로는 처음부터 EM을 해도 섞인 code가 풀리지 않는다",
             "pair 하나를 재현하는지만 보는 학습 신호 안에는 한 code에 섞인 변환들을 떼어 놓는 힘이 없다. 굳은 뒤의 시도 열세 가지에 이어, 처음부터 하는 soft EM과 hard EM도 가로세로를 바꾸는 넷이 25%와 21%로 미리 정한 기준 60%에 못 미쳤다. "
             "hard EM은 code를 변환이 아니라 입력 모양별 전문가로 쪼갠다. pair 하나를 재현하는 것만 요구하면 잘게 쪼갤수록 재현은 쉬워지고 transfer는 떨어진다.",
             "hard EM의 나머지 17개가 support는 98% 넘게 재현하면서 transfer는 69%였고 primitive 하나에 code가 2.8개 붙었다."),
        ],
    },
    {
        "title": "꿈 학습: programmer가 스스로 만든 여러 단계 program으로 배우기",
        "date": "2026-11-23",
        "source": "`docs/project-overview.md` 5.17절, method는 `src/methods/neo_dream.py`",
        "question": "긴 program의 병목이 programmer의 계획이라면, 라벨 없이 그 계획을 가르칠 수 있는가.",
        "did": "8개 환경 한 단계 모델 seed 여섯 각각의 끝에서 programmer만 6,000스텝 더 학습했다. 모델 자신이 만든 여러 단계 program이 학습 데이터다. 격자가 바뀌지 않은 단계와 이미 지난 격자로 돌아오는 단계는 뺀다. 같은 탐색으로 쟀다.",
        "diff": {**ONE_STEP, EXTRA: DREAM, INFER: SEARCH},
        "items": [
            ("table", OVERVIEW, "| 숨은 길이 | 폭 16, 전 | 폭 16, 후 | greedy, 전 | greedy, 후 |", "꿈 학습 전후의 길이별 transfer, seed 여섯 평균",
             "전 열은 한 단계 모델 그대로이고 후 열은 programmer만 꿈으로 더 학습한 것이다. greedy 열은 탐색 없는 이어 쓰기다."),
            ("fig", "reports/2026-11-23-dream/dream_by_length.png", "꿈 학습 전후의 길이별 transfer", "위 표와 같은 측정이다."),
        ],
        "result": "폭 16에서 길이 8이 seed 평균 22%에서 38%로 올랐다. code가 많이 갈라진 seed는 0%에서 그대로다. 대가도 있다. 탐색 없이 쓰면 길이 1이 85%에서 76%로 떨어지고, support 둘을 통과하고도 query에서 틀리는 비율이 길이 8에서 24%에서 36%로 늘었다.",
        "insights": [
            ("new", "dream", "plan", "얼린 executor로 꾼 꿈으로 programmer의 여러 단계 계획을 일부 채울 수 있다",
             "확정된 executor로 모델이 스스로 만든 여러 단계 program을 programmer에게만 가르치면, 라벨 없이 긴 program이 오른다. 폭 16에서 길이 8이 22%에서 38%다. 다만 탐색 없이 쓰면 길이 1이 85%에서 76%로 떨어진다. 먼 목표를 배우면서 한 단계짜리 결정이 흐려진다."),
        ],
    },
    {
        "title": "같은 program의 pair 둘을 한 학습 과제로",
        "date": "2026-11-23",
        "source": "`docs/project-overview.md` 5.18절, 부품은 `babytheorist_episodes`, `neo_episode`, 두 pair 기준의 `neo_em`",
        "question": "지금의 학습 신호에는 같은 code가 다른 입력에서도 같은 일을 하라는 요구가 없다. 그 요구를 넣으면 섞인 code가 풀리는가.",
        "did": "학습 과제를 같은 program을 다른 물체 둘에 적용한 pair 둘로 줬다. 숨은 program 이름은 쓰지 않지만 어느 pair끼리 같은 규칙인지는 알려 준다. 24개 환경에서 seed 셋, 이어서 둘을 더 돌렸고 32개 환경은 seed 하나다. "
               "랩 방식은 첫 pair에서 고른 code를 두 입력에 모두 실행해 두 출력을 학습하고, hard EM은 두 pair의 재현 오차의 합으로 code를 고른다.",
        "diff": {**TWO_PAIR_EM},
        "items": [
            ("table", OVERVIEW, "| 조건 | 가로세로를 바꾸는 넷, seed별 | seed 평균 |", "pair 하나와 pair 둘, 랩 방식과 hard EM, 24개 환경",
             "랩 모델 줄은 code를 고르는 방식이 랩과 같다. 두 pair 줄은 과제가 pair 둘이다. hard EM 두 pair 줄이 이 실험의 공통 설정이다."),
            ("fig", "reports/2026-11-23-em/orientation_by_condition.png", "조건별 가로세로를 바꾸는 넷, 나머지 17개, 회전 계열 NMI, seed마다 점 하나", "위 표와 같은 조건들이다."),
            ("table", OVERVIEW, "| 모델 | 숨은 길이 1 | 2 | 3 | 4 | 전체 |", "한 단계로 학습한 모델을 최대 네 단계까지 이어 쓴 transfer, 24개 환경",
             "줄마다 학습 과제의 꼴과 code를 고르는 방식이 다르고, 평가 때 탐색 없이 이어 쓰는 것은 같다."),
        ],
        "result": "pair 둘을 주는 것만으로 랩 방식도 넷이 17%에서 34%가 된다. 두 pair hard EM은 seed 다섯에서 96%, 35%, 37%, 83%, 71%다. 미리 정한 seed 셋으로는 56%라 기준에 못 미치고 다섯으로는 64%다. "
                  "seed 42는 primitive마다 정답 code를 준 진단과 같은 수준까지 라벨 없이 갔다. seed 43과 44는 넷이 여전히 code 하나를 함께 쓴 채 멈췄다. 나머지 17개는 97%로 입력별 전문가 code가 사라졌다.",
        "insights": [
            ("revise", "no_force", "섞인 code를 푸는 데 빠져 있던 것은 같은 code가 다른 입력에서도 같은 일을 하라는 요구다",
             "섞인 code를 푸는 데 빠져 있던 것은 같은 code가 다른 입력에서도 같은 일을 하라는 요구다. pair 하나만 보면 어떤 방법도 풀지 못했다. 같은 program을 다른 물체 둘에 적용한 pair 둘을 한 과제로 주면 이 요구가 들어간다. "
             "그것만으로 랩 방식도 넷이 17%에서 34%가 되고, 두 pair의 오차로 code를 고르는 hard EM은 seed 다섯 평균 64%, 한 seed는 96%다. 입력별 전문가 code가 사라져 나머지도 97%다. "
             "미리 정한 seed 셋으로는 56%로 기준 60%에 못 미친다. pair의 묶음은 숨은 program보다 약하지만 벤치마크 학습 파일에는 없는 정보다.",
             "두 pair hard EM의 seed 42가 primitive마다 정답 code를 준 진단과 같은 수준까지 갔다."),
            ("new", "early", "code", "두 pair hard EM에서도 초반에 갈라지지 않은 seed는 넷이 한 code를 쓴 채 굳는다",
             "두 pair hard EM에서도 초반에 갈라지지 않은 seed는 넷이 code 하나를 함께 쓴 채 35% 근처에서 굳는다. 잘 풀린 seed는 7,200스텝에 이미 95%였다."),
            ("revise", "one_step", "깨끗한 한 단계 code는 이어 쓸 수 있고, code가 깨끗할수록 여러 단계 과제도 나아진다",
             "primitive 하나짜리 과제로 program 한 단계만 학습한 모델은 추가 학습 없이 여러 단계로 이어 쓸 수 있다. code가 깨끗할수록 잘 된다. 24개 환경에서 길이 2의 transfer가 랩 방식 35%, 두 pair hard EM 78%다. "
             "다만 한 단계 과제만 남기는 데 숨은 길이를 읽는다.",
             "두 pair hard EM 모델을 네 단계까지 이어 쓴 값을 랩 방식과 나란히 쟀다."),
        ],
    },
    {
        "title": "꿈 학습에 검증된 실제 pair 섞기",
        "date": "2026-11-24",
        "source": "`docs/project-overview.md` 5.19절",
        "question": "꿈 학습만 하면 탐색 없이 쓸 때 길이 1이 85%에서 76%로 떨어진다. 이 손해를 없앨 수 있는가.",
        "did": "꿈 학습의 위치 절반을 실제 학습 pair로 채웠다. 실제 pair마다 모든 code를 실행해 재현 오차가 가장 작은 code의 출력이 정확히 같을 때만 남긴다. 숨은 깊이 대신 executor로 검증해 한 단계짜리 pair를 고르는 것이다. 8개 환경 seed 여섯이다.",
        "diff": {**ONE_STEP, EXTRA: DREAM_MIX, INFER: SEARCH},
        "items": [
            ("table", OVERVIEW, "| 숨은 길이 | 폭 16: 전 / 꿈만 / 실제 pair 섞음 |", "꿈만 한 것과 검증된 실제 pair를 섞은 것, seed 여섯 평균",
             "셋 다 같은 한 단계 모델에서 출발하고 programmer만 더 학습한다. 꿈 학습은 랩 NEO에 없는 단계다."),
            ("fig", "reports/2026-11-24-dream-mix/dream_by_length.png", "검증된 실제 pair를 섞은 꿈 학습 전후", "위 표와 같은 측정이다."),
        ],
        "result": "길이 1의 손실이 사라졌고 긴 program은 꿈만 했을 때보다 높다. 탐색 없이 쓴 길이 3이 44%에서 62%가 됐다. 폭 16 길이 8은 42%다. code가 크게 갈라진 seed는 그대로 0%다.",
        "insights": [
            ("revise", "dream", "꿈은 먼 목표로 가는 첫걸음을, 검증된 실제 pair는 한 단계짜리 결정을 가르친다",
             "확정된 executor로 모델이 스스로 만든 여러 단계 program을 programmer에게만 가르치면 라벨 없이 긴 program이 오른다. 꿈만 하면 한 단계짜리 결정이 흐려지는데, 위치의 절반을 executor가 한 code로 정확히 재현하는 실제 pair로 채우면 그 손해가 사라진다. "
             "8개 환경 폭 16에서 길이 8이 22%에서 42%이고 탐색 없이 쓴 길이 1은 86%로 유지된다.",
             "꿈만 한 것과 실제 pair를 섞은 것을 같은 seed 여섯에서 나란히 쟀다."),
        ],
    },
    {
        "title": "처음에 굳은 배정을 이른 탐색으로 풀기",
        "date": "2026-11-24",
        "source": "`docs/project-overview.md` 5.20절",
        "question": "두 pair hard EM에서 넷이 code 하나에 묶인 채 굳는 seed를 초반의 탐색을 강하게 해서 풀 수 있는가.",
        "did": "두 형태를 돌렸다. 구제 보너스를 한 스텝 0.5 nats씩 200까지 주는 것과, 온도를 10에서 시작해 첫 6,000스텝에 0.01로 내리는 것이다. 온도를 내린 run의 끝에서 온도를 다시 올렸다 내리며 12,000스텝 더 학습하는 것도 했다. "
               "남은 덩어리가 대칭인 물체에서 두 변환이 같은 결과를 내는 탓인지도 셌다.",
        "diff": {**TWO_PAIR_EM, ASSIGN: TWO_PAIR_EM[ASSIGN] + ". 형태 둘을 더 시험한다. 구제 보너스를 열 배로 키운 것과, 온도를 10에서 시작해 첫 6,000스텝에 0.01로 내리며 상위 네 code로 학습하는 것"},
        "items": [
            ("table", OVERVIEW, "| 형태 | seed 43 | seed 44 | seed 45 | 평균 | 나머지 17개 |", "가로세로를 바꾸는 넷의 끝값, 형태별",
             "기본 줄이 두 pair hard EM이고 아래 두 줄은 초반에 code를 고르는 방식만 다르다."),
            ("inline", """| 형태 | seed 42 | 43 | 44 | 45 | 46 | 평균 |
|---|---|---|---|---|---|---|
| 기본 | 96% | 35% | 37% | 83% | 71% | 64% |
| 온도를 처음부터 천천히 내리는 형태 | 50% | 73% | 51% | 61% | 50% | 57% |
| 그 run의 끝에서 다시 달군 뒤 | | 72% | 50% | 63% | | |""",
             "seed 다섯으로 본 두 형태와 다시 달구기",
             "공통 설정에서 온도의 시간표만 다르다."),
        ],
        "result": "구제 강화는 어느 seed가 풀리느냐만 바꿨다. 온도를 천천히 내리는 형태는 35%에 멈춘 seed가 없고 나머지가 모두 100%지만 완전히 풀리는 seed도 없다. 넷이 한꺼번에 묶였던 덩어리가 둘이나 셋짜리로 줄었을 뿐이다. 다시 달궈도 그대로다. "
                  "programmer 출력으로 굳은 code의 과제를 묶어 보면 순도가 0.28로 우연 수준이다. 남은 짝은 혼동으로 설명되지 않는다. 과제의 대부분은 두 변환을 구별할 정보를 갖고 있다.",
        "insights": [
            ("revise", "early", "배정은 초반에 정해지고, 굳은 뒤에는 보너스를 키워도 온도를 다시 올려도 풀리지 않는다",
             "두 pair hard EM에서 넷의 갈라짐은 2,400스텝과 7,200스텝 사이에 정해진다. 굳은 뒤에는 구제 보너스를 키워도, 온도를 다시 올렸다 내려도 풀리지 않는다. 온도를 처음부터 천천히 내리면 멈추는 seed가 사라지지만 완전히 풀리는 seed도 사라져 평균이 57%로 기본의 64%보다 낮다. "
             "programmer 출력에는 넷을 가를 정보가 없고, 남은 덩어리는 대칭인 물체에서의 혼동으로 설명되지 않는다.",
             "보너스 강화, 온도 시간표, 다시 달구기를 같은 seed에서 나란히 돌렸고, 굳은 code의 과제를 programmer 출력으로 묶은 순도가 우연 수준이었다."),
        ],
    },
    {
        "title": "깨끗한 code 위의 꿈 학습, 24개 환경",
        "date": "2026-11-24",
        "source": "`docs/project-overview.md` 5.21절",
        "question": "꿈 학습은 8개 환경에서만 시험했다. 두 pair hard EM으로 세운 24개 환경의 깨끗한 code 위에서도 programmer의 계획을 채우는가. code가 섞인 모델에서도 그런가.",
        "did": "두 pair hard EM의 code가 깨끗한 seed 셋, 넷이 섞인 채 멈춘 seed 하나, 그리고 pair 하나로 학습한 랩 방식 seed 하나 각각의 끝에서 검증된 실제 pair를 반 섞은 꿈 학습을 6,000스텝 했다. 24개 환경 평가 과제 3,000개를 폭 16으로 탐색해 쟀다.",
        "diff": {**TWO_PAIR_EM, EXTRA: DREAM_MIX, INFER: SEARCH + ". 폭 16"},
        "items": [
            ("table", OVERVIEW, "| 모델 | 꿈 학습 전 | 후 | 평가 파일 전체 |", "24개 환경에서 꿈 학습 전후의 숨은 길이 1, 2, 3, 4",
             "첫 줄은 pair 하나로 학습하고 code를 고르는 방식이 랩과 같은 모델이다. 나머지는 공통 설정인 두 pair hard EM이다. 모두 꿈 학습과 탐색을 쓴다."),
        ],
        "result": "code가 깨끗한 모델에서 꿈 학습은 길이 4를 seed 셋 평균 43%에서 54%로 올렸다. code가 섞인 랩 방식 모델에서는 거의 효과가 없다. support 둘을 통과하고도 query에서 틀리는 비율은 2에서 7%로 낮게 유지됐다.",
        "insights": [
            ("revise", "dream", "깨끗한 code가 먼저이고, 꿈 학습은 그 위에서 계획을 채운다",
             "확정된 executor로 모델이 스스로 만든 여러 단계 program과, executor가 한 code로 정확히 재현하는 실제 pair를 반씩 programmer에게만 가르치면 라벨 없이 긴 program이 오른다. "
             "깨끗한 code가 먼저다. 24개 환경에서 code가 깨끗한 모델은 길이 4가 43%에서 54%로 오르고, code가 섞인 모델에서는 거의 효과가 없다.",
             "같은 꿈 학습을 code가 깨끗한 모델 셋과 섞인 모델 둘에 했다."),
        ],
    },
    {
        "title": "라벨 없는 검증 점수로 run 고르기",
        "date": "2026-11-24",
        "source": "`docs/project-overview.md` 5.22절, 도구는 `scripts/babytheorist/pair_validation.py`",
        "question": "두 pair hard EM은 seed에 따라 넷이 35%에서 96%로 갈린다. 어느 run이 잘 풀렸는지를 라벨 없이 알 수 있는가.",
        "did": "학습에 쓰지 않은 두 pair 과제 4,000개를 새로 만들었다. programmer가 첫 pair를 보고 고른 code가 두 pair를 모두 재현하는 비율을 점수로 삼는다. 숨은 program도 깊이도 읽지 않는다. 두 pair hard EM 열 run에서 쟀다.",
        "diff_note": "모델은 앞의 두 pair hard EM run들이고 run을 고르는 점수만 더했다.",
        "diff": {**TWO_PAIR_EM, EXTRA: "run 고르기. 학습에 쓰지 않은 두 pair 과제에서 programmer가 고른 code가 두 pair를 모두 재현하는 비율이 가장 높은 run을 쓴다"},
        "items": [
            ("table", OVERVIEW, "| run | 검증 점수 | 가로세로를 바꾸는 넷 |", "라벨 없는 검증 점수와 가로세로를 바꾸는 넷",
             "run은 모두 두 pair hard EM이다. 검증 점수는 라벨 없이 재고, 넷의 값은 숨은 primitive로 채점한 것이다."),
        ],
        "result": "검증 점수와 넷의 순위 상관이 0.98이다. 가장 높은 점수의 run이 가장 잘 풀린 run이다. 섞인 code는 두 pair를 함께 재현하지 못하니 이 점수가 낮아진다.",
        "insights": [
            ("new", "validation", "measure", "학습에 쓰지 않은 두 pair 과제의 재현율이 잘 풀린 run을 라벨 없이 고른다",
             "학습에 쓰지 않은 두 pair 과제에서 programmer가 첫 pair를 보고 고른 code가 두 pair를 모두 재현하는 비율은 라벨 없이 잘 풀린 run을 고른다. 24개 환경 열 run에서 가로세로를 바꾸는 넷과의 순위 상관이 0.98이다."),
        ],
    },
    {
        "title": "평가 때의 탐색 다듬기: 남은 거리 예측기와 답 투표",
        "date": "2026-11-24, 일부 2026-11-27",
        "source": "`docs/ablations.md` 3절",
        "question": "탐색이 긴 program을 찾지 못하는 것과, support를 통과하고도 query에서 틀리는 것을 평가 때의 방법만으로 줄일 수 있는가.",
        "did": "꿈으로 학습한 남은 거리 예측의 기댓값을 갈래의 순위에 더했다. support를 통과한 program이 여럿일 때 처음 통과한 것을 쓰는 대신 그 program들을 query에 실행해 표를 모으는 방식 셋을 견줬다. 꿈 학습을 네 배로 늘린 것과 탐색 길이를 10에서 14로 늘린 것도 쟀다.",
        "diff": {**ONE_STEP, EXTRA: DREAM_MIX,
                 INFER: SEARCH + ". 여기에 둘을 더한다. 꿈으로 학습한 남은 거리 예측의 기댓값을 갈래의 순위에 더하고, support를 통과한 program이 여럿이면 query에 실행한 답으로 투표하되 program 길이가 하나 늘 때마다 표의 무게를 반으로 한다"},
        "items": [
            ("fig", "figures/neo_architecture.png", "11월 24일 시점의 최선을 한 장으로 그린 것. 깨어 있을 때의 검증된 탐색, code를 배우는 방식, 잠의 꿈 학습",
             "state encoder, state decoder, codebook, executor는 랩 부품 그대로다. 랩 NEO에 없는 것은 support로 검증하는 탐색, 남은 거리를 예측하는 critic, 두 pair를 모두 실행해 code를 고르는 E 단계와 죽은 code의 보너스, 그리고 꿈 학습이다."),
            ("table", ABLATIONS, "| 조건 | 결과 |", "support를 통과한 program이 여럿일 때 답을 고르는 방식. 앞에서부터 처음 통과, 같은 무게 투표, 가장 짧은 것만, 길이로 무게를 준 투표",
             "8개 환경 줄은 한 단계 모델에 꿈 학습을 한 것이고, 24개 환경 줄은 두 pair hard EM에 꿈 학습을 한 것이다. 평가 때 답을 고르는 방식만 다르다."),
            ("inline", """| 질문 | 조건 | 결과 |
|---|---|---|
| 남은 거리 예측기를 순위에 더하면 | 8개 환경 seed 42와 45, 레벨 5와 7, 폭 4와 16 | 여덟 조건 모두 오름. 폭 4에서 크고, 한 예가 67.5%에서 74.3% |
| 꿈 학습을 네 배인 24,000스텝으로 | 8개 환경 seed 42와 45 | seed 42는 2에서 4점, seed 45는 그대로 |
| 탐색 길이를 10에서 14로 | 레벨 7, 폭 16, seed 45와 42 | 처음 통과한 program으로 78.9%에서 80.7%, 71.6%에서 72.6% |""",
             "탐색의 다른 조정들",
             "모두 한 단계 모델에 꿈 학습을 한 것이고 평가 때의 탐색이나 꿈의 길이만 다르다."),
        ],
        "result": "길이로 무게를 준 투표가 일곱 조건 가운데 여섯에서 가장 좋았다. 같은 무게로 투표하면 긴 과제는 오르지만 짧은 과제가 내려간다. 남은 거리 예측기는 폭이 좁을 때 크게 돕는다. 꿈을 네 배로 늘린 이득은 작다.",
        "insights": [
            ("revise", "false_pass", "support를 통과한 program이 여럿이면 짧은 program에 무게를 주어 투표한다",
             "support 둘을 통과하고도 query에서 틀리는 program이 있고, 긴 program에서는 찾지 못하는 것이 더 큰 손실이다. 통과한 program이 여럿이면 query에 실행한 답으로 투표하되 길이가 하나 늘 때마다 무게를 반으로 하는 방식이 처음 통과한 program을 쓰는 것보다 낫다. "
             "8개 환경 레벨 7에서 71.6%가 77.4%가 된다. 찾지 못하는 쪽은 남은 거리 예측을 순위에 더하면 준다.",
             "투표 방식 셋과 예측기를 여러 seed와 레벨에서 견줬다."),
            ("revise", "dream", "깨끗한 code가 먼저이고, 꿈 학습은 그 위에서 계획을 채운다. 더 오래 꿔도 이득은 작다",
             "확정된 executor로 모델이 스스로 만든 여러 단계 program과, executor가 한 code로 정확히 재현하는 실제 pair를 반씩 programmer에게만 가르치면 라벨 없이 긴 program이 오른다. "
             "깨끗한 code가 먼저다. code가 섞인 모델에서는 거의 효과가 없다. 꿈 학습을 네 배로 늘려도 이득은 0에서 4점이다.",
             "꿈 학습을 24,000스텝으로 늘린 run 둘을 쟀다."),
        ],
    },
    {
        "title": "큰 환경의 code 형성, 깊이 없이 과제 고르기, 본보기로 code 나누기",
        "date": "2026-11-24에서 27",
        "source": "`docs/ablations.md` 1절과 4절",
        "question": "두 pair hard EM의 갈림이 52개와 64개 환경에서도 같은가. 숨은 길이를 읽지 않고 한 단계 과제를 고를 수 있는가. 섞인 code를 본보기 과제로 나눌 수 있는가.",
        "did": "52개와 64개 환경에서 두 pair hard EM의 기본 형태와 온도를 내리는 형태를 seed 셋씩 돌리고, 검증 점수가 잘 풀린 run을 고르는지 봤다. 숨은 길이 대신 가장 잘 맞는 code의 오차로 과제를 줄 세워 절반이나 30%만 학습하는 것을 돌렸다. "
               "섞인 code의 과제 일부를 본보기로 삼아 새 code를 학습한 뒤 재배정하는 것도 돌렸다.",
        "diff": {**TWO_PAIR_EM, DATA: "24, 32, 52, 64개 환경의 두 pair 과제. 한 갈래는 숨은 길이 대신 모델의 오차로 과제를 고른다"},
        "items": [
            ("table", ABLATIONS, "| 환경 | 랩 pair 하나 | 랩 pair 둘 |", "환경 크기별 가로세로를 바꾸는 넷",
             "랩 pair 하나 열은 부품과 code를 고르는 방식이 랩과 같은 한 단계 학습이고, 랩 pair 둘 열은 거기에 두 pair 과제만 더한 것이다. 나머지 열이 hard EM이다."),
            ("inline", """| 질문 | 비교 | 결과, 넷 | 결과, 나머지 |
|---|---|---|---|
| 숨은 길이를 읽지 않고 한 단계 과제를 고르면 | 가장 잘 맞는 code의 오차로 과제를 줄 세워 절반만 학습 | 34%, 33%, 35% | 89%, 99%, 99% |
| | 같은 방식으로 30%만 학습 | 40%, 57%, 46% | 70%, 68%, 67% |
| 섞인 code를 본보기 과제로 나누면 | 본보기로 새 code를 학습한 뒤 재배정하고 12,000스텝 이어 학습 | 학습 과제 재현은 35%에서 94%, 평가는 34%. 이어 학습 33%, 대조 35% | |""",
             "숨은 길이 없이 과제 고르기와 본보기로 code 나누기, 24개 환경",
             "두 pair hard EM에서 학습 과제를 고르는 방식, 또는 굳은 뒤에 code를 나누는 단계만 다르다."),
        ],
        "result": "52개와 64개 환경에서도 기본 형태는 seed 셋 가운데 하나만 98%와 92%로 풀리고 나머지는 37에서 39%다. 검증 점수의 순위는 넷의 순위와 정확히 같아 크기마다 가장 잘 풀린 run을 고른다. "
                  "숨은 길이를 읽지 않고 과제를 고르면 깊이로 고른 데이터에 못 미친다. 많이 버리면 넷은 오르고 나머지가 내려간다. 본보기로 만든 새 code는 학습 과제는 재현하지만 평가에서는 그대로다.",
        "insights": [
            ("revise", "validation", "학습에 쓰지 않은 두 pair 과제의 재현율이 잘 풀린 run을 라벨 없이 고른다. 큰 환경에서도 그렇다",
             "학습에 쓰지 않은 두 pair 과제에서 programmer가 첫 pair를 보고 고른 code가 두 pair를 모두 재현하는 비율은 라벨 없이 잘 풀린 run을 고른다. 24개 환경 열 run에서 순위 상관이 0.98이고, 52개와 64개 환경에서는 순위가 정확히 같다. "
             "seed 여러 개를 돌려 이 점수로 고르면 세 크기에서 넷이 96에서 98%인 run이 나온다.",
             "52개와 64개 환경의 run 여섯씩에서 같은 점수를 쟀다."),
            ("new", "scale_same", "grow", "두 pair hard EM의 seed에 따른 갈림은 52개와 64개 환경에서도 같다",
             "두 pair hard EM은 환경이 커져도 같은 식으로 갈린다. 가로세로를 바꾸는 넷이 52개 환경에서 98%, 39%, 38%, 64개 환경에서 37%, 92%, 38%다."),
            ("new", "depth_free", "code", "숨은 길이를 읽지 않고 한 단계 과제를 고르는 방법은 아직 없다",
             "한 단계 학습은 primitive 하나짜리 과제를 고르는 데 숨은 길이를 읽는다. 모델의 오차로 과제를 줄 세워 고르면 깊이로 고른 데이터에 못 미친다. 절반을 남기면 넷이 34% 근처이고, 30%만 남기면 넷은 오르지만 나머지가 68%로 내려간다."),
            ("new", "exemplar", "code", "굳은 code를 본보기 과제로 나누면 새 code가 변환이 아니라 그 물체들을 외운다",
             "굳은 뒤에 섞인 code의 과제 일부를 본보기로 삼아 새 code를 학습하면, 학습 과제 재현은 35%에서 94%로 오르지만 평가는 34%로 그대로다. 새 code가 변환이 아니라 본보기의 물체를 외운다."),
        ],
    },
]
