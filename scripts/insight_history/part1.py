"""Experiments of 2026-11-20 to 2026-11-23 morning: the lab setting on BabyTheorist, the cheap one-step screen, amount,
the size ladder, the growing environment and the orientation family."""

from .base import ASSIGN, BOOK, CODEC, DATA, EXEC, EXTRA, INFER, INFO, ONE_STEP, PROG, SCHED, STEPS

OVERVIEW = "project-overview.md"

INTRO = (
    "이 문서는 toy-NEO에서 한 실험을 처음부터 순서대로 놓은 것이다. 실험마다 네 가지를 적는다. 무엇을 물었고 무엇을 했는지, "
    "그 실험의 설정이 랩 NEO와 어디가 다른지, 그 실험의 figure와 table, 그리고 그 실험까지 쌓인 insight 전체다. "
    "figure와 table마다 그것이 랩 NEO와 무엇이 다른 설정에서 나왔는지를 한 줄씩 붙였다. "
    "insight 정리에서는 그 실험이 새로 알려 준 것과 앞선 말을 고친 것을 온전한 문장으로 적고, 그대로인 것은 한 줄씩으로 줄여 적는다. "
    "앞선 말이 고쳐질 때는 고치기 전의 말과 무엇을 보고 고쳤는지를 함께 남긴다. 마지막 절에 지금의 정리 전체가 온전한 문장으로 있다. "
    "table은 원래 문서에서 그대로 가져왔고, 이 문서는 `scripts/build_insight_history.py`로 다시 만든다."
)

LAB_NOTE = (
    "랩 NEO는 MLML 랩의 GeneralizedL2T에 있는 transformer NEO를 BabyTheorist의 15칸 격자로 옮긴 설정을 말한다. "
    "첫 실험만 랩의 공식 NEO, 곧 격자 하나를 latent 하나에 담는 GridWorld 설정을 기준으로 한다. "
    "랩이 논문에서 BabyTheorist에 쓴 설정과 코드는 받지 못했다. 아래 표의 항목이 실험마다 다른 점을 적는 틀이다."
)

EXPERIMENTS = [
    {
        "title": "랩의 공식 NEO를 GridWorld 설정 그대로 BabyTheorist에",
        "date": "2026-11-20",
        "source": "`docs/reports/2026-11-20-first-numbers.md`와 `docs/project-overview.md` 5.1절",
        "question": "지금의 랩 NEO가 BabyTheorist의 가장 쉬운 설정에서 무엇을 하는가. 그 전에 state encoder와 state decoder가 15칸 격자를 그대로 다시 그리는가.",
        "did": "랩 CNN state encoder와 state decoder는 격자가 10칸이 아니면 오류를 내서, 층은 같고 크기 계산만 다른 부품을 만들었다. "
               "latent 크기 다섯 가지와 칸마다 token을 두는 방식을 사전학습하고, 그 위에 공식 NEO를 32차원과 256차원 state로, MLP 폭을 키운 변형 둘로, 이동 넷만 쓰는 데이터로 seed 셋씩 학습했다.",
        "diff_note": "이 실험만 랩의 공식 NEO를 쓴다. 아래 표의 오른쪽 열이 공식 NEO의 부품이다.",
        "diff": {
            CODEC: "랩 CNN과 층이 같고 크기 계산만 다른 cnn_vae_any로 격자 하나를 latent 하나에 담는다. latent는 32차원에서 256차원까지 다섯 크기",
            PROG: "MLP. 폭을 키운 변형 둘 포함",
            BOOK: "code 6개. 학습 중에는 표본으로 뽑는다",
            EXEC: "FiLM MLP",
            STEPS: "최대 네 단계. latent를 그대로 넘긴다",
            SCHED: "학습률 5e-4, batch 128, 39,100스텝",
        },
        "items": [
            ("inline", """| state 표현 | 물체 하나 | 물체 둘 | 물체 셋 |
|---|---|---|---|
| 32차원 latent 하나 | 0.89 | 0.60 | 0.23 |
| 256차원 latent 하나 | 0.997 이상 | 0.997 이상 | 0.997 이상 |
| 칸마다 token | 1.0 | 1.0 | 1.0 |""",
             "state encoder와 state decoder가 격자를 칸 하나 틀리지 않고 다시 그리는 비율",
             "state encoder와 state decoder만 잰 것이다. 32차원 줄이 랩 GridWorld 설정의 크기이고 나머지 두 줄은 이 레포가 바꾼 크기다."),
            ("inline", """| 설정 | 예시 쌍 재현 | transfer |
|---|---|---|
| 공식 NEO, GridWorld 설정 그대로, seed 셋 평균 | 1% | 0.02% |
| latent를 256차원으로 | | 0에 가까움 |
| MLP를 넓힘, 또는 primitive를 이동 넷으로 줄임 | | 0 |""",
             "공식 NEO의 BabyTheorist 레벨 1 성능",
             "첫 줄은 격자 크기에 맞춘 state encoder 말고는 랩 GridWorld 설정 그대로다. 아래 줄은 latent 크기, MLP 폭, 데이터의 primitive 수를 하나씩 바꿨다."),
            ("fig", "reports/first-numbers/failures_mlp_d32_s43_depth1.png", "공식 NEO가 primitive 하나짜리 과제에서 내놓은 틀린 답의 예, 32차원 latent, seed 43",
             "격자 크기에 맞춘 state encoder 말고는 랩 GridWorld 설정 그대로다."),
        ],
        "result": "32차원 latent 하나는 15칸 격자를 다시 그리지 못한다. latent를 키우면 그리지만 transfer는 그대로 0에 가깝다. "
                  "공개된 GridWorld 설정으로는 논문의 BabyTheorist 점수가 나오지 않는다. 이 관찰로 칸마다 token을 두는 transformer NEO로 넘어갔다.",
        "insights": [
            ("new", "state_size", "state", "격자 하나를 latent 하나에 담는 것은 15칸 격자에 모자라지만, 공식 NEO가 0에 머무는 이유는 그것만이 아니다",
             "격자 하나를 32차원 latent 하나에 담는 랩 GridWorld 설정은 15칸 격자를 다시 그리지 못한다. 물체 셋이면 23%만 온전히 그린다. 칸마다 token을 두면 100%다. "
             "그런데 latent를 키워도 공식 NEO의 transfer는 0에 가깝다. 표현 크기만의 문제가 아니다."),
        ],
    },
    {
        "title": "랩 transformer NEO의 첫 숫자와 code 대 primitive 표",
        "date": "2026-11-20에서 21",
        "source": "`docs/project-overview.md` 5.2절",
        "question": "칸마다 token을 두는 랩 transformer NEO는 BabyTheorist 레벨 1을 얼마나 배우는가. 막힌다면 primitive 하나를 배우는 데서 막히는가, 이어 붙이는 데서 막히는가.",
        "did": "랩 transformer NEO 설정을 15칸 격자로 옮겨 seed 다섯을 12,000스텝, seed 하나를 36,000스텝 학습했다. 진단으로 primitive 하나짜리 과제만으로 학습한 run과 이동 넷만 쓰는 run을 더했다. "
               "길이 1 과제에서 고른 code 번호와 숨은 primitive를 센 표를 run마다 남겼다.",
        "diff": {SCHED: "12,000스텝 또는 36,000스텝. 랩 일정은 13시간이 걸려 줄였다"},
        "items": [
            ("table", OVERVIEW, "| run | 예시 쌍 재현 | transfer |", "랩 transformer NEO의 레벨 1 재현과 transfer",
             "학습 스텝 수 말고는 랩 설정 그대로다. 아래 두 줄은 진단으로 학습 데이터만 바꿨다. 하나는 숨은 길이를 읽어 primitive 하나짜리 과제만 남겼고, 하나는 primitive를 이동 넷으로 줄였다."),
            ("fig", "reports/first-numbers/codes_over_training.png", "학습이 진행되는 동안 primitive마다 쓰는 code가 바뀌는 모습", "학습 스텝 수 말고는 랩 설정 그대로다."),
            ("fig", "reports/first-numbers/curves_by_flops.png", "학습에 든 계산을 가로축으로 한 학습 곡선", "학습 스텝 수 말고는 랩 설정 그대로다."),
            ("fig", "reports/first-numbers/failures_transformer_s42_depth1.png", "primitive 하나짜리 과제에서 틀린 답의 예, seed 42", "학습 스텝 수 말고는 랩 설정 그대로다."),
        ],
        "result": "12,000스텝의 transfer는 seed에 따라 3%에서 25%이고 36,000스텝은 40%다. primitive 하나짜리 과제에서 같은 code 하나가 예시 쌍과 새 입력 쌍을 모두 재현하면 transfer가 100%였고, "
                  "primitive 둘짜리 과제의 transfer는 하나짜리의 제곱에 가까웠다. code는 색 바꾸기, 이동, 뒤집기, 회전 순으로 얻고 회전은 어떤 설정에서도 배우지 못했다. "
                  "처음에는 primitive가 초반에 갈리고 안 바뀐다고 가설을 세웠는데, 체크포인트 열 개를 따라가 보니 여러 code를 이어 설명하는 단계를 거쳐 차례로 code를 얻었다.",
        "insights": [
            ("new", "one_code", "code", "transfer를 정하는 것은 어디서나 통하는 code 하나를 가진 primitive의 수다",
             "transfer는 같은 code 하나가 support pair와 query를 모두 재현할 때만 된다. 전체 transfer를 정하는 것은 어떤 입력에서도 통하는 code 하나를 가진 primitive의 수다. "
             "랩 transformer NEO가 막히는 곳은 이어 붙이기보다 primitive마다 code 하나를 찾는 일이다."),
            ("new", "settle", "code", "여러 단계로 학습하면 primitive는 여러 code로 설명되는 단계를 거쳐 차례로 code를 얻는다",
             "여러 단계로 학습하는 랩 transformer NEO에서 primitive는 code 하나로 굳기 전에 여러 code를 이어 설명하는 단계를 거친다. code를 얻는 순서는 seed마다 같아 색 바꾸기, 이동, 뒤집기, 회전이다."),
            ("new", "easy_only", "code", "primitive 하나짜리 과제만 주면 오히려 덜 배운다",
             "랩 설정에서 primitive 하나짜리 과제만으로 학습하면 primitive 둘만 배우고 transfer 23%에서 멈춘다. 쉬운 과제만 주면 primitive를 더 못 찾는 것으로 보인다."),
            ("new", "seeds", "measure", "seed 하나의 값으로 말하지 않는다",
             "seed마다 차이가 크다. 같은 랩 설정의 transfer가 3%에서 25%로 갈린다. seed 하나의 값으로 말하지 않는다."),
        ],
    },
    {
        "title": "잠: 겹친 code를 합치고 비운 code를 다시 쓰기",
        "date": "2026-11-21",
        "source": "`docs/project-overview.md` 5.3절",
        "question": "잠을 넣으면 code를 얻지 못한 primitive가 code를 얻는가.",
        "did": "code 8개로 학습한 모델의 14,400스텝에서 색 바꾸기가 code 둘을 차지하고 회전과 아래로 이동은 code가 없었다. 잠에서 겹친 code 둘을 합쳐 하나를 비우고, 비운 code를 설명하지 못하는 학습 쌍에 배정한 뒤 36,000스텝까지 이어 학습했다. "
               "대조로 합치기만 한 것과 비운 code를 아무 방향에나 둔 것을 돌렸다. 이어서 꺼 둔 code를 켜는 codebook으로 잠을 여러 번 넣는 형태도 돌렸다.",
        "diff": {
            BOOK: "code 8개. 넓힌 형태는 꺼 둔 줄을 켤 수 있는 vq_growable이고 모든 줄을 켜면 랩 VQ와 출력이 같다",
            SCHED: "36,000스텝",
            EXTRA: "잠. 겹친 code 둘을 합쳐 하나를 비우고 설명하지 못하는 학습 쌍에 배정한다. 떼어 둔 저장분의 재현이 1%p 넘게 떨어지지 않을 때만 받아들인다",
        },
        "items": [
            ("table", OVERVIEW, "| 조건 | 최종 transfer |", "잠 한 번과 대조의 최종 transfer, seed 42",
             "code 수 8개와 잠 말고는 랩 설정 그대로다. 맨 아래 줄은 code 16개인 랩 설정이다. 단계 사이에 확률을 넘기는 랩 방식으로 잰 값이라 다음 실험에서 다시 읽게 된다."),
            ("fig", "reports/first-numbers/curves_by_flops_sleep.png", "잠을 넣은 run과 대조의 학습 곡선, 가로축은 학습에 든 계산", "code 8개와 잠 말고는 랩 설정 그대로다."),
            ("fig", "reports/first-numbers/curves_by_flops_growing.png", "잠에서 code를 켜 가는 형태의 학습 곡선", "codebook이 vq_growable이고 잠을 여러 번 넣는다. 나머지는 랩 설정 그대로다."),
        ],
        "result": "비운 code를 설명하지 못하는 쌍에 배정한 run만 61%로 올랐고 합치기만 한 run은 그대로였다. 겹친 code가 있는 때는 seed 42의 한 시점뿐이었다. "
                  "code를 켜 가는 형태는 재현만 올리고 transfer는 그대로였다. 새 code가 이미 code를 가진 primitive를 입력에 따라 나누는 데 쓰였다.",
        "insights": [
            ("new", "first_sleep", "memory", "잠에서 비운 code를 설명 못 하는 쌍에 배정하면 transfer가 오른다",
             "잠에서 겹친 code를 합쳐 비우고 그 code를 설명하지 못하는 학습 쌍에 배정하면 transfer가 44%에서 61%로 오른다. 효과는 합치기가 아니라 배정에서 난다. seed 하나의 값이다."),
            ("new", "split_by_input", "code", "code를 더 주면 primitive가 아니라 입력에 따라 쪼개는 데 쓰인다",
             "code를 더 켜 주면 재현은 오르지만 transfer는 그대로다. 새 code가 code 없는 primitive에 가지 않고, 이미 code를 가진 primitive를 입력에 따라 나누는 데 쓰인다."),
        ],
    },
    {
        "title": "단계 사이로 넘어가는 보이지 않는 정보",
        "date": "2026-11-21에서 22",
        "source": "`docs/project-overview.md` 5.4절",
        "question": "잠을 넣은 모델은 위아래 뒤집기를 code 하나로, 좌우 뒤집기를 그 code 두 번으로 설명하고 85%를 옮겼다. 두 번 뒤집으면 원래 격자여야 하는데 어떻게 된 것인가.",
        "did": "중간 격자를 봤다. 36,000스텝 학습 다섯을 단계 사이에 격자만 넘기는 방식으로 다시 쟀다. 통로를 학습 때부터 막는 두 모델도 돌렸다. 랩 코드에 있는 형태와, 다음 단계가 격자에서 시작하되 gradient는 흘리는 새 부품이다.",
        "diff": {STEPS: "평가에서 단계 사이에 확률이 가장 높은 색만 남긴 격자를 넘긴다. 학습 때부터 막는 형태는 랩 코드의 reencode_hard와 새 부품 neo_visible_state"},
        "items": [
            ("fig", "reports/first-numbers/handover_rules.png", "단계 사이에 확률을 넘길 때와 격자만 넘길 때 같은 program이 내는 중간 격자",
             "모델은 랩 설정으로 학습한 것이고, 실행할 때 단계 사이로 넘기는 것만 바꿨다."),
            ("inline", """| run | 확률을 넘겨 잰 transfer | 격자만 넘겨 잰 transfer |
|---|---|---|
| 36,000스텝 학습 다섯 | 최고 61.4% | 40.0%, 41.8%, 39.8%, 42.8%, 36.8% |
| 그 가운데 잠 한 번을 넣은 run | 61.4% | 36.8% |""",
             "같은 모델을 두 방식으로 잰 transfer",
             "학습은 랩 설정이거나 앞 실험의 잠을 넣은 것이고, 평가에서 단계 사이로 넘기는 것만 다르다."),
        ],
        "result": "확률을 넘기면 둘째 단계가 좌우로 뒤집은 정답을 내고, 격자만 넘기면 입력으로 되돌아간다. 같은 code가 깨끗한 격자를 받으면 위아래로 뒤집고 자기가 낸 확률을 받으면 다른 일을 한다. "
                  "격자만 넘기면 다섯 run이 모두 37에서 43%이고 잠의 61%도 37%가 된다.",
        "insights": [
            ("new", "handover", "state", "단계 사이에 확률을 넘기면 격자에 보이지 않는 정보가 건너간다",
             "랩 transformer NEO는 program의 단계 사이에 칸마다의 색 확률을 넘기고, 학습된 모델은 격자에 보이지 않는 그 부분으로 정보를 넘긴다. 같은 code가 깨끗한 격자와 자기가 낸 확률에서 다른 일을 한다."),
            ("retire", "first_sleep", "61%는 단계 사이 확률에 기댄 program에서 나온 값이었다. 격자만 넘기면 37%로 다른 학습과 같다. code의 수, 잠, code를 켜는 규칙 가운데 40% 근처의 값을 움직인 것은 없다."),
            ("new", "two_handover", "measure", "transfer는 확률을 넘기는 방식과 격자만 넘기는 방식 둘 다로 잰다",
             "transfer 정확도 하나로는 지름길이 가려진다. 이후 transfer는 단계 사이에 확률을 넘기는 방식과 격자만 넘기는 방식 둘 다로 잰다."),
        ],
    },
    {
        "title": "학습을 빠르게 하고 한 단계짜리 싼 판으로 부품 찾기",
        "date": "2026-11-22",
        "source": "`docs/reports/2026-11-22-fast-search.md`와 `docs/project-overview.md` 5.5절",
        "question": "transformer NEO 하나에 여섯 시간이 걸려서는 아무것도 찾을 수 없다. 어디가 느리고, 어떤 부품이 primitive 여덟을 code 하나씩으로 배우는가.",
        "did": "forward를 torch.compile로 묶고 token 수를 절반으로 줄인 programmer를 만들어 한 스텝을 152ms에서 44ms로 줄였다. 그다음 primitive 하나짜리 학습 쌍만 쓰고 program을 한 단계로 제한한 싼 판에서 조건 80개쯤을 한꺼번에 돌렸다. "
               "programmer, code를 다시 뿌리는 codebook, 가중치 평균, batch, executor 폭과 깊이, commitment, code 차원, code 수를 바꿨다.",
        "diff": {**ONE_STEP, BOOK: "조건에 따라 code 8개에서 32개, 안 쓰는 code를 지금의 programmer 출력으로 다시 뿌리는 vq_restart, commitment 0.25에서 1.0",
                 EXEC: "조건에 따라 폭 128 또는 256"},
        "items": [
            ("table", OVERVIEW, "| 조건 | 재현 | transfer |", "한 단계 싼 판에서 부품을 바꾼 결과, primitive 하나짜리 평가 과제",
             "첫 줄은 데이터와 program 길이만 싼 판이고 부품과 학습 일정은 랩 설정 그대로다. 그 아래는 줄에 적힌 부품을 바꿨다. programmer 교체는 token_pair_transformer, 다시 뿌리기는 vq_restart다."),
        ],
        "result": "랩 programmer는 처음 1,800스텝 동안 code를 하나만 쓴다. programmer를 바꾸면 300스텝부터 배운다. 다시 뿌리기는 code를 전부 쓰게 해 재현을 올리지만 뒤집기를 code 여러 개로 쪼개 transfer를 깎는다. "
                  "같은 데이터 양이면 작은 batch로 오래 도는 쪽이 낫다. code 수는 primitive 여덟에 8개면 다 찾지 못하고, 남는 code가 적을수록 쪼개기가 줄어 transfer가 오른다. 여기서 고른 부품이 이후의 기준 run이 됐다.",
        "insights": [
            ("new", "prog_finds", "code", "랩 programmer는 code를 늦게 찾고, 칸마다 이어 붙인 넓은 programmer가 빨리 찾는다",
             "랩 transformer NEO가 느렸던 이유는 code를 못 찾아서였다. 랩 programmer는 처음 1,800스텝 동안 code를 하나만 쓰고, 지금 칸과 목표 칸을 칸마다 이어 붙인 폭 256 programmer는 300스텝부터 배운다."),
            ("revise", "split_by_input", "남는 code와 다시 뿌리기는 primitive를 입력에 따라 쪼갠다",
             "code를 더 주거나 안 쓰는 code를 다시 뿌리면 재현은 오르고 transfer는 내려간다. 남는 code가 뒤집기 같은 primitive를 입력에 따라 쪼개는 데 쓰이기 때문이다. code 수는 primitive 수보다 조금 많은 정도가 가장 좋았다.",
             "한 단계 싼 판에서 code 수와 다시 뿌리기를 바꿔 본 결과가 잠에서 본 것과 같은 방향이었다."),
        ],
    },
    {
        "title": "찾은 부품을 다섯 단계 모델에 넣기: code가 주소가 된다",
        "date": "2026-11-22",
        "source": "`docs/project-overview.md` 5.6절",
        "question": "싼 판에서 고른 부품이 랩의 다섯 단계 모델도 고치는가.",
        "did": "고른 부품을 단계 사이에 확률을 넘기는 다섯 단계 모델에 넣었다. 그리고 같은 부품으로 격자만 넘기며 학습하는 두 형태를 돌렸다.",
        "diff": {PROG: ONE_STEP[PROG], BOOK: "code 12개 또는 16개, 조건에 따라 code 차원 4",
                 STEPS: "다섯 단계. 확률을 넘기는 형태와, 격자만 넘기는 reencode_hard, neo_visible_state", SCHED: "batch 128, 6,000에서 12,000스텝"},
        "items": [
            ("inline", """| 측정 | 값 |
|---|---|
| 재현, 7,200스텝 | 81% |
| transfer | 13% |
| 6,000스텝 모델을 격자만 넘겨 잰 재현 | 76.3%에서 2.3%로 |
| primitive 하나짜리 과제의 program | 전부 code 둘. 같은 primitive에 쓰는 code 쌍이 색 바꾸기 17가지, 좌우 뒤집기 78가지 |""",
             "확률을 넘기는 다섯 단계 모델에 빠른 부품을 넣은 결과",
             "programmer, batch, code 차원이 랩과 다르고 단계 사이로 넘기는 것은 랩과 같다."),
            ("table", OVERVIEW, "| 모델 | 재현 | transfer | 숨은 길이 1, 2, 3, 4의 transfer |", "격자만 넘기며 학습한 다섯 단계 모델",
             "첫 줄은 랩 설정으로 학습한 모델을 격자만 넘겨 평가한 것이다. 아래 두 줄은 programmer와 학습 일정이 싼 판에서 고른 것이고, 단계 사이로 넘기는 방식이 줄에 적힌 대로다."),
        ],
        "result": "확률을 넘기면 재현은 다섯 배 빨리 오르는데 transfer는 13%다. code 둘이 합쳐져 변환 하나를 가리키는 주소가 된다. "
                  "격자만 넘기며 학습하면 주소는 막히지만 code가 대충 하는 단계와 고치는 단계로 나뉘고 transfer는 랩 설정과 같은 40% 근처다.",
        "insights": [
            ("new", "objective", "code", "예시를 다시 만들기만 하면 채워지는 목표 아래에서 여러 단계를 함께 학습하면 code가 primitive가 아닌 것으로 굳는다",
             "NEO의 학습 목표는 예시를 다시 만들기만 하면 채워지고, 단계마다 격자 위의 연산 하나를 하라는 요구는 어디에도 없다. 여러 단계를 한꺼번에 학습하면 code가 primitive가 아닌 것으로 굳는다. "
             "확률을 넘기면 code 둘이 변환 하나를 가리키는 주소가 되고, 격자만 넘기면 대충 하는 단계와 고치는 단계로 나뉜다. 부품을 재현이 빨리 오르는 쪽으로 고를수록 이 해로 곧장 간다."),
            ("revise", "handover", "단계 사이에 확률을 넘기면 격자에 보이지 않는 정보가 건너가고, code 둘이 주소가 된다",
             "랩 transformer NEO는 program의 단계 사이에 칸마다의 색 확률을 넘기고, 학습된 모델은 격자에 보이지 않는 그 부분으로 정보를 넘긴다. "
             "이 통로가 열린 채 학습하면 첫 단계가 주소의 앞자리를 확률에 적고 둘째 단계가 뒷자리와 합쳐 실행하는 program이 생긴다. 격자만 넘겨 재면 재현이 76%에서 2%로 떨어진다.",
             "빠른 부품을 넣은 다섯 단계 모델에서 primitive 하나짜리 과제의 program이 모두 code 둘이었다."),
        ],
    },
    {
        "title": "한 단계로만 배운 모델은 이어 쓸 수 있다, 그리고 primitive 수 늘리기",
        "date": "2026-11-22",
        "source": "`docs/project-overview.md` 5.7절",
        "question": "여러 단계를 한꺼번에 학습하는 것이 code를 망가뜨린다면, 한 단계로 배운 code는 이어 쓸 수 있는가. primitive가 늘면 어떻게 되는가.",
        "did": "한 단계 모델을 추가 학습 없이 평가 때만 다섯 단계까지 이어 쓰게 했다. programmer가 executor가 낸 격자를 다시 보고 다음 code를 고른다. 되는 한 단계 모델에서 여러 단계 학습을 이어 가는 것도 조건 일곱으로 돌렸다. "
               "그리고 같은 물체 모양으로 primitive를 32개까지 늘린 데이터를 직접 만들어 한 단계 학습을 했다.",
        "diff": {**ONE_STEP, INFER: "평가 때만 다섯 단계까지 이어 쓴다. 단계마다 programmer가 executor가 낸 격자를 보고 다음 code를 고르고, 단계 사이에는 격자만 넘긴다"},
        "items": [
            ("table", OVERVIEW, "| 모델 | 재현 | 레벨 1 transfer | 숨은 길이 1, 2, 3, 4, 5 이상 |", "여러 단계로 학습한 랩 설정과 한 단계로만 학습한 모델, 레벨 1",
             "첫 줄은 랩 설정을 격자만 넘겨 평가한 것이다. 둘째 줄은 이 실험의 공통 설정이다."),
            ("table", OVERVIEW, "| primitive 수 | code 수 | primitive 하나짜리 transfer |", "primitive 수를 늘린 한 단계 학습",
             "공통 설정에 더해 데이터가 이 레포의 생성기로 만든 primitive 묶음이다. 11개 줄만 BabyTheorist의 데이터다. 맨 아래 줄은 executor 폭이 256이다."),
        ],
        "result": "이어 쓴 레벨 1 transfer는 seed 42가 83%이고 seed 여섯 평균은 64%로, 여러 단계로 학습한 랩 설정의 40%보다 높다. 숨은 길이를 읽지 않고 전체 학습 쌍으로 한 단계 학습을 하면 19에서 35%다. "
                  "되는 한 단계 모델에서 여러 단계 학습을 이어 가면 150에서 300스텝 사이에 무너진다. primitive가 11개까지는 거의 다 되고 16개부터 회전, 뒤집기, 전치가 8에서 24%로 빠진다.",
        "insights": [
            ("retire", "easy_only", "빠른 부품으로 primitive 하나짜리 과제만 한 단계 학습하면 오히려 primitive를 잘 찾는다. 앞의 관찰은 랩 programmer가 code를 늦게 찾던 탓이었다."),
            ("new", "one_step", "code", "primitive 하나짜리 과제로 한 단계만 배운 code는 이어 쓸 수 있다",
             "primitive 하나짜리 과제로 program 한 단계만 학습한 모델은 추가 학습 없이 여러 단계로 이어 쓸 수 있다. 레벨 1 transfer가 seed 여섯 평균 64%로, 여러 단계로 학습한 랩 설정의 40%보다 높다. "
             "다만 숨은 길이를 읽어 과제를 고른 진단이고 seed에 따라 31%에서 84%로 갈린다."),
            ("new", "multi_breaks", "code", "되는 모델에 여러 단계 학습을 이어 가면 무너진다",
             "되는 한 단계 모델에서 여러 단계 학습을 이어 가면 수백 스텝 안에 83%에서 6에서 19%로 무너진다. 망가지는 것은 programmer가 code를 고르는 능력이다."),
            ("new", "similar", "code", "primitive가 16개를 넘으면 회전, 뒤집기, 전치 계열이 빠진다",
             "primitive가 11개까지는 거의 다 code 하나씩을 얻지만 16개를 넘으면 회전, 뒤집기, 전치 계열이 8에서 24%로 빠지고 이동은 code의 기준이 칸 수로 바뀐다. 예시 하나로는 위로 3칸과 자기 높이인 3칸만큼 위로가 구별되지 않는다."),
            ("new", "length_decay", "plan", "이어 쓰면 길이가 늘수록 transfer가 줄고 길이 6부터 급히 준다",
             "한 단계 모델을 이어 쓰면 길이 하나마다 transfer가 0.8배쯤으로 줄다가 길이 6부터 빨리 준다."),
            ("revise", "seeds", "seed 하나의 값으로 말하지 않는다",
             "seed마다 차이가 크다. 한 단계 모델의 이어 쓴 transfer를 seed 하나로 83%라고 했다가 여섯 seed 평균 64%로 고쳤다. seed 하나의 값으로 말하지 않는다.",
             "seed를 여섯으로 늘리자 31%에서 84%로 갈렸다."),
        ],
    },
    {
        "title": "program 길이를 점진적으로 늘리기",
        "date": "2026-11-22에서 23",
        "source": "`docs/project-overview.md` 5.8절",
        "question": "program 길이를 1, 2, 4처럼 점진적으로 키워 가며 학습하면 여러 단계 학습이 code를 망가뜨리지 않는가.",
        "did": "한 단계 모델에서 출발해 길이 2까지를 숨은 깊이 2 이하인 pair로 3,000스텝, 이어서 길이 4까지를 깊이 4 이하인 pair로 3,000스텝 학습했다. "
               "대조 둘로 효과가 어디서 오는지 갈랐다. 길이는 2로 줄이되 모든 깊이의 pair를 주는 것과, 길이는 5인데 깊이 2 이하 pair만 주는 것이다.",
        "diff": {**ONE_STEP, DATA: "한 단계 모델에서 출발해, 그때 허락된 길이 이하의 숨은 깊이를 가진 pair만 준다",
                 STEPS: "길이 2, 이어서 길이 4까지. 단계 사이에는 격자만 넘긴다", SCHED: "단계마다 3,000스텝. 학습률 3e-5에 grounding 항을 끈 쪽과 기본 학습률 쪽",
                 INFO: "입력과 출력 격자, 그리고 pair를 거르는 데 숨은 깊이"},
        "items": [
            ("table", OVERVIEW, "| 조건 | 3,000스텝 끝 transfer |", "길이를 점진적으로 늘린 학습과 대조", "공통 설정 그대로이고 줄마다 rollout 길이와 주는 pair의 깊이가 다르다."),
            ("fig", "reports/2026-11-22-scaling/program_length.png", "program 길이를 한 번에 늘릴 때와 점진적으로 늘릴 때", "공통 설정 그대로다."),
        ],
        "result": "seed 42에서는 두 단계 내내 83% 근처를 지켰다. 효과는 rollout 길이가 아니라 지금 허락된 길이로 설명할 수 없는 pair를 빼는 데서 나왔다. seed 43과 44에서는 무너져 세 seed 평균이 63%, 50%, 40%다.",
        "insights": [
            ("revise", "multi_breaks", "여러 단계 학습은 허락된 길이로 설명할 수 없는 pair를 억지로 맞출 때 code를 휘게 한다",
             "되는 한 단계 모델에서 여러 단계 학습을 이어 가면 수백 스텝 안에 무너진다. 허락된 길이로 설명할 수 없는 pair를 빼면 seed 42에서는 83%를 지킨다. 효과는 rollout 길이가 아니라 데이터 제한에서 온다. "
             "다만 출발 code가 덜 깨끗한 seed 43과 44에서는 무너졌고, 깊이를 제한하려면 숨은 깊이를 읽어야 한다.",
             "길이를 줄이고 모든 깊이를 준 대조는 25%로 무너졌고, 길이는 5인데 깊이 2 이하만 준 대조는 85%였다."),
        ],
    },
    {
        "title": "양을 code에 담기: 학습하지 않은 칸 수를 말할 수 있는가",
        "date": "2026-11-22에서 23",
        "source": "`docs/reports/2026-11-22-kind-and-amount.md`와 `docs/project-overview.md` 5.9절",
        "question": "한 칸, 두 칸, 세 칸 이동을 배운 모델이 네 칸 이동을 학습하지 않고도 말할 수 있는가.",
        "did": "네 방향 한 칸에서 여섯 칸 이동 24개의 묶음에서 네 칸 이동 넷을 학습에서 뺐다. 네 갈래를 시험했다. code를 종류와 정수 양으로 부르고 양만큼 회전시켜 만드는 codebook, 양이 더해지게 하는 학습 항, "
               "code 하나와 실행 횟수로 된 program, 그리고 programmer가 종류와 양을 함께 말하고 executor가 단위 code를 양만큼 반복 실행하는 모델이다.",
        "diff": {**ONE_STEP, DATA: "네 방향 한 칸에서 여섯 칸 이동 24개. 네 칸 이동 넷은 학습 파일에서 빼고 평가에만 둔다. 모두 primitive 하나짜리다",
                 BOOK: "갈래에 따라 랩 VQ, code를 종류와 양으로 부르는 vq_orbit, 회전 각을 고정한 Lie algebra codebook",
                 ASSIGN: "갈래에 따라 랩 방식, 양이 더해지게 하는 항을 더한 neo_compose, code 하나를 횟수만큼 실행하는 neo_repeat, programmer가 종류와 양을 고르는 neo_counted",
                 INFER: "갈래에 따라 code 하나를 1번에서 6번 이어 실행하는 탐색을 쓴다"},
        "items": [
            ("table", OVERVIEW, "| 모델 | 학습한 20개, 설명 / transfer |", "codebook에 구조만 넣은 첫 시도, seed 42",
             "첫 줄은 codebook이 랩 VQ다. 둘째 줄은 codebook만 vq_orbit이다. 오른쪽 두 열은 평가 때 code 둘짜리 program이나 code 하나의 반복을 탐색한 값이다."),
            ("table", OVERVIEW, "| seed | 끝 checkpoint의 4칸 transfer | 비고 |", "회전 각을 고정한 Lie algebra codebook의 학습하지 않은 네 칸 이동",
             "codebook이 Lie algebra 형태이고 programmer가 종류와 양을 함께 고르며 executor가 단위 code를 양만큼 반복 실행한다. 대조 줄은 항목을 관계없는 벡터로 둔 것이다."),
            ("fig", "reports/2026-11-22-scaling/shift_family.png", "작은 이동 묶음에서 모델별로 본 결과", "공통 설정 그대로이고 선마다 codebook과 실행 방식이 다르다."),
            ("fig", "reports/2026-11-22-scaling/heldout_curve.png", "Lie algebra codebook이 학습하지 않은 네 칸 이동을 맞히는 비율의 학습 곡선", "codebook이 Lie algebra 형태인 것과 관계없는 벡터인 것의 비교다."),
        ],
        "result": "codebook에 구조만 넣은 첫 시도는 학습하지 않은 양을 내지 못했다. 양이 더해지게 하는 학습 항은 모두 학습을 망가뜨렸다. 양은 실행 횟수에 이미 들어 있었다. code 하나를 여러 번 이어 실행하는 탐색으로는 네 칸을 100% 재현했다. "
                  "programmer가 종류와 횟수를 스스로 고르는 Lie algebra codebook 모델은 학습에서 뺀 네 칸 이동을 92%, 99%, 53% 풀었고 관계없는 벡터로 둔 대조는 0에서 4%였다.",
        "insights": [
            ("new", "amount_exec", "amount", "양이 더해지는 성질은 loss가 아니라 실행 방식으로 보장해야 한다",
             "양이 더해지는 성질은 loss로 가르칠 수 없었고 실행 방식으로 보장해야 했다. 덧셈을 loss로 건 시도는 모두 학습을 망가뜨렸고, code 하나를 여러 번 이어 실행하면 학습하지 않은 네 칸 이동이 100% 재현됐다."),
            ("new", "orbit", "amount", "구조를 넣은 codebook은 본 적 없는 양을 말한다",
             "code를 종류와 양으로 부르고 양만큼 회전시켜 만드는 codebook에서 programmer가 종류와 횟수를 고르게 하면, 학습에서 뺀 네 칸 이동을 seed 셋에서 92%, 99%, 53% 푼다. 항목을 관계없는 벡터로 두면 0에서 4%다."),
            ("new", "shape_vs_function", "exec", "code가 놓인 모양과 executor가 하는 일은 별개다",
             "code 벡터가 놓인 모양과 executor가 그 벡터로 하는 일은 별개다. 이동 칸 수의 code들은 한 평면의 호 위에 순서대로 놓이지만, 그 호를 따라 한 칸 더 간 벡터를 executor에 주면 학습된 code와 cosine이 0.94인데도 정확히 옮기는 비율이 32%다."),
        ],
    },
    {
        "title": "크기 사다리: primitive 8개에서 64개까지",
        "date": "2026-11-22",
        "source": "`docs/reports/2026-11-22-scaling.html`과 `docs/project-overview.md` 5.10절",
        "question": "primitive 수만 늘리면 같은 모델이 어떻게 변하는가. 환경이 커져도 배우는 속도와 닿는 높이가 유지되는가.",
        "did": "primitive 묶음을 8, 16, 24, 32, 52, 64개로 만들어 앞 크기의 묶음이 뒤 크기에 그대로 들어가게 했다. 크기마다 처음부터 같은 24,000스텝을 primitive 하나짜리 pair로 학습했다. "
               "랩 방식의 한 단계 모델, code 하나를 반복 실행하는 모델, Lie algebra codebook 모델을 같은 사다리에 올렸다.",
        "diff": {**ONE_STEP, DATA: "이 레포의 생성기로 만든 primitive 묶음 8, 16, 24, 32, 52, 64개. 숨은 길이를 읽어 primitive 하나짜리 pair만 쓴다"},
        "items": [
            ("table", OVERVIEW, "| 크기 | 더해진 primitive |", "크기마다 더해진 primitive", "데이터의 정의다. 8개가 BabyTheorist의 기본 primitive이고 나머지는 이 레포가 더했다."),
            ("table", OVERVIEW, "| 크기 | 8 | 16 | 24 | 32 | 52 | 64 |", "환경 크기별 학습 batch 정답률과 transfer", "공통 설정 그대로다. 부품과 code를 고르는 방식은 랩과 같다."),
            ("table", OVERVIEW, "| 크기 | 길이 1 | 길이 2 | 길이 3 | 길이 4 |", "숨은 program 길이별 transfer", "공통 설정에 더해, 평가 때 단계마다 격자로 읽어 다음 code를 고르며 이어 쓴다."),
            ("fig", "reports/2026-11-22-scaling/curves_by_size_lab_neo_one_code.png", "랩 방식 한 단계 모델, 환경 크기마다 한 선", "공통 설정 그대로다."),
            ("fig", "reports/2026-11-22-scaling/final_by_size.png", "과제 종류별로 본 끝 성능", "공통 설정 그대로다."),
            ("fig", "reports/2026-11-22-scaling/old_primitives.png", "환경이 커질 때 이미 있던 primitive의 transfer", "공통 설정 그대로다."),
            ("fig", "reports/2026-11-22-scaling/by_depth.png", "숨은 program이 길어질 때", "공통 설정에 더해 평가 때 이어 쓴다."),
            ("fig", "reports/2026-11-22-scaling/steps_to_reach.png", "일정 수준에 닿는 데 걸린 학습량", "공통 설정 그대로다."),
            ("fig", "reports/2026-11-22-scaling/curves_by_model.png", "모델을 나란히 놓은 학습 곡선", "선마다 모델이 다르다. 랩 방식, code 하나를 횟수만큼 실행하는 모델, Lie algebra codebook 모델이다."),
            ("fig", "reports/2026-11-22-scaling/curves_by_size_one_code_x_count.png", "code 하나를 횟수만큼 실행하는 모델, 환경 크기마다 한 선", "program이 code 하나와 실행 횟수로 되어 있다."),
            ("fig", "reports/2026-11-22-scaling/curves_by_size_one_code_x_count_no_gradient_between_applications.png", "같은 모델에서 실행 사이 gradient를 끊은 것", "program이 code 하나와 실행 횟수이고 반복 실행 사이에 gradient를 끊는다."),
            ("fig", "reports/2026-11-22-scaling/curves_by_size_kind_count_fixed_angle_rotation.png", "회전 각을 고정한 Lie algebra codebook 모델, 환경 크기마다 한 선", "codebook이 Lie algebra 형태이고 programmer가 종류와 양을 고른다."),
        ],
        "result": "환경이 커져도 학습 batch 정답률이 50%에 닿는 스텝은 3,500에서 4,750으로 느려지지 않는다. 갈리는 것은 닿는 높이다. 처음 8개의 transfer는 8개 환경에서 96%인데 16개 환경에서 60%다. "
                  "16개에서 들어온 것은 기존 회전과 뒤집기와 비슷한 일을 하는 primitive다. 기존 것과 겹치지 않는 primitive는 98% 근처이고 해를 끼치지 않는다.",
        "insights": [
            ("new", "speed_flat", "grow", "환경이 커져도 배우는 속도는 느려지지 않고 닿는 높이가 갈린다",
             "primitive를 8개에서 64개로 늘려도 학습 batch 정답률이 50%에 닿는 스텝은 느려지지 않는다. 갈리는 것은 닿는 높이다."),
            ("revise", "similar", "환경의 어려움은 primitive 수가 아니라 비슷한 primitive가 함께 있느냐가 정한다",
             "환경의 어려움은 primitive 수가 아니라 비슷한 일을 하는 primitive가 함께 있느냐가 정한다. 처음 8개의 transfer가 8개 환경에서 96%인데 회전과 전치가 더해진 16개 환경에서 60%로 내려간다. 기존 것과 겹치지 않는 primitive는 98% 근처이고 해를 끼치지 않는다.",
             "64개 환경의 transfer가 16개와 24개보다 높았다. 52개와 64개에 더한 것은 대부분 쉬운 이동이다."),
            ("revise", "length_decay", "이어 쓴 transfer는 길이 1의 값을 길이만큼 곱한 값에 가깝다",
             "한 단계 모델을 이어 쓴 transfer는 길이 1의 transfer를 길이만큼 곱한 값에 가깝다. 16개 이상 환경에서는 길이가 하나 늘 때마다 대략 절반이 된다.",
             "여섯 크기에서 길이별 값을 나란히 놓았다."),
        ],
    },
    {
        "title": "커지는 환경: 새 primitive 하나를 배우는 비용",
        "date": "2026-11-22에서 23",
        "source": "`docs/project-overview.md` 5.11절",
        "question": "모델 하나가 학습을 멈추지 않는 동안 환경이 커질 때, 새 primitive 하나를 배우는 비용이 어떻게 변하고 옛것은 유지되는가.",
        "did": "모델 하나를 8, 16, 24, 32, 52, 64개 환경으로 단계마다 6,000스텝씩 이어 학습했다. codebook은 처음부터 128개다. 대조는 환경을 8개로 둔 채 단계만 나눈 사슬이다. "
               "새 primitive의 비용은 새로 들어온 묶음이 transfer 50%에 닿을 때까지 primitive 하나당 본 pair 수로 센다.",
        "diff": {**ONE_STEP, DATA: "학습 도중 환경이 8, 16, 24, 32, 52, 64개로 커진다. 단계마다 그 단계의 전체 묶음으로 6,000스텝",
                 BOOK: "처음부터 code 128개. 모델에 따라 랩 VQ나 Lie algebra codebook",
                 EXEC: "모델에 따라 랩 executor, code 하나를 반복 실행하는 형태, 공유 up-projection과 code별 low-rank를 둔 형태"},
        "items": [
            ("table", OVERVIEW, "| 모델 | 16 | 24 | 32 | 52 | 64 |", "커지는 환경에서 새 primitive의 비용과 처음 8개의 유지",
             "랩 방식 줄은 공통 설정 그대로, 곧 부품과 code를 고르는 방식이 랩과 같고 codebook만 128개다."),
            ("fig", "reports/2026-11-22-scaling/growing.png", "점점 커지는 환경에서의 transfer", "선마다 모델이 다르다. 랩 방식, code 하나 반복, Lie algebra codebook이다."),
            ("fig", "reports/2026-11-22-scaling/growing_speed.png", "새로 들어온 primitive를 배우는 속도", "선마다 모델이 다르다. 랩 방식, code 하나 반복, Lie algebra codebook이다."),
        ],
        "result": "랩 방식에서 새 primitive 하나의 비용은 primitive 하나당 본 pair로 세면 8천에서 1만 6천으로 거의 평평하다. 다만 새 것이 옛것을 밀어낸다. 처음 8개는 64개 단계 끝에 53%로, 같은 시간 8개만 본 대조의 98%보다 크게 낮다. "
                  "Lie algebra codebook 모델은 이 사슬에서 처음 8개가 5에서 39%에 머물렀다.",
        "insights": [
            ("new", "cost_flat", "grow", "새 primitive 하나의 비용은 본 pair 수로 세면 환경 크기에 거의 평평하다",
             "커지는 환경에서 랩 방식이 새 primitive 하나를 배우는 비용은 primitive 하나당 본 pair로 세면 8천에서 1만 6천으로 거의 평평하다."),
            ("new", "pushes_out", "grow", "새 것이 옛것을 밀어낸다",
             "커지는 환경에서 새 primitive가 들어오면 옛것이 밀려난다. 처음 8개의 transfer가 64개 단계 끝에 53%로, 같은 시간 8개만 본 대조의 98%보다 크게 낮다."),
            ("revise", "orbit", "구조를 넣은 codebook은 작은 이동 묶음에서만 본 적 없는 양을 말했다",
             "code를 종류와 양으로 부르는 codebook은 이동 24개의 작은 묶음에서 학습에서 뺀 네 칸 이동을 92%, 99%, 53% 풀었다. 그러나 커지는 환경에 넣으면 처음 8개가 5에서 39%에 머물러 그 결과가 넘어오지 않았다.",
             "같은 모델을 커지는 환경의 사슬에 올렸다."),
        ],
    },
    {
        "title": "회전 계열: 큰 환경에서 무너지는 일곱 개의 원인",
        "date": "2026-11-22에서 23",
        "source": "`docs/project-overview.md` 5.12절",
        "question": "24개 환경에서 회전, 뒤집기, 전치 계열 일곱 개만 12에서 20%로 무너진다. 어느 부품이 못 하는 것인가.",
        "did": "code 벡터의 비슷함, 원리적으로 구별되지 않는 pair의 비율, 생성원만 학습시킨 경우를 차례로 확인했다. primitive마다 정답 code를 주는 진단과 programmer 출력의 선형 분류기로 세 부품이 각각 배울 수 있는지 봤다. "
               "programmer가 이 계열을 어느 code로 보내는지 셌다. 그리고 라벨 없이 푸는 시도 열세 가지와 여덟 배 긴 학습을 돌렸다.",
        "diff": {**ONE_STEP, DATA: "24개 환경. 숨은 길이를 읽어 primitive 하나짜리 pair만 쓴다",
                 ASSIGN: "시도마다 다르다. 랩 방식, 가까운 후보를 실행해 가장 잘 재현한 code만 학습, 굳은 뒤에 시작하는 soft EM, programmer와 executor 번갈아 얼리기, 그리고 primitive마다 정답 code를 주는 진단",
                 EXEC: "시도마다 랩 executor, code마다 executor 하나, 공유 부분과 code별 low-rank, 층 단위로 공유와 code별을 나눈 것",
                 EXTRA: "시도 가운데 잠. 자주 쓰이는데 자기 pair를 60% 미만으로 재현하는 code 옆에 새 code를 놓는다"},
        "items": [
            ("table", OVERVIEW, "| 시도 | 출발 | 무엇을 바꿨나 |", "회전 계열에 시도한 것 전부, 24개 환경 seed 42",
             "첫 줄이 공통 설정, 곧 부품과 code를 고르는 방식이 랩과 같은 한 단계 학습이다. 줄마다 무엇을 바꿨나 열에 적힌 것 하나만 다르다. 맨 아래 줄은 라벨을 쓰는 진단이다."),
            ("table", OVERVIEW, "| 200,000스텝 run, 끝 | rot90 |", "여덟 배 길게 학습한 run", "공통 설정에서 학습 길이만 200,000스텝이다. 둘째 줄은 executor 폭이 두 배다."),
            ("fig", "reports/2026-11-22-scaling/orientation_attempts.png", "회전 계열에 시도한 것 전부", "표와 같은 run들이다."),
            ("fig", "reports/2026-11-22-scaling/executor_split.png", "executor의 어느 층을 code끼리 공유할 것인가", "executor를 공유 층과 code별 층으로 나눈 시도들이다. 나머지는 공통 설정이다."),
        ],
        "result": "primitive마다 정답 code를 주면 같은 공유 executor가 이 계열을 83에서 98%로 해낸다. programmer 출력으로 일곱을 가르는 정확도는 보통 학습에서 38%이고 정답 code로 학습하면 82%다. 세 부품 모두 배울 수 있다. "
                  "programmer는 90도 회전, 270도 회전, 전치, 반대 전치 넷을 같은 code로 보낸다. 넷은 bbox의 가로세로를 바꾼다는 공통점이 있다. 라벨 없이 시도한 열세 가지는 모두 9에서 21%였다. "
                  "여덟 배 길게 학습하면 뒤집기 둘은 풀리지만 넷은 17에서 24%로 남는다. 도중에 executor가 이 계열을 한꺼번에 담지 못한다고 해석했는데, 정답 code 진단을 보고 버렸다.",
        "insights": [
            ("revise", "similar", "큰 환경의 실패는 비슷한 변환이 한 code에 섞인 채 굳는 것이다",
             "큰 환경에서 깨지는 것은 부품이 아니라 정답 없는 code 배정이다. bbox의 가로세로를 바꾸는 넷, 곧 90도와 270도 회전, 전치, 반대 전치가 처음에 한 code로 묶이면 그 code는 넷 가운데 어느 것도 맞추지 못하고, 더 나은 code가 생기지 않아 그대로 굳는다.",
             "programmer가 넷을 같은 code 둘로 보내는 것을 셌고, primitive마다 정답 code를 주면 같은 부품이 83에서 98%를 했다."),
            ("new", "exec_capable", "exec", "executor는 정답 code가 주어지면 회전 계열을 해낸다. 용량 문제가 아니다",
             "executor는 primitive마다 code 하나가 정답으로 주어지면 회전 계열을 83에서 98%, 나머지를 100% 해낸다. 큰 환경의 실패는 executor의 용량 문제가 아니다. 폭을 두 배로 해도 풀리지 않는다."),
            ("new", "no_force", "code", "pair 하나만 보는 학습 신호 안에는 섞인 code를 떼어 놓는 힘이 없다",
             "pair 하나를 재현하는지만 보는 학습 신호 안에는 한 code에 섞인 변환들을 떼어 놓는 힘이 없다. 후보를 실행해 고르기, 잠에서 섞인 code 옆에 새 code 놓기, 번갈아 얼리기, 굳은 뒤의 soft EM을 포함해 라벨 없이 시도한 열세 가지가 모두 9에서 21%였다."),
            ("new", "long_training", "code", "오래 학습하면 일부만 풀린다",
             "24개 환경을 여덟 배인 200,000스텝 학습하면 뒤집기 둘은 78%와 65%로 풀리지만 가로세로를 바꾸는 넷은 17에서 24%로 남는다."),
        ],
    },
]
