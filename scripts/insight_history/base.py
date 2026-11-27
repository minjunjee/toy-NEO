"""The lab NEO setting every experiment is compared against, and the topics the insights are grouped under."""

DATA = "데이터"
CODEC = "state encoder와 state decoder"
PROG = "programmer"
BOOK = "codebook"
EXEC = "executor"
STEPS = "program 길이와 단계 사이로 넘기는 것"
ASSIGN = "code를 고르고 학습하는 방식"
SCHED = "학습 일정"
INFER = "평가 때의 추론"
INFO = "학습이 읽는 정보"
EXTRA = "랩 NEO에 없는 단계"

# the lab transformer NEO of GeneralizedL2T, moved to BabyTheorist (docs/lab-code-README.md, docs/project-overview.md 4)
LAB = {
    DATA: "BabyTheorist single-A 레벨 1의 학습 파일. pair 하나씩이고 70%가 primitive 여럿짜리다",
    CODEC: "칸마다 128차원 token 하나. 미리 학습해 얼린다",
    PROG: "지금 칸 token과 목표 칸 token을 따로 넣는 6층 transformer, 폭 128",
    BOOK: "랩 VQ. cosine으로 가장 가까운 code를 고른다. code 16개, 16차원, commitment 0.25",
    EXEC: "4층 AdaLN-Zero transformer, 폭 128. code는 층마다 scale, shift, gate를 만드는 조절 신호다",
    STEPS: "최대 다섯 단계. 단계 사이에 칸마다의 색 확률을 넘긴다",
    ASSIGN: "programmer가 pair 하나를 보고 낸 출력에 가장 가까운 code를 쓴다. 길이는 길이마다 0.95를 곱한 cross-entropy가 가장 작은 것으로 고르고, 그 길이의 cross-entropy, grounding 항 0.1, codebook loss로 모든 부품을 함께 학습한다",
    SCHED: "학습률 3e-4, batch 256 episode, 78,200스텝, 가중치 평균 0.999",
    INFER: "programmer가 support pair 하나를 보고 고른 code 열을 query에 그대로 실행한다. 탐색하지 않는다",
    INFO: "입력과 출력 격자만 읽는다",
    EXTRA: "없다",
}

TOPICS = [
    ("state", "state 표현과 단계 사이로 넘기는 것"),
    ("code", "code 형성: programmer와 codebook이 primitive마다 code를 세우는 일"),
    ("exec", "executor: code가 어디서나 같은 변환인가"),
    ("plan", "여러 단계 program: programmer의 계획과 평가 때의 탐색"),
    ("amount", "양과 구조를 넣은 codebook"),
    ("grow", "환경이 커질 때와 아는 것 위에 쌓기"),
    ("memory", "빠른 기억과 잠"),
    ("measure", "재는 방법"),
]

# settings that recur; each is the difference from LAB
ONE_STEP = {
    DATA: "primitive 하나짜리 학습 pair만 쓴다. 숨은 길이를 읽어 고른다",
    PROG: "지금 칸과 목표 칸을 칸마다 이어 붙인 token을 넣는 token_pair_transformer, 폭 256",
    BOOK: "랩 VQ 그대로이고 code 수만 primitive 수의 두 배",
    STEPS: "program 한 단계",
    SCHED: "batch 128 pair, 24,000스텝, 가중치 평균 끔, programmer 학습률은 0.25배, forward를 torch.compile로 묶음",
    INFO: "입력과 출력 격자, 그리고 pair를 고르는 데 숨은 길이",
}
TWO_PAIR_EM = {
    **ONE_STEP,
    DATA: "이 레포의 생성기로 만든 primitive 묶음. 같은 program을 다른 물체 둘에 적용한 pair 둘이 한 과제이고 primitive 하나짜리만 쓴다",
    ASSIGN: "hard EM. 과제마다 codebook의 모든 code를 두 pair에 실행해 오차의 합이 가장 작은 code를 고르고, executor는 그 code로 두 pair를 학습하고 programmer는 그 code를 맞히게 학습한다. 아무도 고르지 않는 code는 오차를 깎아 주는 보너스로 구제한다",
    INFO: "입력과 출력 격자, 어느 pair 둘이 같은 규칙인지, 그리고 한 단계 과제만 남기는 데 숨은 길이",
}
PRICE = {
    **TWO_PAIR_EM,
    ASSIGN: TWO_PAIR_EM[ASSIGN] + ". 그리고 학습 초반에 최근에 많이 쓰인 code일수록 오차에 가격을 더한다. 무게 2 nats를 3,000스텝 유지하고 8,000스텝에 0으로 내린다",
}
WIDER = {
    **PRICE,
    DATA: "가격 run과 같은 두 pair 과제인데, 학습 pair의 물체가 먼저 무작위 primitive를 0에서 4개 거친 뒤 과제의 primitive를 받는다",
}
