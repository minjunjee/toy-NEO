# 인수인계

## 2026-11-20 레포를 만든 직후의 상태

Claude Code가 썼다. claude.ai 채팅의 Claude가 이어서 본다.

이 레포에는 문서만 있다. 코드, 데이터, 실험 결과는 없다. 브랜치는 `code/init`이다. 레포는 처음에 공개였고 사용자가 비공개로 바꾼 뒤에 push했다. 랩 NEO 코드는 다음 차례에 이전 레포의 commit `b30af0d`에서 꺼내 넣는다.

읽는 순서는 `docs/discussion-2026-11-20.md`, `docs/research-plan.md`, `docs/facts.md`다. 앞의 것이 왜 이렇게 잡았는지이고, 가운데가 무엇을 할지이고, 뒤의 것이 무엇에 기대는지다.

### 정해진 것

메인 주제는 evaluation이고 scaling은 그 아래의 일이다. 측정 도구를 먼저 만들지 않고, 더 어려운 상황에서 NEO를 돌려 어떻게 틀리는지 관찰하는 것부터 한다. 무대는 BabyTheorist이고 모델은 랩 NEO 그대로다. 자세한 것은 `docs/decisions.md`에 있다.

### 다음에 할 일

채팅 쪽에서 할 일은 연구 계획을 검토하고 첫 지시서를 쓰는 것이다. 계획에서 Claude Code가 확신하지 못하는 곳은 셋이다.

1. 처음 돌릴 설정값. codebook의 code 수 열둘과 program의 최대 길이 4는 랩의 GridWorld 설정에서 비율만 따온 것이다.
2. object의 수만 바꾼 데이터를 생성 옵션으로 만들 수 있는지. 설명서에는 옵션이 있다고 적혀 있고 돌려 보지는 않았다.
3. 길이 방향을 정해진 절차대로 이어서 배우게 할지, 레벨마다 처음부터 따로 배우게 할지. 계획에는 정해진 절차를 따른다고 적었다.

Claude Code 쪽에서 다음에 할 일은 `docs/research-plan.md`의 준비 단계다. 랩 NEO 코드를 들여오고, BabyTheorist 파일을 읽는 dataset 부품을 쓰고, 데이터를 만들고, 학습 시간을 잰다.

### 사용자가 정해야 하는 것

1. 랩이 논문에서 NEO를 BabyTheorist에 돌릴 때 쓴 학습 코드와 설정을 받을 수 있는가.
2. Agent가 뜻하는 것이 배운 theory로 목표에 닿는 것인가, 무엇을 실험할지 스스로 고르는 것인가.
3. Claude Code가 낸 세 가지 수정을 모두 받는 것인가. BabyTheorist를 쓰는 것, 지금 NEO를 그대로 돌리는 것, 한 방향씩 키우는 것이다.

### 이전 레포와의 관계

이전 레포 `minjunjee/neo-vocabulary-restructuring`은 비공개이고 그대로 둔다. 마지막 상태는 브랜치 `code/two-properties-round-1`에 있다. 거기서 넘어오는 것은 `docs/facts.md` 4절의 가설 셋뿐이다. BabyTheorist 저장소는 서버의 `/home/minjun_dev/neo-vocabulary-restructuring/runs/external/babytheorist`에 받아 두었다.
