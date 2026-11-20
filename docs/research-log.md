# 연구 로그

시각은 KST다. 한 일과 나온 것을 이어 적는다.

## 2026-11-20

- 15:40쯤 사용자가 연구를 scaling과 evaluation의 두 축으로 다시 짜자고 제안함. 논의는 `docs/discussion-2026-11-20.md`에 있다.
- 16:10쯤 이전 레포에서 랩 NEO 코드를 읽음. 평가가 둘이고 정식 추론이 스텝마다 격자로 풀었다가 다시 읽는다는 것을 확인함. `docs/facts.md` 1절.
- 16:15 BabyTheorist 공개 저장소의 설명서와 코드를 읽음. 모드, primitive, 레벨과 program 길이의 관계, 파일 형식, 정해진 절차를 확인함. `docs/facts.md` 2절.
- 16:20 `minjunjee/toy-NEO`가 로그인 없이 열리는 공개 레포인 것을 확인함. 랩 NEO 코드를 넣지 않기로 하고 사용자에게 알림.
- 16:22 BabyTheorist 생성기를 작게 돌림. 학습 과제 2,000개에 `single-A`가 10초, `multi-A`가 13초. `single-A` 레벨 1의 program 길이는 1이 592개, 2가 1,031개, 5 이상이 110개였다.
- 16:24 로컬에 레포를 만들고 문서 일곱을 씀. `CLAUDE.md`, `README.md`, `docs/` 아래의 인수인계, 연구 계획, 논의 기록, 확인한 사실, 정한 것이다. 코드는 없다.
- 16:38 사용자가 레포를 비공개로 바꿈. 로그인 없이 열리지 않는 것을 확인한 뒤 `code/init`을 push함.
- 16:43 사용자가 NEO 모델 자체의 autoresearch와 ARC-AGI-1, 2에서의 실행을 제안함. 랩의 `ahn-ml/autoresearch-neo` 설명서에서 GridWorld 결과 표를 확인함. 논의를 `docs/discussion-2026-11-20-autoresearch-and-arc.md`에 적음. 연구 계획은 사용자의 답을 받은 뒤에 고친다.
- 17:01 사용자가 숫자를 뽑으면서 진행하라고 함. 브랜치 `code/baseline-observation`을 만들고 랩 코드를 들여옴. 랩 테스트 15개가 통과한다.
- 17:03 BabyTheorist 데이터를 만듦. `single-A` 레벨 1, 2, 3과 `single-B` 레벨 2, 3, object만 둘과 셋으로 바꾼 `single-A` 레벨 1, `multi-A` 레벨 1이다. 작업자 16개로 250,000개가 1분 안에 끝난다. 파일은 `data/babytheorist/`에 있다.
- 17:07 state encoder와 state decoder의 사전학습을 latent 크기 다섯과 칸마다 token 방식으로 시작함. 17:37에 끝남. 512차원은 발산해서 17:20에 멈췄다.
- 17:13 ARC 천장 계산이 끝남. 별도 작업으로 맡겼고 Claude Code가 검증 스크립트와 본 스크립트를 다시 돌려 같은 값을 확인했다.
- 17:20 transformer NEO seed 42를 GPU 4번에서 12,000스텝으로 시작함.
- 17:37 랩 NEO를 32차원과 256차원 state로 seed 셋씩 시작함. 18:05에 끝남. 전부 transfer 정확도가 0에 가깝다.
- 18:05 MLP의 폭을 state보다 넓게 키운 변형 둘을 seed 셋씩 시작함. 128차원 state에 폭 512, 256차원 state에 폭 1,024다.
