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
- 18:35 MLP의 폭을 키운 랩 NEO 여섯 개가 끝남. 전부 0이다. state를 넘기는 방식을 바꿔 다시 평가하면 예시 쌍 재현이 0에서 5% 사이다.
- 18:18 이동 넷만 쓰는 데이터 `single_translate`를 만들고 랩 NEO를 학습함. 19:10에 끝남. 여기서도 0에 가깝다.
- 19:33 transformer NEO seed 42가 12,000스텝을 마침. 예시 쌍 재현 30.3%, transfer 정확도 7.1%. 길이가 1인 과제에서 재현 45.5%, transfer 14.1%. 마지막 3,000스텝 동안 재현이 10%에서 30%로 올랐다.
- 19:33 밤사이 돌 줄을 세움. GPU 4번에 primitive 하나짜리 과제만 쓰는 진단, 이동 넷만 쓰는 진단, seed 44. GPU 5번에 seed 43과 36,000스텝짜리 긴 학습.
- 19:35 보고서 `docs/reports/2026-11-20-first-numbers.md`를 쓰고 `code/baseline-observation`을 push함.
- 19:50 그림에서 본 설명을 숫자로 확인함. transformer NEO seed 42의 primitive 하나짜리 이동 과제에서 새 object의 크기가 예시와 같으면 transfer 17.3%, 다르면 9.2%다. 크기 차이가 절반을 설명하지만 주된 원인은 아니다. 보고서의 문장을 고쳤다.
- 21:08 transformer NEO seed 43이 끝남. 재현 63.0%, transfer 24.5%로 seed 42의 세 배가 넘는다. code 대 primitive 표에서 seed 43은 위아래 이동을 각각 code 하나에 담았다.
- 21:35 primitive 하나짜리 과제만으로 학습한 transformer NEO가 끝남. primitive 둘만 배우고 23%에서 멈췄다. 막히는 곳이 이어 붙이기가 아니라 primitive를 찾는 일이라는 가설을 보고서 5.2절에 적음.
- 22:50 체크포인트 열 개를 따라가며 primitive마다 code가 언제 생기는지를 잼. primitive는 차례로 code를 얻고, code 하나로 굳기 전에 여러 code로 설명하는 단계를 거친다. 21:35에 적은 가설, 곧 초반에 갈리고 안 바뀐다는 것은 틀렸다. 측정도 고쳤다. code 하나를 고른 것이 아니라 code 하나로 출력을 정확히 만든 것만 센다.
- 22:17 같은 설정의 seed 45부터 48을 밤사이 줄에 더함. 사용자가 연속 벡터 기준선은 돌리지 말고 아직 모르는 것을 돌리라고 함.
- 23:38 이동 넷만 쓰는 데이터로 학습한 transformer NEO가 끝남. 재현 24.8%, transfer 2.3%. 설명은 하는데 옮겨 가지 않는다.
- 23:40 36,000스텝 일정의 seed 42가 10,800스텝에서 재현 70.4%, transfer 27.6%. 같은 seed의 12,000스텝 일정은 같은 시점에 25.6%와 6.9%였다. 학습률 일정의 차이다.

## 2026-11-21

- 01:40 transformer NEO seed 44가 끝남. 재현 62.5%, transfer 17.2%. seed 셋의 평균은 재현 51.9%, transfer 16.3%다. 색 바꾸기, 이동, 뒤집기, 회전의 순서로 code를 얻는 것은 seed 44에서도 같다.
- 03:20 사용자가 잠을 넣어 보자고 하고, 학습 곡선을 계산량 기준으로 그리자고 함. `scripts/flops.py`와 `scripts/babytheorist/curves_by_flops.py`를 만듦.
- 03:25 primitive 하나짜리 과제를 예시 쌍과 새 입력 쌍이 같은 code로 설명되는지로 나눔. 같은 code 하나가 양쪽을 재현하면 transfer가 100%다. programmer가 놓친 code는 거의 없다.
- 03:35 36,000스텝 학습이 끝남. 재현 89.8%, transfer 39.8%.
- 03:46 나뉜 code를 합치고 저장분을 다시 실행해 고치는 잠을 학습이 끝난 모델에 적용함. 고치는 방법 둘 모두 후보 여덟이 떨어졌다. 색 바꾸기의 두 code는 벡터만 맞추면 98.1%까지 돌아오지만 통과선에 못 미친다.
- 03:47 code를 8개로 준 36,000스텝 학습을 GPU 5번에서 시작함. 10개로 준 학습은 GPU 4번에서 seed 46 뒤에 돈다.
