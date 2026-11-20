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
