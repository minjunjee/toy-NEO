# 인수인계

## 2026-11-23 22:30, 탐색, 꿈 학습, 두 pair 학습 뒤의 상태

Claude Code가 썼다. 23일 오후 사용자와 정한 세 실험과 17:25부터 자정까지의 자율 연구다. 전체는 `docs/project-overview.md`와 그림이 들어간 `docs/project-overview.html`의 5.13절부터 5.18절에 있다.

- support로 검증하는 탐색. 한 단계 모델 seed 여섯을 다시 학습하지 않고, programmer의 선호를 누적한 점수로 상위 몇 갈래를 남기며 실행하고 둘째 support로 거른다. 폭 16에서 길이 8 transfer가 seed 평균 7%에서 22%다. 도구는 `scripts/babytheorist/support_search.py`.
- 정답 program을 primitive별 code로 적어 실행하면 code가 깨끗한 seed는 길이 8에서 92%에서 97%다. 긴 program의 병목은 executor가 아니라 programmer의 계획이다. 라벨을 쓰는 진단이다.
- 꿈 학습. programmer가 얼린 executor로 만든 1에서 8단계 program의 중간 격자마다 다음 code를 배운다. 폭 16의 길이 8이 seed 평균 22%에서 38%. greedy의 길이 1은 85%에서 76%로 손해. method `neo_dream`.
- 처음부터 하는 hard EM과 soft EM은 pair 하나로는 회전 계열을 풀지 못했다. 미리 정한 기준 60%에 넷 21%와 25%. hard EM은 code를 입력별 전문가로 쪼갠다.
- 같은 program의 pair 둘을 한 과제로 주고 두 pair 오차의 합으로 code를 고르는 hard EM은 넷이 seed 평균 56%, seed 42는 96%로 완전히 풀렸다. 32개 환경 seed 42는 59%. 숨은 program은 쓰지 않지만 pair의 묶음을 쓴다. seed 45와 46이 23시 반쯤 끝난다.
- 사용자가 정할 것. pair의 묶음을 쓰는 것을 받아들일지. 받아들이면 다음은 처음에 굳은 seed를 잠에서 두 pair 기준으로 다시 여는 것과, 묶음을 데이터에서 스스로 찾는 것이다.

## 2026-11-23 05:35, soft EM과 200,000스텝 학습 뒤의 상태

Claude Code가 썼다. 사람이 볼 정리는 minjunjee.com/autoresearch_report의 덱이고, 생성기는 사이트 저장소의 `tools/slides-autoresearch-report`, 숫자 스냅샷은 `docs/reports/2026-11-22-scaling/deck_data.json`이다. 도는 실험은 없다.

- 같은 24개 환경을 원래의 8배인 200,000스텝 학습하면 랩 executor의 회전 계열 평균이 17%에서 36%로 오른다. 두 뒤집기는 78%와 65%로 풀리고, 가로세로를 바꾸는 넷은 17%에서 24%로 남는다. executor 폭을 두 배로 하면 22%다. 학습률 일정도 같이 늘렸으므로 원인을 가르려면 일정을 고정한 run이 하나 더 필요하다.
- code를 숨은 변수로 두고 code 48개를 모두 실행해 재현 오차로 책임 비율을 정하는 soft EM을 랩 run의 끝에서 12,000스텝 이어 하면 회전 계열 평균이 20%와 21%로 대조 16%보다 조금 높다. programmer와 executor를 번갈아 얼리면 16%로 대조와 같다. 정답 없이 섞인 code를 푸는 방법은 아직 없다.
- 사용자가 정할 것. 넷이 묶인 덩어리를 풀 다음 방법. 같은 code가 다른 입력에도 같은 변환을 하는지를 보는 기준, 또는 일정을 고정한 긴 run으로 오래 학습한 효과부터 확인하는 것.

## 2026-11-23 04:00, 확장성과 양, 그리고 회전 계열을 따라간 밤 뒤의 상태

Claude Code가 썼다. 그림과 읽는 법은 `docs/reports/2026-11-22-scaling.html`에 있고, 같은 페이지가 claude.ai에 비공개 artifact로 올라가 있다. 양을 code에 담는 실험의 글은 `docs/reports/2026-11-22-kind-and-amount.md`다. 숫자의 원본은 각 run의 `train.jsonl`과 `docs/reports/2026-11-22-scaling/*.json`이다. 도는 실험은 없다.

- 사용자가 정한 방향. 확장성을 가장 먼저 본다. 연산량은 보지 않고, 환경이 커질 때 적응이 로그처럼 쉬운지를 모든 크기에서 그래프로 본다.
- Lie algebra codebook. 종류마다 회전 평면을 배우고 회전 각을 고정한 codebook에서 programmer가 종류와 양을 고르고 executor가 단위 code를 양만큼 실행하면, 학습에서 뺀 4칸 이동을 끝 checkpoint에서 seed 42, 43, 44가 92%, 99%, 53%로 말한다. seed 44는 21,600스텝에 76%였다가 끝에 학습한 이동까지 57%로 떨어져 불안정하다. 항목을 관계없는 벡터로 두면 세 seed 모두 0%에서 4%다. 앞서 seed 44를 94%로 적은 것은 학습한 이동까지 합친 가벼운 평가 숫자를 잘못 읽은 것이다. 작은 이동 묶음에서만 본 결과다.
- 커지는 환경. 랩 NEO가 새 primitive 하나를 반쯤 배우는 데 드는 예시는 16개에서 64개 환경까지 8천에서 1만 6천 쌍으로 평평하다. 처음 8개는 32개 환경까지 버티다가 52개와 64개에서 내려간다. 반복으로 이어 쓰는 모델은 큰 환경에서 새 것이 싸지지만 그 묶음이 반복에 유리하게 생겼다.
- program 길이를 점진적으로 늘리는 것. seed 42에서는 83%를 지켰지만 43과 44에서는 12%와 25%로 무너졌다. 효과는 rollout 길이가 아니라 지금 길이로 설명할 수 없는 pair를 빼는 데서 나온다.
- 회전 계열. 24개 환경에서 일곱 개가 15%에서 20%이고, programmer, executor, code 고르는 방식, 잠에서 code 나누기 열한 가지를 바꿔도 그대로다. code를 정답으로 정해 주면 같은 부품으로 83%에서 98%다. 정답 없는 code 배정이 비슷한 변환끼리 섞여 굳는 것이 원인이다.
- 새로 만든 부품. `vq_orbit`, `neo_compose`, `neo_repeat`, `neo_counted`, `per_code` executor, `lora_code` executor, `neo_expert`, `neo_compete`, `neo_oracle`, 데이터 `babytheorist_pairs_labeled`. 도구는 `scaling_figures.py`, `build_scaling_page.py`, `scale_table.py`, `amount_table.py`, `heldout_curve.py`, `split_executor_checkpoint.py`, `split_mixed_codes.py`, `sleep_cycles.py`.
- 사용자가 정할 것. 섞인 code를 푸는 다음 방법. 지금 잠의 기준은 pair 하나를 재현하는지만 보는데, 같은 code가 다른 입력에도 같은 일을 하는지를 봐야 한다. 그리고 게임 같은 play 환경으로 목표를 옮길지.

## 2026-11-22 14:05, 관찰 한 바퀴를 마친 뒤의 상태

Claude Code가 썼다. 근거는 `docs/reports/2026-11-22-fast-search.md`의 6절부터 14절이고, 사람이 읽을 정리는 `docs/reports/2026-11-22-summary.html`이다. 도는 실험은 없다.

- primitive 하나짜리 학습 쌍으로 program 한 단계만 학습한 모델이, 추가 학습 없이 다섯 단계까지 이어 쓰면 레벨 1의 transfer가 seed 여섯 평균 64%다. 31%에서 84%로 갈린다. 같은 부품으로 여러 단계를 학습하면 seed 셋 평균이 code 16개에서 32%, 12개에서 45%다. 랩 설정은 36,000스텝에 40%다.
- 한 단계 모델에서 시작해 여러 단계 학습을 이어 가면 transfer가 83%에서 내려간다. 학습률 3e-4나 grounding 항이 켜진 조건에서는 300스텝 안에 무너져 6%에서 19%로 끝나고, 학습률 3e-5에 grounding 항을 끄면 2,400스텝에 50%다.
- 길이를 읽지 않고 전체 학습 쌍으로 한 단계 학습을 하면 이어 쓴 transfer가 19%에서 35%다. 한 단계 학습의 결과는 벤치마크 점수가 아니다.
- primitive를 늘리면 11개까지는 primitive 하나짜리의 transfer가 98%이고 16개부터 59%에서 85%다. 회전과 뒤집기 계열이 빠지고, 24개에서는 위로 가는 code가 칸 수로 나뉜다. program의 길이는 5까지 43%에서 46%, 8에서 10%에서 17%다.
- 사용자가 정할 것. 고치는 단계로 넘어갈지. 후보는 보고서 14절에 적었다. 랩이 BabyTheorist에 쓴 모델 코드는 아직 받지 못했다. `learning-to-theorize`에는 BabyTheorist가 없다.
- 도구. 여러 단계 과제로 이어 쓰는 평가는 `scripts/babytheorist/observe_rollout.py`에 `--rollout hard --max-steps 5`를 준 것이다. 한 단계 모델에서 시작하는 학습은 method `neo_warm`이다. seed와 조건의 config는 `make_configs.py`에 모두 있다.

## 2026-11-22 아침, 부품을 빠르게 찾은 뒤의 상태

Claude Code가 썼다. 근거는 `docs/reports/2026-11-22-fast-search.md`다.

- 사용자가 22일 새벽에 더 정한 것. 부수적인 부품과 설정은 도움 없이 직접 범위를 잡아 찾는다. 주제는 NEO의 아키텍처다. 실험은 먼저 최대한 가속하고 싼 판을 만들어 여러 조건을 한꺼번에 돌린다. 디스크를 계속 확인한다. `/home`이 있는 SSD가 58GB, `/data`가 437GB 남아 있다.
- 속도. 학습 한 스텝이 152ms에서 44ms가 됐다. `train.compile: parts`와 새 programmer `token_pair_transformer`다. 재는 도구는 `scripts/bench_step.py`다.
- 빠른 판. primitive 하나짜리 학습 쌍과 program 한 단계로 줄인 학습이고 하나에 5분이다. config는 `make_configs.py`의 `one_step`과 묶음 함수들, 띄우기는 `scripts/run_queue.py`, 표는 `scripts/babytheorist/onestep_table.py`다. `run_queue.py`는 GPU 메모리와 디스크의 남은 양을 보고 시작하고, 묶음 실험은 체크포인트를 하나만 남긴다. 지울 때는 `scripts/prune_checkpoints.py`를 쓴다.
- 빠른 판에서 나온 것. 재현은 99%, transfer는 92%까지 간다. 느렸던 이유는 code를 못 찾아서였다. 안 쓰는 code를 다시 뿌리는 `vq_restart`는 재현을 올리지만 primitive를 여러 code로 쪼개서 transfer를 깎는다. 작은 batch로 자주 갱신하는 쪽이 낫다. 쪼개지는 것은 뒤집기와 회전이다.
- 본 모델에서 나온 것. 같은 부품을 다섯 단계짜리 모델에 쓰면 6,000스텝에 재현 76%, transfer 12%이고, 격자만 넘기면 재현이 2%다. 모델이 code 두 개를 변환 하나의 주소로 쓰고 그 주소를 단계 사이의 확률로 넘긴다. NEO의 학습 목표에는 단계마다 격자 위의 연산 하나를 하라는 요구가 없다.
- 지금 도는 것. 격자만 넘기며 학습하는 본 모델 둘, 빠른 판 두 묶음 33개다. primitive를 8개에서 32개까지 늘린 데이터는 `data/babytheorist/many_p{8,16,24,32}`에 있고 `scripts/babytheorist/generate_many_primitives.py`로 만든다. 진행은 `runs/queue/runner_fastneo1.log`와 `runner_onestep5.log`에 적힌다.
- 다음에 할 일. 격자만 넘기며 학습한 본 모델의 재현과 transfer를 읽는다. primitive 수를 늘린 빠른 판에서 code 하나로 옮겨 가는 primitive의 수가 어떻게 변하는지 읽는다. 긴 program 쪽은 레벨 5와 7의 데이터가 `data/babytheorist/single_a/`에 준비되어 있다.

## 2026-11-22 새벽, 사용자가 방향을 정한 뒤의 상태

Claude Code가 썼다. 사용자가 2026-11-22 00:20에 정한 것이 지금의 방향이다.

- 지금은 고치지 않는다. primitive의 수와 program의 길이를 크게 늘려서 NEO가 못하는 것을 보이고, 왜 못하는지를 분석한다. 고치는 것은 그 뒤에 그 분석을 써서 한다. 잠, code를 켜는 규칙, `neo_visible_state` 같은 고치는 쪽 실험은 더 늘리지 않는다.
- 한 번에 하나씩 늘린다. primitive 쪽은 4, 8, 11개와 그 이상이고 길이는 레벨 1 그대로다. 길이 쪽은 primitive 8개에 레벨 3, 5, 7이다. 모델은 지금의 transformer NEO 그대로다.
- 랩이 BabyTheorist에 쓴 모델과 설정은 아직 없다. `autoresearch-neo`의 README가 가리키는 BabyTheorist commit `e51a7d1`이 든 레포의 링크를 사용자에게 요청했다. 받으면 그 설정으로 논문 숫자가 나오는지부터 본다.
- 정리 문서는 `docs/reports/2026-11-21-summary.html`이다. 만드는 스크립트는 `scripts/babytheorist/build_summary_page.py`다.

## 2026-11-21 밤, 단계 사이로 넘어가는 정보를 찾은 뒤의 상태

Claude Code가 썼다. 보고서의 5.11절이 근거다. 아래 저녁 상태의 내용 가운데 "잠 한 번만 61%"는 이것으로 고쳐 읽는다.

- 랩의 transformer NEO 설정은 program의 한 단계가 다음 단계에 칸마다의 색 확률을 넘긴다. 학습된 모델은 격자에 보이지 않는 그 부분으로 정보를 넘긴다. 잠을 넣은 모델에서 같은 code는 깨끗한 격자를 받으면 위아래로 뒤집고, 자기가 낸 확률을 받으면 다른 일을 한다.
- 격자만 넘기는 방식으로 재면 긴 학습 다섯의 transfer가 37%에서 43%다. 잠을 넣은 학습의 61%는 37%가 된다. code의 수, 잠, code를 켜는 규칙 가운데 이 값을 움직인 것은 없다.
- transfer는 앞으로 두 방식 모두로 잰다. `scripts/babytheorist/observe_rollout.py`와 `program_table.py`가 도구다.
- 상한을 둔 잠은 seed 43에서 기준 학습보다 transfer가 10%p 낮아 21,600스텝에서 멈췄다. 보고서 5.12절이다. 설명하지 못하는 쌍에 code를 새로 주는 것 자체는 지금까지 transfer를 올리지 못했다.
- 지금 도는 것. GPU 4번은 새 부품 `neo_visible_state`의 code 8개 seed 42이고 22일 02:20쯤 끝나며 seed 43이 이어진다. GPU 5번은 랩의 `reencode_hard` 학습 seed 42이고 04:20쯤 끝나며 seed 43이 이어진다. 진행은 `runs/queue/runner_gpu4d.log`와 `runner_gpu5f.log`에 적힌다.
- 다음에 할 일. 통로를 막고 학습한 두 모델이 여러 단계짜리 program을 배우는지, transfer가 40%를 넘는지 본다. 넘지 못하면 남는 문제는 5.10절의 것이다. 뒤집기와 회전이 어떤 입력에서도 통하는 code 하나를 얻지 못한다.

## 2026-11-21 저녁, 긴 학습 다섯을 나란히 본 뒤의 상태

Claude Code가 썼다. 브랜치는 `code/baseline-observation`이다. 보고서의 5.10절이 이 상태의 근거다.

- 36,000스텝을 돈 학습 다섯 가운데 넷이 transfer 40%에서 44%로 끝났다. code 16개, code 8개의 seed 둘, code 8개로 시작해 잠마다 하나씩 켠 것이다. 14,400스텝에 잠을 한 번 넣은 seed 42만 61%다. 그 61%는 아직 하나뿐인 값이다.
- 오전에 돌린 대조 둘은 끝났다. 합치기만 한 것은 44.5%, 비운 code를 아무 방향에나 둔 것은 47.6%로, 효과는 비운 code를 설명하지 못하는 쌍에 배정한 데서 났다.
- 설명하지 못하는 쌍이 남았다는 것만으로 잠마다 code를 켜면 재현은 68%에서 80%로 오르고 transfer는 그대로다. 새 code가 이미 code를 가진 primitive를 입력에 따라 나누는 데 쓰인다.
- seed 43은 겹친 code 없이도 같은 자리에서 멈춘다. 뒤집기 둘과 회전이 code 하나를 얻지 못하고 여러 code를 이은 program으로 설명되며, 그 program은 옮겨 가지 않는다.
- primitive 둘짜리 과제의 transfer는 하나짜리 과제의 transfer를 제곱한 값과 거의 같다. 전체 transfer를 정하는 것은 어떤 입력에서도 통하는 code 하나를 가진 primitive의 수로 보인다.
- 지금 도는 것. GPU 4번은 seed 44의 code 8개 학습이고 그 뒤에 seed 42의 학습이 이어진다. 켤 수 있는 code를 9개로 막고 3,600스텝마다 잠을 넣는 조건이다. GPU 5번은 같은 조건의 seed 43이고 seed 44가 이어진다. 22일 06:30쯤 모두 끝난다. 진행은 `runs/queue/runner_gpu4c.log`와 `runner_gpu5e.log`에 적힌다.
- 다음에 할 일. 두 조건의 seed 셋 평균을 비교한다. 잠이 seed 43 같은 경우에 도움이 되지 않으면, 여러 code를 이은 program으로 설명된 primitive에 code 하나를 주는 방법을 따로 찾아야 한다. 회전은 어떤 설정에서도 못 배웠다.

## 2026-11-21 오전, 잠을 한 번 넣어 본 뒤의 상태

Claude Code가 썼다. 브랜치는 `code/baseline-observation`이다. `docs/reports/2026-11-20-first-numbers.md`의 5.5절부터 5.9절이 밤사이에 더해진 것이다.

- transformer NEO를 36,000스텝까지 학습하면 재현 90%, transfer 40%에서 멈춘다. transfer는 같은 code 하나가 예시 쌍과 새 입력 쌍을 모두 재현할 때만 된다는 것을 쟀다.
- code를 8개로 줄이면 재현 68%, transfer 44%다. 거기에 14,400스텝에서 잠을 한 번 넣으면 재현 77%, transfer 61%다. 잠은 겹친 code 둘을 합쳐 하나를 비우고, 비운 code를 설명하지 못하는 학습 쌍에 배정한다. 든 계산은 학습 24스텝 분량이다. seed 하나의 결과다.
- 사용자가 2026-11-21에 정한 것은 둘이다. 잠을 넣어 본다. 학습 곡선의 가로축을 학습에 든 계산으로 그린다. 연속 벡터 기준선은 돌리지 않는다.
- 지금 도는 것은 대조 둘이다. 합치기만 한 것과 비운 code를 아무 방향에나 둔 것이다. 14시쯤 끝난다. 진행은 `runs/queue/runner_gpu4b.log`와 `runner_gpu5b.log`에 적힌다.
- 다음에 할 일. 대조 둘의 결과로 잠의 어느 부분이 효과를 냈는지 가른다. 다른 seed에서 다시 나오는지 본다. 잠을 한 번이 아니라 학습 도중에 되풀이해서 넣는 method로 만든다. 회전은 어떤 설정에서도 못 배웠으므로 따로 본다.

## 2026-11-20 저녁, 첫 숫자를 뽑은 뒤의 상태

Claude Code가 썼다. 브랜치는 `code/baseline-observation`이다. 먼저 `docs/reports/2026-11-20-first-numbers.md`를 읽는다.

### 나온 것

- 랩 코드를 그대로 들여왔고, BabyTheorist 파일을 읽는 dataset, 15칸 격자를 받는 CNN state encoder와 state decoder, 길이별 정확도와 code 대 primitive 표를 내는 평가, 틀린 과제를 그리는 스크립트, 작업 줄을 돌리는 스크립트를 더했다. 테스트 35개가 통과한다.
- 랩 NEO는 GridWorld 설정 그대로는 `single-A` 레벨 1에서 transfer 정확도가 0이다. state를 키우거나 MLP를 넓히거나 primitive를 이동 넷으로 줄여도 0이다.
- 32차원 latent 하나는 15칸 격자를 그대로 다시 그리지 못한다. 칸마다 token을 두면 완전히 다시 그린다.
- transformer NEO는 12,000스텝에서 예시 쌍 재현 30%, transfer 정확도 7%이고 끝까지 오르는 중이었다. seed 하나의 값이다.
- transformer NEO는 primitive 하나를 code 여러 개의 열로 설명한다. 길이가 1인 과제의 절반에서 code를 넷 이상 썼다.
- BabyTheorist의 기본 primitive만으로 풀리는 ARC 과제는 ARC-AGI-1 학습 400개 가운데 8개, 평가 400개 가운데 0개다.

### 지금 돌고 있는 것

19:35 현재 GPU 4번과 5번에서 작업 줄이 돌고 있다. 줄은 `runs/queue/jobs_gpu4.txt`와 `runs/queue/jobs_gpu5.txt`에 있고 진행은 같은 폴더의 `runner_gpu4.log`와 `runner_gpu5.log`에 적힌다.

- GPU 4번: primitive 하나짜리 과제만으로 학습하는 transformer NEO, 이동 넷만 쓰는 데이터로 학습하는 transformer NEO, transformer NEO의 seed 44. 각각 2시간쯤이다.
- GPU 5번: transformer NEO의 seed 43, 그다음 36,000스텝짜리 긴 학습. 긴 학습은 6시간쯤이다.

끝난 run의 seed 평균 표는 `python scripts/babytheorist/summarize.py`로 뽑는다.

### 다음에 할 일

1. 긴 학습이 끝나면 transformer NEO가 어디까지 가는지와 code 대 primitive 표가 어떻게 달라지는지 본다.
2. 진단 둘의 결과로 transformer NEO가 primitive 하나를 배우는 데서 막히는지, 이어 붙이는 데서 막히는지 가른다.
3. autoresearch의 한 바퀴로 쓸 빠른 설정을 정한다. transformer NEO는 한 번에 두 시간이 넘어서 그대로는 바퀴가 느리다.

### 사용자가 정해야 하는 것

1. 랩이 논문에서 BabyTheorist에 쓴 NEO의 설정과 코드를 받을 수 있는가. 공개된 GridWorld 설정으로는 논문의 점수가 나오지 않는다. 지금 가장 값이 큰 일이다.
2. Agent가 뜻하는 것이 배운 theory로 목표에 닿는 것인가, 무엇을 실험할지 스스로 고르는 것인가.
3. 평가의 기본을 정답을 보여 주지 않는 엄격한 방식으로 삼아도 되는가. 랩 평가기는 새 입력에 실행하다가 정답과 같아지면 멈춘다.

## 2026-11-20 레포를 만든 직후의 상태

Claude Code가 썼다. claude.ai 채팅의 Claude가 이어서 본다.

이 레포에는 문서만 있다. 코드, 데이터, 실험 결과는 없다. 브랜치는 `code/init`이다. 레포는 처음에 공개였고 사용자가 비공개로 바꾼 뒤에 push했다. 랩 NEO 코드는 다음 차례에 이전 레포의 commit `b30af0d`에서 꺼내 넣는다.

읽는 순서는 `docs/discussion-2026-11-20.md`, `docs/discussion-2026-11-20-autoresearch-and-arc.md`, `docs/research-plan.md`, `docs/facts.md`다. 둘째 논의 기록은 연구 계획을 쓴 뒤에 나온 것이라 계획에 아직 들어가지 않았다. 사용자가 NEO 모델 자체를 autoresearch하고 싶다고 했고 ARC-AGI-1과 2에서도 돌려 보자고 했다. 앞의 것이 왜 이렇게 잡았는지이고, 가운데가 무엇을 할지이고, 뒤의 것이 무엇에 기대는지다.

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
3. Claude Code가 낸 수정 가운데 남은 둘을 받는 것인가. 지금 NEO를 그대로 돌리는 것과 한 방향씩 키우는 것이다. BabyTheorist를 쓰는 것에는 사용자가 동의했다.
4. NEO 모델의 autoresearch를 관찰 뒤에, 변경마다 관찰된 실패 하나를 겨냥하게 하는 조건으로 하는 것인가.
5. ARC-AGI를 모델을 고르는 데 쓰지 않는 온도계로 두고, NEO를 돌리기 전에 primitive만으로 풀리는 과제의 수를 먼저 세는 것인가.

### 이전 레포와의 관계

이전 레포 `minjunjee/neo-vocabulary-restructuring`은 비공개이고 그대로 둔다. 마지막 상태는 브랜치 `code/two-properties-round-1`에 있다. 거기서 넘어오는 것은 `docs/facts.md` 4절의 가설 셋뿐이다. BabyTheorist 저장소는 서버의 `/home/minjun_dev/neo-vocabulary-restructuring/runs/external/babytheorist`에 받아 두었다.
