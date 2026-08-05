# Optimization Method Comparison

이산 조합 공간(30 ordinal 컬럼, ≈10^15 조합)에서 다목적(6개) 블랙박스 최적화를
다루는 프레임워크. 실제 TEST 가 비싸므로, **이미 확보한 실측 관측으로 세운
반응표면**을 계산기로 삼아 알고리즘을 돌려보고 검증한다.

한때 이 저장소는 합성 벤치마크 5종(bm1~bm5)으로 optimizer 11종을 겨루는
벤치마크 프레임워크였다. 그 코드는 전부 걷어냈고 배운 것은
[`lesson_learned.md`](lesson_learned.md) 에 있다 (코드는 `legacy` 브랜치).

## 파일 구성

| 파일 | 역할 |
|---|---|
| `space.py` | **탐색 공간 표준 명세** — signed 정수 범위 SearchSpace. 임의의 문제 기하(x_min/x_max/blocks)가 여기를 통과해 표준화되고, 나머지 전부가 이 인터페이스만 소비한다 |
| `calculator.py` | 문제 정의 — **X → raw y_raw 계산기**. 실측 관측으로 세운 반응표면 `SurfaceCalculator` |
| `optimizer.py` | **나머지 전부** — stateless optimizer 11종 + 히스토리 누적 + 온라인 스케일링/sense 통일/scalarization (공유 score 파이프라인) + 파일 교환 셸(x.txt/y_raw.bin) + 체크포인트(history.jsonl/state.pkl) |
| `runner.py` | calculator ↔ optimizer 를 **반복 호출하는 기계** (ask → 순차 평가 → tell) |
| `make_dataset.py` | **관측 데이터셋 생성** — one-hot 스크리닝 / 무작위 / 반복측정 설계 → `obs.jsonl` |
| `lesson_learned.md` | 합성 벤치마크 시대에서 건진 것 (설계·알고리즘·관측모델 교훈) |
| `doc/algo/` | 알고리즘 소개 문서 (`xgb_tr`, Chow-Liu 트리 EDA) |

## 문제 구조

- **X**: 30개 ordinal 컬럼, 각 컬럼은 **signed 정수 구간 [x_min, x_max]** 의 값
  (기본 기하: cardinality 2~30을 0 중심으로 배치, 예: card 30 → [−15, 14]).
  전체 조합 ≈ 10^15. 무효 조합 없음. `space.SearchSpace` 가 표준 명세이며,
  값↔슬롯 변환은 반드시 `x − x_min` 오프셋을 거친다 ([0, card) 산술 금지).
- **블록 구조** (도메인 지식, optimizer가 활용 가능):
  - `common` (col 0–9) → 6개 목적 전부에 영향. **trade-off가 이 블록에 인코딩**됨.
  - `set1` (col 10–14) → y11, y12, y13 에만 영향 (유효차원 15, 쉬움)
  - `set2` (col 15–29) → y21, y22, y23 에만 영향 (유효차원 25, 병목)
  - `set1 ⫫ set2 | common`
- **y_raw (관측 원형)**: 6개 스칼라가 아니라 **구조화 관측**이다 —
  boolean 타원 마스크 2장 (`mask1`/`mask2`, 각 (b, 128, 128), 가우시안 필드
  G≥0.5 임계로 렌더) + 스칼라 2개 (`y13`/`y23`).
  - **y11/y12** = mask1 의 max height / max width, **y21/y22** = mask2 의
    max height / max width — 측정(마스크→수치)은 optimizer 의
    `convert_y_raw` 이음새 소관. 최대화 y11,y12,y21,y22 / 최소화 y13,y23.
  - 스칼라는 j그룹 은닉 스케일 적용, 타원 측정치는 픽셀 단위 —
    **값 범위 사전 정보 없음** 전제는 동일 (온라인 스케일러가 흡수).
  - 관측 노이즈: 타원 = **격자 양자화(반축 픽셀 반올림)뿐 — 결정적**,
    스칼라 = 주효과 표준편차의 5% 가우시안.
- **평가**: 병렬 불가(순차). 실제 TEST 는 비싸므로 반응표면으로 대역한다.

## 스케일링 / Scalarization

값 범위를 모르므로 **매 tell마다 전체 히스토리에 robust quantile(p5–p95)
스케일러를 재적합**하고, 모든 목적을 "1 = best" 방향으로 통일한 뒤 스칼라화한다.
이 파이프라인은 **optimizer.py 소유**(`RobustScaler` + `SCORERS`)이며, 탐색
구동은 `OptimizerBase.tell` 내부에서 이 구현을 쓴다. sense(max/min 방향)
적용은 `RobustScaler.transform` 한 곳뿐이다.

주의: 스케일러가 온라인이라 **과거 점수가 매 tell 마다 바뀐다.** surrogate 를
점수에 피팅하면 라벨이 비정상성을 띠어 전체 재학습이 강제된다 — 목적별 raw y
에 모델을 세우면 이 문제가 없다.

| scorer | 정의 | 용도 |
|---|---|---|
| `sum` | 정규화 값 평균 | baseline (한 목적 폭락을 못 막음) |
| `chebyshev` | augmented Chebyshev, ρ=0.01 (기본값) | 최악 목적 방어 — 실전 배포 후보 |
| `owa` | bottom-2 OWA (최악 2개 평균) | Chebyshev보다 완만한 안전장치 |

run 간 비교 시에는 같은 벤치마크의 모든 관측을 합친 **pooled 스케일러**로
재점수화한다 (run마다 정규화 기준이 달라 직접 비교가 불가하기 때문).

## Optimizer (ask-tell, stateless)

optimizer 인스턴스는 설정만 갖고, 탐색 상태(히스토리 포함)는 순수 dict 다.
상태는 pickle 직렬화 가능 → 파일 체크포인트/재개 지원. runner 는 점수를
전혀 모른다 — tell 이 이번 batch 의 raw 관측만 받으면, 베이스 클래스가
히스토리 누적 → 스케일러 재적합 → 전 관측 재점수 → 알고리즘 훅(`_update`)
호출까지 처리한다.

```python
state = opt.init_state(seed)
while budget_left:
    X_batch, state = opt.ask(state)         # 후보 1 batch 제안
    # ... calculator 로 순차 평가 ...
    state = opt.tell(state, X_batch, Y_raw)  # 증분 raw 관측 통보
```

구현: `random`(baseline), `blockwise_coord`, `ga`, `sa`, `pso`, `aco`,
`tpe`(직접 구현), `xgb_surrogate`, `eda_tree`(Chow-Liu 의존성 트리 EDA —
[doc/algo/chow_liu_eda.md](doc/algo/chow_liu_eda.md)), `gomea_block`
(블록-FOS GOMEA), `xgb_tr`(trust-region XGB, **현 챔피언** —
[doc/algo/xgb_trust_region.md](doc/algo/xgb_trust_region.md)).

### blockwise_coord — 블록-인지 좌표 local search

블록 구조(도메인 지식)를 명시적으로 활용하는 random-restart hill climbing:

1. **초기점**: marginal-balanced 설계(컬럼별 모든 레벨이 균등 등장) `n_init=32`
   개를 관측하고, 관측 best 를 incumbent 로 삼는다.
2. **스윕**: 라운드마다 `block_order`(기본 **common → set2 → set1**)를 따라
   각 변수를 **1-hop(ordinal ±1)** 스윕하며 변수별 best-improvement 를
   채택한다. common 을 매 라운드 재방문해 블록 간 결합을 흡수한다.
   (set2 를 set1 보다 먼저 다듬는 이유: 유효차원 25의 병목 블록이라
   개선 여지가 크기 때문)
3. **재시작**: 라운드 내 개선이 없으면 수렴으로 판단하고, restart 이력에서
   덜 쓰인 레벨을 우선 뽑는 marginal-balanced 새 점으로 random-restart —
   남은 예산을 다른 basin 탐색에 쓴다 (random-restart hill climbing).
4. **캐시**: 같은 X 재평가는 캐시로 회피해 예산을 아낀다. 탐색은 노이즈
   관측 점수로 하고, 참 점수의 anytime 평가는 calculator 가 그대로 담당한다.

## 파일 교환 & 체크포인트

프로세스 분리 실행을 위한 파일 형식 — 전부 optimizer.py 의 **셸 계층** 소유이며,
ask/tell 은 파일의 존재를 모르는 순수 함수로 유지된다. 공통 규율:
**원자적 쓰기**(tmp + `os.replace`), **fail-loud**(형식/범위/정합성 위반 즉시 raise,
조용한 대체·건너뛰기 금지).

**두 실행 모드**: (1) 기본 `run_single` 은 in-process(함수 호출·배열 전달, 빠름,
파일 없음 — 체크포인트는 `--checkpoint-dir` 로 opt-in). (2) `--separate DIR` 은
**프로세스 분리** — runner 가 `optimizer.py --serve-step` 과
`calculator.py --serve-eval` 을 별도 서브프로세스로 번갈아 spawn 하고, 두
프로세스는 **공유 메모리가 없으므로** 아래 파일들이 유일한 통신 수단이다.
실제 문제가 파일 매개 프로세스 분리를 계약으로 요구한다면 이 모드가 그 계약을
강제한다 — 파일 교환이 실제로 일어나야만 스텝이 진행되므로 in-process 우회가
물리적으로 불가능하다. optimizer 는 매 스텝 새 프로세스로 뜨며 state.pkl +
history.jsonl 이 스텝 간 유일한 기억이다. 한 스텝의 핸드셰이크:
`opt-step`(state 로드 → 직전 y_raw ingest → ask → x.txt·state 저장) →
`calc-eval`(x.txt 읽기 → 평가 → y_raw.bin) → 반복, 예산 소진 시 `done` 마커.
(검증 지문: x.txt / y_raw.bin 의 `eval_index` 가 매 라운드 배치 크기만큼 전진.)

| 파일 | 형식 | 방향/성격 | 함수 |
|---|---|---|---|
| `x.txt` | 텍스트 | optimizer → calculator. 다음 후보 배치 | `write_x` / `read_x` |
| `y_raw.bin` | 바이너리 | calculator → optimizer. raw 관측 | `write_y_raw` / `read_y_raw` |
| `history.jsonl` | jsonl | 체크포인트 — **관측의 진실**, append-only | `append_history` / `load_history` |
| `state.pkl` | pickle | 체크포인트 — 알고리즘 상태 + RNG + 스케일러 + 점수 캐시 | `save_state` / `load_state` |

**x.txt** — 1행 헤더 `# eval_index=<int>` (배치 첫 평가의 전역 카운터 — 노이즈
시딩·대응 검증용), 2행부터 한 줄 = 해 하나 `[15,0,-1,...]` (signed 정수,
텍스트 왕복 무손실).

**y_raw.bin** — 내부 구조를 우리가 통제하지 못하는 **불투명 바이너리**로 취급.
실제 문제의 레이아웃이 다르면 교체 지점 두 개만 갈아끼운다 (하류 불변):
`read_y_raw`(① bin 디코딩) → `convert_y_raw`(② 관측 원형 → 표준 (b, K) float64
— 마스크 측정 + 스칼라 통과, NaN/inf 즉시 raise). 레퍼런스 레이아웃:
int64 `eval_index, b, G, n_scalar` + uint8 mask1/mask2 (b·G·G씩) +
float64 y13/y23 (b씩), LE. in-process 경로에서도 같은 이음새를 지난다 —
`OptimizerBase.tell` 이 구조화 y_raw 를 받아 내부에서 `convert_y_raw` 를 호출.

**history.jsonl** — 한 줄 = tell 한 번:
`{"eval_index":0,"X":[[...]],"y_raw":[[...]]}`. y_raw 필드에는 마스크 원형이
아니라 **변환된 (b, K) 측정치**를 기록한다 (마스크는 용량·가독성 문제로 보존
안 함 — 변환이 결정적이라 정보 손실은 측정 정의 그 자체뿐). X 는 정수,
측정치는 json 의 shortest-round-trip repr 라 float64 무손실. 사람이 읽고
diff 할 수 있으며 pkl 없이도 post-hoc 분석(anytime 곡선·pooled 재점수)이
가능하다. 로드 시 **eval_index 연속성**을 검증해 빠지거나 중복된 batch 를
즉시 잡는다.

**state.pkl** — 히스토리를 제외한 나머지 (알고리즘 상태, RNG, 스케일러 파라미터,
점수 캐시 — 점수는 스케일러 *이력* 에 의존하는 파생 상태라 관측이 아닌 상태로
분류). `load_state(state.pkl, history.jsonl)` 이 두 파일을 합쳐 완전한 state 를
재구성하며, **pkl 의 n_evals ≠ jsonl 누적 평가 수면 즉시 raise** (정합성).
재개 계약: 중단 후 재개 = 무중단 실행과 **동일 궤적** (`python optimizer.py`
자가 점검이 sa/ga 로 검증). jsonl 이 진실이므로 pkl 이 깨져도 히스토리를
tell 로 재생(replay)해 상태를 재구성할 수 있다.

## 워크플로 — 데이터 만들기 → 반응표면 → 알고리즘 검증

```
make_dataset.py  ──▶  obs.jsonl  ──▶  SurfaceCalculator  ──▶  runner.py
  (설계 + 측정)        (관측)        (반응표면)             (알고리즘 구동)
```

관측 파일은 **append-only jsonl 한 개**다. 한 줄이 관측 하나이고, 마스크
원형까지 그 줄 안에 들어간다:

```json
{"i":0,"block":"one_hot","X":[...30개...],"Y":[...6개...],
 "mask1":{"shape":[128,128],"runs":[[55,63,2],[56,62,5],...]},"mask2":{...}}
```

마스크는 행 단위 run-length(`[row, col0, len]`)다 — 무손실이고, blob 은 행마다
연속 구간이 한두 개뿐이라 조밀하며(111점 242KB; base64 비트팩이면 634KB),
무엇보다 눈으로 읽힌다. 관측이 이 프로젝트의 진실이므로 사람이 읽고 git 으로
diff 할 수 있어야 한다는 요구가 형식을 정했다 — `history.jsonl` 과 같은 원리다.
관측이 늘면 `--append` 로 이어쓴다 (전체 재작성 없음).

### 1. 관측 설계 (`make_dataset.py`)

세 블록이 각각 다른 질문에 답한다:

| 블록 | 점 수 | 무엇을 알려주나 |
|---|---|---|
| **one-hot 스크리닝** | 1 + 30 | 기준점(전 컬럼 x_min)에서 **한 컬럼만 x_max**. 컬럼당 1점으로 주효과 크기를 잰다 |
| **무작위** | 60 | 지형 전반의 눈금. 보간의 재료이자 조합 효과의 단서 |
| **기본값 반복측정** | 20 | 같은 X 반복 → **관측 노이즈 바닥**. 게이트 설정과 보간 오차 비교의 기준 |

one-hot 블록은 블록 구조를 실제로 복원한다 — 시뮬레이션 장치에서 y11 의
주효과가 정확히 common+set1 15컬럼에서만 검출된다(오검출 0). 노이즈 바닥이
없으면 "모델이 틀린 건지 측정이 시끄러운 건지" 구분할 수 없다.

관측 장치는 `SimulatedInstrument`(다이아몬드에 가까운 blob). 실제 장치가
확보되면 이 클래스만 갈아끼우면 되고 하류는 그대로다.

### 2. 반응표면 (`calculator.SurfaceCalculator`)

예측 모델이 아니라 **재생기 + 국소 보간기**다. 관측점은 실측을 그대로 내고,
관측에서 멀면 값을 지어내지 않고 "데이터 없음"으로 표시한다. 평활 회귀
(릿지·GBM·nugget 있는 GP)는 첫 번째 성질을 원리적으로 못 지킨다.

판정 사다리 — **목적 그룹별로 독립**(y1x 는 데이터가 있고 y2x 는 없는 X 가 존재):

| 상태 | 조건 | 값 |
|---|---|---|
| `exact` | 30컬럼 전부 일치하는 관측이 있다 | 그 실측값 그대로 (마스크는 바이트 그대로) |
| `exact_block` | 해당 목적의 의존 블록이 전부 일치 | 그 관측들을 합성 |
| `interp` | 최근접 거리 ≤ `d_gate`, 이웃 불일치 ≤ `spread_gate` | k최근접 역거리가중 보간 |
| `no_data` | 그 외 | 지어내지 않음 (`flag`/`pessimistic`/`strict` 정책) |

- 거리는 **블록 제한 정규화 해밍** — 목적이 의존하지 않는 컬럼의 차이는 세지 않는다.
- 근거 관측이 여럿이면 스칼라는 가중평균, boolean 마스크는 **blob 형상 보간**
  (무게중심 + 각도별 반경 r(θ) 를 각각 가중평균 후 재래스터화). 픽셀 단위
  평균은 덩어리 가장자리를 갈라놓는다.
- `neighbors()` / `explain()` 로 근거 관측과 그때의 y 를 되짚고,
  `report()` 로 알고리즘이 커버리지를 얼마나 벗어났는지, `loo_report()` 로
  보간의 정직도를 본다.

### 3. 알고리즘 검증 (`runner.py`)

반응표면이 답하는 것은 "누가 이기나"가 아니라 두 가지다:

1. **완주하는가** — ask → 평가 → tell 전 경로가 프로그램적으로 돈다
2. **터무니없는 걸 추천하지 않는가** — `report()` 의 `no_data_rate` 와
   관측까지의 해밍거리 분포가 그 답이다

`no_data` 정책을 `pessimistic`(커버리지 밖 = 최악 관측치)으로 두면 이 검증이
실제로 판별력을 갖는다. 관측 111점 · budget 80 에서 11종 전원 완주했고,
"추천이 데이터에 근거하는가"가 뚜렷하게 갈렸다:

| optimizer | no_data | 관측까지 해밍거리 중앙값 |
|---|---|---|
| `xgb_tr` | 27.5% | 0.033 |
| `xgb_surrogate` | 29.4% | 0.067 |
| `gomea_block` / `ga` | 46~49% | 0.13~0.17 |
| `tpe` | 75.0% | 0.433 |
| `pso` / `aco` / `blockwise_coord` | 100% | 0.43~0.47 |

surrogate 계열은 관측 영역으로 수렴하고, 분포/군집 계열은 끝까지 근거 없는
영역만 제안한다. 후자를 실측에 붙이면 **표면이 답을 못 주는 X 만 추천**한다는
뜻이다.

주의: 관측이 소량이면 자유 탐색의 `no_data_rate` 는 원래 높다(10^15 공간에
관측 수백). 이건 고장이 아니라 검증 2의 답 자체이지만, 그래서 이 표면을
**최적화 성능 비교에 쓰면 안 된다** — 지형 대부분이 근거 없는 채움값이다.

## 실행

```bash
pip install numpy xgboost      # xgboost 는 xgb_surrogate / xgb_tr 에만 필요

# 1) 관측 데이터셋 만들기
python make_dataset.py --out obs.jsonl
python make_dataset.py --out obs.jsonl --n-random 120 --n-repeat 30 --seed 1
python make_dataset.py --out obs.jsonl --append --n-random 40 --seed 1   # 관측 추가
python make_dataset.py --selfcheck        # 설계·노이즈 요약 + 불변식 점검

# 2) 알고리즘 구동
python runner.py --optimizer sa --surface-data obs.jsonl --seed 0 --budget 800
python runner.py --optimizer ga --surface-data obs.jsonl --checkpoint-dir ckpt/
#   → ckpt/history.jsonl + ckpt/state.pkl (optimizer.load_state 로 재개 가능)
python runner.py --optimizer ga --surface-data obs.jsonl --budget 780 --separate xchg/
#   → 프로세스 분리: optimizer/calculator 를 별도 서브프로세스로 띄우고
#     xchg/ 의 x.txt·y_raw.bin·history.jsonl·state.pkl 로만 통신 (in-process 우회 불가)
#     커버리지 판정은 xchg/coverage.jsonl 에 append
```

각 모듈은 자가 점검용 `__main__` 을 갖는다:
`python space.py` (공간 불변식), `python calculator.py --surface-selfcheck`
(반응표면 요구 3종 + blob 보간), `python optimizer.py` (전체 optimizer 의
ask-tell 사이클 + pickle 체크포인트), `python make_dataset.py --selfcheck`
(설계 불변식 + 블록 구조 복원).
