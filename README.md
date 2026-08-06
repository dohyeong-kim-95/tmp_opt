# tmp_opt — 미지의 반응표면 위에서 도는 최적화

## 무엇을 하는 프로젝트인가

**REAL 반응표면이 존재한다. 우리는 그것을 모른다.** 30컬럼 조합공간은 ~10^15 라
전수 파악은 불가능하다. 그래서 표면을 알아내는 대신, **그 표면 위에서 좋은 X 를
골라내는 최적화 알고리즘**을 만든다.

좋은 X = **blob(boolean array)이 크고 스칼라값이 작은** 조건.

한 번의 실측이 **5분**이다. 이 비용이 설계 전체를 지배한다 — 알고리즘은 평가
횟수로 평가받고, 관측 한 점 한 점이 자산이다.

## 문제 정의

| | |
|---|---|
| 입력 X | 30컬럼 정수 벡터 |
| 블록 구조 | `common`(0–9) 전 목적에 영향 · `set1`(10–14) → y1x · `set2`(15–29) → y2x |
| | set1 ⫫ set2 \| common (공통 블록을 통해서만 결합) |
| 출력 y_raw | boolean blob 마스크 2장 + 스칼라 2개 |
| 목적 6개 | y11,y12 (mask1) · y13 (스칼라) · y21,y22 (mask2) · y23 (스칼라) |
| 방향 | y11,y12,y21,y22 **최대화** / y13,y23 **최소화** |

## 이번 세션의 범위

1. **jsonl append 누적 관리** — 관측을 효율적으로 쌓는 구조
2. **pseudo-반응표면** — 1번으로 누적한 데이터로 세우는 surrogate
3. **optimizer 연결** — optimizer 가 2번 표면을 쓰도록

### pseudo-반응표면의 성격 (여기서 오해하면 다 틀어진다)

용도는 **optimizer 안의 surrogate** 다 — 5분짜리 실측을 아끼려고, 다음에 잴 X 를
고를 때 표면으로 미리 걸러낸다.

그런데 관측이 **수십 점**뿐이다. 그래서 요구가 둘로 갈린다:

- 데이터가 있는 곳 → **실측과 유사한 수준으로 정확히** 재생
- 그 외 대부분 → **"불확실한 영역"으로 표시**. 값을 지어내지 않는다

즉 예측 정확도를 좇는 평활 회귀(릿지/GBM/nugget GP)가 아니라, **정직한 재생기 +
불확실성 정량화**다. 커버리지 밖에서 그럴듯한 값을 내면 관측에서 먼 곳이 더 좋아
보이는 가짜 최적이 생겨 optimizer 를 망친다.

## 현재 상태

### 1. 누적 관리 — 완료

| 파일 | 책임 |
|---|---|
| `space.py` | 탐색 공간 명세 (30컬럼 signed 정수 · 블록 구조). 기존 repo 그대로 |
| `record.py` | **저장 형식** — `obs.jsonl` 한 줄 = 한 측정. RLE blob 2장 + 스칼라 2개 |
| `score.py` | **점수 정의** — raw → 목적값. 교체 가능 (`area` / `extent` / `area+extent`) |
| `ingest.py` | **머지** — 측정 파일 디렉토리 → `obs.jsonl` (멱등). reader 이음새 |

```bash
python ingest.py --check measurements/m0001.npz    # reader 계약부터 확인
python ingest.py --src measurements/ --out data/obs.jsonl
python ingest.py --out data/obs.jsonl --validate
```

설계 결정 세 가지:

- **raw 만 저장하고 y 는 저장하지 않는다.** 점수 정의(면적 ↔ max 폭·높이)는
  언제든 바뀐다. y 를 파일에 넣으면 바뀌는 순간 전부 stale 이 된다. 지표는
  RLE 에서 바로 계산되므로(200점 × 8목적 재채점 12ms) 전량 재채점이 공짜다.
- **마스크 shape 를 줄마다 기록한다.** 두 장은 **서로 크기가 다르다**. 전역
  격자 상수 하나로 가정하면 그 자리에서 깨진다. 측정끼리의 일관성은
  `validate()` 가 검사한다.
- **원본 형식은 reader 이음새 하나로 격리한다.** `read_npz` 가 참고 구현이고,
  실제 형식에 맞는 함수를 채워 `register()` 하면 하류 전부가 그대로 돈다.
  `--check` 로 머지 전에 계약을 확인한다.

### 2. pseudo-반응표면 — 완료

| 파일 | 책임 |
|---|---|
| `surface.py` | **surrogate** — μ(정확 재생 + XGB 배깅) · σ(앙상블 + 노이즈 + 거리) |
| `sim.py` | 검증용 시뮬레이션 측정기. REAL 표면의 근사가 **아니다** |

**μ — 판정 사다리.** `exact`(30컬럼 일치) → `exact_block`(그 목적의 의존 컬럼
일치) → 모델. 앞의 둘이 모델을 덮어쓴다. 트리 앙상블은 관측점도 평활하므로,
모델에만 맡기면 "데이터가 있는 곳은 정확히" 를 원리적으로 못 지킨다.

**σ² = 앙상블 불일치 + 노이즈 바닥 + 거리 팽창.** 거리항이 필수다 —
**트리 앙상블은 외삽에서 오히려 자신만만해진다**(데이터 밖 점도 학습된 leaf 로
떨어져 멤버 불일치가 0 이 된다). GP 와 정반대이고 XGB-BO 의 대표적 함정이다.
노이즈 바닥은 **반복측정에서만** 나온다.

**α — LOO 캘리브레이션.** std(z)=1 이 되게 맞춘다. σ 는 크기가 맞아야 쓸모가
있다. 그리고 그 캘리브레이션이 **optimizer 가 실제로 질의할 거리를 덮는지**까지
`report()` 가 검사한다(안 덮으면 α 는 외삽이고 믿을 수 없다).

**거리** — unit 공간 정규화 L1 을 목적의 의존 컬럼으로 제한. 해밍은 card=30
컬럼에서 1칸 차이와 29칸 차이를 같게 보므로 주 거리로 쓰지 않는다.

### 3. optimizer — 완료

| 파일 | 책임 |
|---|---|
| `optimize.py` | **BO** — ParEGO 스칼라화 + Monte-Carlo EI + 블록 인지 후보 풀 |

```bash
python optimize.py --obs data/obs.jsonl --q 4     # 다음에 잴 X 4점
```

GP-BO 는 이 규모에서 느려 피했다. surrogate 는 `surface.py`, acquisition 만
정석대로 세운다:

- **ParEGO** — 제안마다 λ 를 새로 뽑아 목적을 augmented Chebyshev 로 접는다.
  λ 가 달라 배치가 파레토 전선의 다른 구역을 노린다. 목적 수가 늘어도 싸다.
- **Monte-Carlo EI** — f_λ 가 min 을 품어 정규분포가 아니다. 해석적 EI 를 쓰면
  그 자리에서 틀린다.
- **f\* 는 관측 최댓값이 아니라 사후평균 최댓값** — 노이즈 관측의 최댓값은 위로
  편향돼 있다 (noisy BO 표준 교정).
- **블록 교차** — set1 ⫫ set2 \| common 이므로 common 을 한쪽에서 통째로
  가져오면 한 점의 set1 과 다른 점의 set2 를 붙여도 각 그룹의 성질이 보존된다.
  문제 구조가 실제로 허락하는 조작이다.

### 전체 흐름

```bash
python ingest.py --check measurements/m0001.npz     # reader 계약 확인
python ingest.py --src measurements/ --out data/obs.jsonl
python optimize.py --obs data/obs.jsonl --q 4       # 다음에 잴 X 4점
# 측정 → measurements/ 에 파일이 늘어남 → 다시 ingest → 다시 optimize
```

각 모듈은 자가 점검을 내장한다: `python record.py` / `score.py` / `sim.py` /
`surface.py` / `ingest.py --selfcheck` / `optimize.py --selfcheck`.

### 실측 데이터에 붙일 때 해야 할 일

1. `ingest.read_npz` 를 본떠 원본 형식을 읽는 함수를 짜고 `register()` 한다.
   `--check <파일>` 로 계약부터 확인할 것.
2. `sim.py` 는 지우지 말고 두되 실측 경로에는 쓰지 않는다 (검증 전용).
3. **반복측정을 걸 것.** 노이즈 바닥은 여기서만 나오고, 그게 없으면 σ 의 하한이
   추정치로 대체된다. 한 X 에 몰지 않고 서로 다른 X 에 2회씩 나눠도 dof 가 쌓인다.
4. `optimize.py` 가 찍는 `report()` 경고를 매번 읽을 것 — 캘리브레이션이
   질의 거리를 못 덮으면 σ 는 외삽이고, 그때는 `--alpha` 로 직접 올려 잡는다.

### 참고 자산

과거 자산은 `archive/` 에 참고용으로 둔다.

| | |
|---|---|
| `archive/synth_benchmark/` | 합성 벤치마크 bm1~5 + **optimizer 11종** + doc/ examples/ vis/ |
| `archive/surface/` | 실측 반응표면 `SurfaceCalculator` 스택 (obs.jsonl · make_dataset · run · accept) |

두 계열 모두 **참고용**이다. 새 코드는 백지에서 짜되:
- optimizer 는 `archive/synth_benchmark/optimizer.py` 의 11종을 출발점으로 참고
- 표면의 "정직한 재생기" 철학은 `archive/surface/calculator.py` 를 참고
  (단 그건 mock 용도로 지어졌다 — surrogate 로 쓰려면 불확실성 출력이 필요하다)

## 데이터

실측 데이터는 **존재하고 느리게 생산 중**이다. 원본 형식은 비공개이며, 구조만
설명받아 설계한다. 따라서 `원본 → 표준 형식` 변환은 **이음새로 분리**한다.
