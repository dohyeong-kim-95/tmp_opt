# Optimization Method Comparison

이산 조합 공간(30 ordinal 컬럼, ≈10^15 조합)에서 다목적(6개) 블랙박스 최적화
method 들을 비교하는 벤치마크 프레임워크.

## 파일 구성 (플랫 4파일 구조)

| 파일 | 역할 |
|---|---|
| `space.py` | **탐색 공간 표준 명세** — signed 정수 범위 SearchSpace. 임의의 문제 기하(x_min/x_max/blocks)가 여기를 통과해 표준화되고, 나머지 전부가 이 인터페이스만 소비한다 |
| `calculator.py` | 문제 정의 — **X → raw y0 계산기** (벤치마크 5종 + 노이즈) |
| `optimizer.py` | **나머지 전부** — stateless optimizer 11종 + 히스토리 누적 + 온라인 스케일링/sense 통일/scalarization (공유 score 파이프라인) |
| `runner.py` | calculator ↔ optimizer 를 **반복 호출하는 기계** (ask → 순차 평가 → tell) |
| `benchmark.py` | 여러 run 비교 — pooled 재점수, 토너먼트/격자, top-3, true optimum, 시각화 |
| `doc/algo/` | 알고리즘 소개 문서 (예: Chow-Liu 트리 EDA) |
| `results/` | 실행 결과 (parquet/json) — 실행 시 자동 생성 |
| `vis/` | 시각화 png — 실행 시 자동 생성 |

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
- **y0**: 6개 스칼라. 최대화 y11,y12,y21,y22 / 최소화 y13,y23.
  - j(뒤 숫자)가 같으면 스케일 유사, 다르면 크게 다름. **값 범위 사전 정보 없음.**
  - 관측 노이즈: 주효과 표준편차의 5% 가우시안.
- **평가**: 병렬 불가(순차), run당 예산 800회. 반복측정은 최종 confirmation에서만.

## 스케일링 / Scalarization

값 범위를 모르므로 **매 tell마다 전체 히스토리에 robust quantile(p5–p95)
스케일러를 재적합**하고, 모든 목적을 "1 = best" 방향으로 통일한 뒤 스칼라화한다.
이 파이프라인은 **optimizer.py 소유**(`RobustScaler` + `SCORERS`)이며, 탐색
구동(OptimizerBase.tell 내부)과 리포트(benchmark.py 의 pooled 재점수)가 같은
구현을 공유한다. sense(max/min 방향) 적용은 `RobustScaler.transform` 한 곳뿐이다.

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

## BM3 (BenchmarkHard) 제작 상세

세 벤치마크 중 가장 어려운 bm3_hard 가 어떻게 만들어졌는지 기록해 둔다.
모든 내부 파라미터는 구조 시드(`_structure_seed=303`)로 한 번만 생성되므로
문제 자체는 완전히 재현 가능하고, 노이즈 시드와는 분리되어 있다.

### 입력 표현

레벨 인덱스를 등간격으로 [0,1] 에 매핑한다: `u_i = x_i / (card_i − 1)`.
ordinal 가정이 여기서 쓰인다 — 이웃 레벨은 함수값도 가깝다.

### latent 함수 (목적 k = 1..6)

각 목적의 latent 값은 6개 성분의 합이다. 목적 k 는 자기 그룹의 컬럼만 본다
(group1 = common+set1 → y11,y12,y13 / group2 = common+set2 → y21,y22,y23).

```
f_k(u) = 0.7 · Σ_c w_kc · (1 − (u_c − p_kc)²)          ① 단봉 골격
       + 0.05 · Σ_{8쌍} s_kp · u_a · u_b                ② 블록 내 상호작용
       + Σ_{10쌍} w×_kp · cos(π · (u_common − u_set))   ③ 교차-블록 상호작용
       + Σ_{3컬럼} w†_kp · g(u_c)                        ④ deceptive
       + (1/30) · Σ_c r_kc · sin(15~25 · u_c + φ_kc)    ⑤ rugged
       + gain_k · c(u)                                   ⑥ trade-off
```

| 성분 | 파라미터 | 난이도에 기여하는 방식 |
|---|---|---|
| ① 단봉 골격 | peak p ~ U(0,1), 가중치 정규화, 계수 0.7 | 기본 지형. Easy/Medium 대비 비중을 0.7배로 낮춰 나머지 성분의 영향력을 키움 |
| ② 블록 내 pairwise | 목적당 8쌍, 부호 ±1 랜덤 | 같은 블록 안 변수끼리의 결합 (Medium 과 동일) |
| ③ **교차-블록** | 목적당 10쌍 (common 컬럼 × 자기 set 컬럼), w× ~ U(0.03, 0.07) | `cos(π(u_c − u_s))` 는 common 값이 바뀌면 set 쪽 최적 위치가 **함께 이동**하게 만든다. "common 먼저 고정 → set 최적화" 식 블록 분해가 함정에 빠지는 이유 |
| ④ **deceptive** | 목적당 3컬럼, w† ~ U(0.10, 0.16) | `g(u) = 0.35u + 1.2·max(0, u−0.85)/0.15`. 넓은 구간(u<0.85)에서는 완만한 오르막이라 u≈0.8 언저리 가짜 정상으로 유도되지만, 진짜 정상은 u≈1 근처 **폭 0.15의 급경사** 위에 있다. 해상도가 낮은 탐색과 1-hop 언덕오르기를 저격 |
| ⑤ rugged | 주파수 15~25 (고주파), 진폭 ≈ 0.02 | 노이즈(주효과의 5%)와 구분하기 어려운 잔물결 — 미세한 개선 신호를 오염시킴 |
| ⑥ trade-off | c(u) = common 블록 가중평균, gain ~ U(0.35, 0.5) | c 가 커지면 **모든** latent 가 커진다. 최대화 목적(y·1, y·2)에는 이득이지만 최소화 목적(y·3)에는 손해 → "전부 다 좋은 해"가 존재하지 않음. 충돌이 common 블록에 인코딩되는 지점 |

### 스케일/노이즈 은닉

- raw 출력: `y_k = scale_k · f_k + offset_k`. j 그룹별 스케일 규격
  (j=1: ~수천, j=2: ~1 (음수 오프셋), j=3: ~0.00x)에 ±20% 지터.
  → "j 가 같으면 스케일 유사, 다르면 상이 + 범위 사전정보 없음" 요구 구현.
- 노이즈: `N(0, (0.05·σ_k)²)`, σ_k 는 초기화 때 4,096점 몬테카를로로 추정한
  latent 주효과 표준편차. raw 스케일 적용 전에 더해진다.

### 왜 어려운가 (설계 의도 요약)

1. ③ 때문에 블록을 따로 최적화하면 common 이 움직일 때마다 set 의 최적이
   무효화된다 — common 재방문이 필수.
2. ④ 때문에 국소 개선만 따라가면 가짜 정상(u≈0.8)에 수렴한다 — 진짜 정상은
   레벨 그리드에서 한두 칸 차이의 좁은 basin.
3. ⑤+관측 노이즈 때문에 미세한 개선/악화 판정이 불안정하다.
4. set2 가 15컬럼이라 group2 목적의 유효차원이 25 — 탐색 병목.
   (set1 쪽 유효차원 15는 상대적으로 쉬움)

**사후 발견 (정직 노트)**: BM3 의 g(u) 는 단조증가라 '기만'이 생기는 것은
backbone 과의 합에서뿐이고, deceptive 가중치가 backbone 의 컬럼당 기여보다
크므로 실제로는 **약한 기만**(경계 선호 지형에 가까움)이다. probe 진단
(아래 BM4/BM5 절)에서도 BM3 의 기만성 지표는 낮게 측정된다. 강한 기만이
필요하면 BM4 의 진짜 trap 을 쓸 것.

## BM4 / BM5 — 난이도 요인 분리 변형

"어떤 optimizer 가 이기는가는 벤치마크의 지배적 난이도 요인에 달렸다"는
가설을 검증하기 위한 변형들. 전체 난이도는 BM3 급을 유지하되(trade-off·
노이즈·스케일 규격 동일, 단봉 골격 비중 유사) **지배 요인만 다르다**:

| | BM3 (bm3_hard) | BM4 (bm4_deceptive) | BM5 (bm5_epistasis) |
|---|---|---|---|
| 교차-블록 결합 | 10쌍 × w 0.03–0.07 | 없음 | **20쌍 × w 0.06–0.12** |
| 블록 내 pairwise | 8쌍 × 0.05 | 4쌍 × 0.05 (약) | 12쌍 × 0.08 (강) |
| 기만성 | 약한 g(u) 3컬럼 | **진짜 trap 6컬럼** | 없음 |
| rugged | 있음 (미세) | 없음 | 없음 |

- **BM4 의 trap**: `trap(u) = (v−u)/v (u≤v)`, `1.4(u−v)/(1−v) (u>v)`, v=0.85.
  가짜 정상 u=0 ← 골짜기 → 진짜 정상 u=1 의 고전적 쌍봉. 국소 정보는 전부
  u=0 을 가리키므로 골짜기를 **점프**해야 한다 (card≥4 컬럼만 사용 — card 2
  는 양 끝을 다 보므로 trap 불성립). trap 은 컬럼별 독립(가법적)이라
  상호작용 모델링은 도움이 안 되고 탈출 능력이 승부를 가른다.
- **BM5**: 기만·rugged 없이 매끄럽지만 심하게 비분리 — 어느 컬럼의 최적값도
  다른 컬럼에 조건부다. marginal/좌표 계열의 급소.
- 실행: `python benchmark.py --matrix bm3_hard,bm4_deceptive,bm5_epistasis`
  (탈락식 토너먼트 대신 전 optimizer 를 전 벤치마크에서 완주시키는 격자
  비교 — 요인별 강자 판별용).

## 비교 프로토콜 (토너먼트)

1. **Stage 1** — bm1_easy, 8 optimizers × 3 seeds → 상위 50% 진출
2. **Stage 2** — bm2_medium, 4 optimizers × 5 seeds → 상위 50% 진출
3. **Stage 3** — bm3_hard, 2 optimizers × 10 seeds → 챔피언 결정

순위 기준: seed 평균 최종 best score (pooled chebyshev).

- **top-3 추천**: 챔피언을 서로 다른 init으로 3회 실행 → 후보 3개 →
  각 후보 10회 반복 재측정(confirmation)으로 최종 순위 확정.
- **true optimum**: 챔피언(surrogate 계열이면 최상위 non-surrogate로 대체)을
  무노이즈로 100K회 실행한 참조값. 시각화에 점선으로 표시된다.

## 실행

```bash
pip install numpy scipy scikit-learn polars xgboost matplotlib

# 단일 run (runner.py)
python runner.py --optimizer sa --benchmark bm1_easy --seed 0 --budget 800

# 여러 run 비교 (benchmark.py)
python benchmark.py                       # 토너먼트 + top-3 + 시각화 (800 evals)
python benchmark.py --smoke               # 초소형 예산으로 빠른 동작 확인
python benchmark.py --matrix bm3_hard,bm4_deceptive --seeds 5  # 격자 비교
python benchmark.py --compute-true-optimum  # + true optimum (100K, 오래 걸림)
python benchmark.py --scorer owa          # scalarization 변경
python benchmark.py --plots-only          # 저장된 results/ 로 그림만 재생성
```

각 모듈은 자가 점검용 `__main__` 을 갖는다:
`python calculator.py` (공간/스케일/노이즈 확인), `python optimizer.py`
(전체 optimizer의 ask-tell 사이클 + pickle 체크포인트 검증).
