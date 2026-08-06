# XGB Trust-Region (`xgb_tr`) — CASMOPOLITAN-lite

`optimizer.py` 의 `XGBTrustRegionOptimizer` 전용 문서. 현재 이 리포지토리의
챔피언 optimizer 로, 토너먼트 3단계(easy/medium/hard)와 난이도 요인 격자
(bm3/bm4/bm5) **전부에서 1위**를 기록했다.

---

## 1. 왜 만들었나 — 전신(xgb_surrogate)의 실측 약점

`xgb_surrogate` (XGB 회귀 + 전역 후보 풀 + novelty 보너스)는 bm3_hard 에서
평균 1위였지만 **init seed 에 따른 분산이 치명적**이었다:

- bm3_hard 10 seeds: 0.696 ± **0.102** (동시기 aco ± 0.050)
- top-3 confirmation: **0.702 / 0.511 / 0.260** — init 이 나쁘면 반토막 이하

원인: 후보 풀이 "전역 랜덤 + 상위해 mutation" 이라, 초기 관측이 나쁜 basin
에 몰리면 모델이 그 basin 안만 가리키고 탐색이 그대로 고착된다. 나쁜 시작을
**끊어내는 장치가 없다**.

## 2. 처방의 출처 — 조합공간 BO 의 trust region 계보

연속 고차원 BO 의 TuRBO 에서 시작해 조합/혼합 공간으로 확장된 계보가 있다:

- **CASMOPOLITAN** (Wan et al., 2021): categorical/혼합 공간에서 **해밍
  거리 기반 trust region** 을 정의 — incumbent 로부터 해밍 거리 ≤ R 인
  이웃에서만 acquisition 을 최적화하고, 성공/정체에 따라 R 을 키우고 줄이며,
  수렴하면 restart 한다.
- **Bounce** (Papenmeier et al., 2023): 중첩 부분공간 + trust region 으로
  25-categorical PestControl 벤치마크(우리와 유사한 규모)에서 현 최상위.

핵심 통찰은 간단하다: **전역 surrogate 는 유지하되 후보 생성은 국소화하고,
정체는 반경 축소 → restart 로 명시적으로 처리**한다. 우리 구현은 이 골격에서
GP 를 XGB 앙상블로 바꾼 경량판이라 "CASMOPOLITAN-lite" 로 부른다.
(GP + 해밍 커널이 아닌 XGB 를 쓰는 이유: 이산 ordinal 30컬럼 + 800 evals
규모에서 tree 모델이 값싸고, 이미 xgb_surrogate 로 상호작용 포착력이
실증되었기 때문. SMAC 계열이 "RF surrogate 가 categorical/노이즈에 강건"
을 보여준 것과 같은 노선이다.)

## 3. 알고리즘

ask-tell stateless 구조 (상태는 전부 dict, pickle 가능). 한 사이클:

```
ask:
  1. 관측 < n_startup 또는 restart 직후 → 균등 랜덤 batch (re-seed)
  2. incumbent = 현재 trajectory(마지막 restart 이후) 내 최고 관측
  3. 후보 n_candidates 개 생성: incumbent 에서 해밍 거리 d ~ U(1..R) 인
     이웃 (컬럼당 70% ordinal ±1 스텝 / 30% 랜덤 레벨 점프)
  4. 앙상블 UCB = μ(x) + κ·σ(x) 상위에서, 기평가 제외 batch_size 개 선택

tell (runner 가 전체 히스토리를 재정규화 점수와 함께 통보):
  5. trajectory best 가 이번 batch 에서 나왔으면 succ+=1, 아니면 fail+=1
  6. succ ≥ succ_tol → R ← min(2R, r_max)      (연속 개선: 반경 확대)
     fail ≥ fail_tol → R ← R/2                 (연속 정체: 반경 축소)
     R < 1           → restart                 (반경 붕괴 = 국소 수렴 판정)
  7. refit_interval 마다 XGB 앙상블(n_ensemble개, 시드/부표본 상이)을
     전체 히스토리로 재학습
```

### restart 의 의미 (분산 문제의 해결 지점)

R 이 1 미만으로 붕괴하면 그 basin 은 다 판 것으로 보고, **랜덤 re-seed 로
trajectory 를 새로 시작**한다. 이때 모델과 히스토리는 유지되므로:

- 다음 trajectory 는 이전 지식(전역 모델)을 갖고 시작하고,
- 전역 best 는 runner 의 best-so-far 가 보존하므로 잃는 것이 없다.

즉 "나쁜 초기 basin 고착"이라는 xgb_surrogate 의 실패 모드가 구조적으로
불가능해진다 — 나쁜 basin 은 fail 카운터가 쌓여 반경이 붕괴하고 자동으로
버려진다.

### 판정의 노이즈/재정규화 안전성

개선 판정(5)은 "trajectory best 의 인덱스가 이번 batch 에 속하는가"로
한다. 매 tell 마다 과거 점수까지 재정규화되는 우리 구조에서, 서로 다른
시점의 점수를 직접 비교하면 스케일러 변화가 개선/정체 판정을 오염시킨다 —
같은 스케일러 아래에서의 argmax 위치 비교는 이 문제가 없다.

## 4. 앙상블 UCB — novelty 휴리스틱의 대체

xgb_surrogate 는 탐험을 "가장 가까운 기관측점까지의 해밍 거리" (novelty)
로 유도했다. xgb_tr 은 서로 다른 random_state/부표본(subsample 0.7)으로
학습한 XGB `n_ensemble`(4)개의 예측 표준편차 σ 를 모델 불확실성으로 쓴다:

```
acquisition(x) = mean_i μ_i(x) + κ · std_i μ_i(x)
```

- novelty 는 "아직 안 가본 곳"만 알지만, σ 는 "모델들이 서로 동의하지 않는
  곳"(데이터가 부족하거나 지형이 험한 곳)을 가리킨다 — 원칙적 탐험 신호.
- trust region 이 이미 탐색을 국소화하므로, κ 는 1.0 정도의 보수적 값으로
  둔다 (전역 탐험은 restart 가 담당한다는 역할 분리).

## 5. 하이퍼파라미터

| 파라미터 | 기본값 | 역할 / 튜닝 가이드 |
|---|---|---|
| `n_startup` | 30 | 랜덤 시동 관측 수. 스케일러/모델 안정화 하한 |
| `batch_size` | 4 | ask 당 제안 수. 작을수록 적응 빠름, 모델 재학습 빈도와 균형 |
| `n_candidates` | 300 | TR 내 후보 풀 크기 |
| `kappa` | 1.0 | UCB 탐험 계수. TR 이 좁을수록 낮춰도 됨 |
| `r_init` | 8 | 초기 해밍 반경 (30컬럼의 ~1/4) |
| `r_max` | 15 | 반경 상한 (컬럼 수의 절반 — 그 이상이면 사실상 전역) |
| `succ_tol` | 3 | 연속 개선 이 횟수 → 반경 2배 |
| `fail_tol` | 8 | 연속 정체 이 횟수(batch 단위) → 반경 절반 |
| `refit_interval` | 4 | tell 이 횟수마다 앙상블 재학습. 히스토리가 커지면 `max(refit_interval, N/2000)` 으로 자동 확대 |
| `n_ensemble` | 4 | 앙상블 크기. 키우면 σ 추정 안정, 비용 선형 증가 |
| `max_train_size` | 4000 | 학습 표본 상한 (elite 절반 + 랜덤 절반 서브샘플). 800 evals 에서는 no-op, 100K 급 장기 실행용 |

예산 800 기준 감각: batch 4 → 200 tell. fail_tol 8 이면 정체 시 반경이
8→4→2→1 로 붕괴해 restart 까지 ~32 batch(128 evals) — 한 run 에서 3~5개
trajectory 를 시도할 수 있는 배분이다.

## 6. 실측 성적 (이 리포지토리)

| 실험 | 결과 |
|---|---|
| xgb_surrogate 와 1:1 (bm3_hard, 5 seeds, 동일 pooled scaler) | mean 0.619 vs 0.556, std **0.050 vs 0.093**, 최악 seed **0.573 vs 0.474** |
| 토너먼트 (11종) | 3단계 전부 1위 → 챔피언 |
| top-3 confirmation | 0.768 / 0.750 / 0.491 (전신: 0.702 / 0.511 / 0.260) |
| 요인 격자 (bm3 복합 / bm4 기만 / bm5 상호작용, 5 seeds) | **전부 1위**: 0.804±0.058 / 0.781±0.039 / 0.747±0.080 |

요인 격자에서 전천후인 이유 — 세 난이도 요인 각각에 대응하는 부품이 있다:

- 교차-블록 **상호작용** → 전역 XGB 모델이 학습 (bm5)
- **기만**(trap 골짜기) → 후보의 30% 랜덤 레벨 점프 + restart 가 골짜기를
  건너뜀 (bm4)
- **rugged/노이즈** → 앙상블 σ 가 험한 곳을 식별, TR 이 검증된 지역에 집중
  (bm3)

## 7. 한계와 개선 여지

- **여전히 남은 꼬리 위험**: top-3 의 #3(0.491)처럼 나쁜 trajectory 로
  예산을 다 쓰는 seed 가 드물게 나온다. restart 시점의 남은 예산이 부족하면
  회복 불가 — 예산 인지형 restart(남은 예산 < 임계치면 best trajectory 로
  복귀) 가 다음 개선 후보.
- ~~100K 스케일 부적합~~ → 해결됨: 학습 표본 상한(`max_train_size`) +
  적응형 refit 간격 + 기관측 set 증분 유지로 100K evals ≈ 30~40분.
  true-optimum 계산에도 사용 가능하다 (실측: 10K evals 168초).
- **블록 구조 미활용**: TR 은 해밍 거리만 쓴다. 블록 단위 반경(common 은
  보수적으로, set 은 공격적으로)을 두는 구조 인지형 TR 은 시도해 볼 가치가
  있다.

## 8. 참고문헌

- X. Wan et al. (2021). *Think Global and Act Local: Bayesian Optimisation
  over High-Dimensional Categorical and Mixed Search Spaces* (CASMOPOLITAN).
  ICML. https://arxiv.org/abs/2102.07188
- L. Papenmeier et al. (2023). *Bounce: Reliable High-Dimensional Bayesian
  Optimization for Combinatorial and Mixed Spaces.* NeurIPS.
  https://arxiv.org/abs/2307.00618
- D. Eriksson et al. (2019). *Scalable Global Optimization via Local
  Bayesian Optimization* (TuRBO). NeurIPS. — trust region BO 의 원형.
- M. Lindauer et al. (2022). *SMAC3.* JMLR. https://arxiv.org/abs/2109.09831
  — tree 계열 surrogate + 앙상블 불확실성 노선.
- 관련 문서: `doc/algo/research_better_algorithms.md` (선정 배경),
  `doc/algo/chow_liu_eda.md` (대조군 — 생성 모델 접근의 실패 사례).
