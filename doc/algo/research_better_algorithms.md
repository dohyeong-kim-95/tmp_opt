# 리서치: 이 문제를 더 잘 풀 수 있는 알고리즘들

토너먼트 실측 결과(9종, 800 evals)를 근거로, 문제 프로파일에 맞는 상위
알고리즘 계열을 조사해 우선순위를 매긴다.

## 0. 문제 프로파일과 실측 근거 요약

| 특성 | 시사점 |
|---|---|
| 30 ordinal 컬럼 (card 2~30), ~10^15 | GP보다 트리/히스토그램 계열 모델이 자연스러움 |
| 블록 구조 기지(旣知): set1 ⫫ set2 \| common | 구조를 "배우지 말고 주입"할 여지 — 학습 예산 절약 |
| BM3: 교차-블록 결합 + deceptive + rugged | 상호작용 모델링 필수, 국소 탐색만으로는 가짜 정상에 갇힘 |
| 노이즈 5%, 병렬 불가, 800 evals | 표본효율 + 노이즈 강건성이 승부처 |
| 실측: xgb_surrogate가 hard 1위 (0.696±0.102) | **상호작용을 잡는 회귀 surrogate 가 유효** — 단, init 분산이 큼 |
| 실측: eda_tree(Chow-Liu) 최하위권 | 분포에서 구조를 "배우는" 건 이 예산에선 실패 — 신호 대비 표본 부족 |
| 실측: ACO 안정적 2위권 (top-3 재현성 우수) | univariate 증분 갱신의 강건함 — 분산 낮은 baseline |

핵심 교훈 두 개:
1. **상호작용은 회귀(surrogate)로 잡는 게 이 예산에서 통했고, 생성 모델
   (EDA)로 배우는 건 통하지 않았다.**
2. **챔피언(xgb)의 약점은 성능 상단이 아니라 분산이다** (init seed 에 따라
   0.26~0.70). 개선 방향은 "더 높게"보다 "더 안정적으로".

---

## 1. 계열별 후보

### A. Linkage 기반 진화 알고리즘 — GOMEA / LTGA / DSMGA-II / hBOA ★추천 1순위

[GOMEA(Gene-pool Optimal Mixing EA)](https://github.com/CWI-EvolutionaryIntelligence/GOMEA)
계열은 변수 의존성 모델(FOS, Family of Subsets)에 따라 **부분해를 통째로
donor 에서 복사해보고 즉시 채택/롤백**하는 optimal mixing 을 쓴다.
[concatenated trap(=deceptive) 벤치마크에서 런타임이 증명된](https://arxiv.org/abs/2407.08335)
사실상 유일한 계열 — **deceptive 성분을 정면으로 겨냥**한다.

우리 문제에 특히 맞는 이유:
- FOS 를 학습할 필요가 없다. **블록 구조를 알고 있으므로 FOS =
  {common, set1, set2, (개별 컬럼들)} 로 고정 주입**하면 linkage 학습
  예산이 0이 된다 (eda_tree 실패의 정확한 해독제).
- optimal mixing 의 "블록 통째 이식 + 즉시 검증"은 1-hop 스윕
  (blockwise_coord)이 못 넘는 가짜 정상을 블록 단위 점프로 넘을 수 있다.
- 평가 1회 단위로 진행되어 ask-tell(batch=1)에 자연스럽게 맞는다.
- 주의: 노이즈 하에서 "즉시 채택/롤백" 비교가 흔들릴 수 있음 — 비교 마진
  또는 racing(계열 D)과 결합 권장.
- 구현 비용: **낮음** (mutation 대신 블록 복사 + 수락 판정이 전부).
- 참고: [Parameterless GOMEA](https://arxiv.org/abs/2109.05259) (population
  sizing 자동화), hBOA(베이지안 네트워크 + niching, deceptive 대응의 원조
  — 단 구현 무거움).

### B. 조합공간 Bayesian Optimization — CASMOPOLITAN / Bounce / BOCS / COMBO ★추천 2순위(경량 이식)

전용 조합 BO 들은 PestControl(25 categorical — 우리와 유사한 규모)에서
검증되어 있다. [Bounce(2023)](https://arxiv.org/abs/2307.00618)가 현재
이 벤치마크에서 최상위이고, trust-region 기반
CASMOPOLITAN 이 그 다음, [BODi 는 카테고리 순서를 섞으면 성능이 급락](https://arxiv.org/html/2307.00618v2)
한다고 보고된다. [고차원 이산 BO 서베이/벤치마크(2024)](https://arxiv.org/html/2406.04739v2)도 참고.

- 공통 아이디어: **GP surrogate + Hamming 커널 + trust region**(현재 best
  주변 해밍 반경 안에서만 acquisition 최적화, 정체 시 반경 축소/재시작).
- 우리 프레임워크엔 GP 를 통째로 이식하기보다 **trust-region 메커니즘만
  xgb_surrogate 에 접목**하는 게 실효적이다 ("CASMOPOLITAN-lite"):
  후보 풀을 전역 랜덤 대신 incumbent 중심 해밍 공 안에서 생성하고,
  정체 시 반경을 줄이고 restart. **xgb 의 init 분산 문제를 직접 공략**한다.
- 구현 비용: lite 버전 **낮음**, 원본(GP+MCMC/부분공간) **높음**.

### C. RF/GBM surrogate SMBO 의 정석화 — SMAC 방식 ★추천 2순위(기존 개선)

[SMAC3](https://github.com/automl/SMAC3) 는 RF surrogate + EI 로
categorical/이산 혼합 공간의 표준이며, 벤치마크들에서
[상호작용이 있는 문제에서 TPE 를 능가](https://arxiv.org/pdf/2110.12654)
하고 [고분산(노이즈) 목적에도 TPE 보다 강건](https://arxiv.org/pdf/2109.09831)
하다고 보고된다 — **우리 실측(xgb > tpe on hard)과 정확히 일치**하는 문헌
근거.

현재 xgb_surrogate 의 업그레이드 경로:
- novelty 보너스(휴리스틱) → **앙상블 분산 기반 EI/UCB**(원칙적 불확실성):
  RF 트리별 예측 분산, 또는 XGBoost quantile 회귀로 대체.
- log-EI, 후보 풀에 지역 탐색 추가 등 SMAC 의 세부 설계 차용.
- 구현 비용: **낮음~중간**.

### D. 노이즈 대응 — racing / 적응 재표본 (알고리즘 횡단 개선)

[F-Race 계열](https://link.springer.com/chapter/10.1007/978-3-540-75514-2_9)
racing 은 "나쁜 후보를 통계적 확신이 서는 즉시 탈락"시키는 프로토콜로,
[비선형 증가 재표본 전략](https://inria.hal.science/inria-00633006v1) 등과
함께 노이즈 하 비교의 표준 도구다.
- 우리 적용처: ① SA/GOMEA 의 수락 판정(마진 도입), ② incumbent 교체 시
  재측정 1회 추가, ③ top-3 confirmation 을 순차 racing 으로 (현재는 고정
  10회). 5% 노이즈에서 잘못된 채택을 줄여 **모든 optimizer 의 분산을
  낮춘다**.
- 구현 비용: **낮음**. 단 재측정은 평가 예산을 소모하므로 trade-off 존재.

### E. Surrogate-assisted EA 하이브리드 — GA/ACO 제안 + 모델 사전선별

[SAEA 서베이(2024)](https://link.springer.com/article/10.1007/s40747-024-01465-5)
가 정리하듯, 조합공간에서 흔한 최상위 레시피는 "population 이 후보를
만들고, surrogate 가 평가 전에 걸러내는" 구조다.
- 우리 적용: GA(또는 ACO)가 세대당 100개 후보를 만들면 xgb 가 상위 20개만
  실평가하도록 선별. **ACO 의 안정성 + xgb 의 상호작용 인지**를 결합 —
  챔피언의 고분산 문제에 대한 population 측 해법.
- 구현 비용: **낮음** (기존 부품 재조합).

### F. 검토했으나 후순위

| 후보 | 후순위 사유 |
|---|---|
| 혼합정수 CMA-ES | 연속 완화+반올림은 PSO 실측 부진과 같은 함정, ordinal 2-level 컬럼에서 정보 손실 |
| MCTS 계열 (LaMCTS 등) | 공간 분할 학습에 예산 소모, 800 evals 에서 근거 부족 |
| LLM 기반 탐색 (예: [LLM-driven SAEA](https://arxiv.org/pdf/2507.02892)) | 흥미롭지만 평가 루프에 외부 의존성·비용 추가, 이 문제 규모에선 과함 |
| BODi | 카테고리 순서 민감성 보고 — ordinal 이라 덜하겠지만 Bounce 가 상위호환 |
| hBOA | deceptive 대응은 최상급이나 구현 무겁고, 블록 FOS 고정 GOMEA 가 예산 효율적 대체 |

---

## 2. 우선순위 제안 (구현 비용 대비 기대효과)

| 순위 | 후보 | 겨냥하는 약점 | 비용 |
|---|---|---|---|
| 1 | **블록-FOS GOMEA** (`gomea_block`) | deceptive + 구조 활용 (eda_tree 의 해독제) | 낮음 |
| 2 | **xgb + trust region + 앙상블 EI** (CASMOPOLITAN-lite/SMAC화) | 챔피언의 init 고분산 | 낮음~중간 |
| 3 | **surrogate-screened GA/ACO** | 안정성과 상단의 결합 | 낮음 |
| 4 | **racing 프로토콜** (수락 판정·confirmation) | 노이즈로 인한 오채택 | 낮음 |
| 5 | Bounce/CASMOPOLITAN 원본 이식 | 표본효율 상한 갱신 | 높음 |

1~3은 서로 배타적이지 않다 — 특히 1과 2는 탐색 철학이 달라(모델-프리 구조
활용 vs 모델 기반 지역화) 토너먼트에서 상보적인 비교축이 된다.

## 2.5 실측 후기 (구현 후 업데이트)

리서치의 정직한 사후 검증 기록. **1순위였던 블록-FOS GOMEA 는 실측에서
탈락했다** (`gomea_block` 으로 구현, 토너먼트 bm1 9위 / bm3 직접 대결 5 seeds
에서 xgb 0.735 > aco 0.666 > ga 0.611 > gomea 0.509).

원인 분석 — GOMEA 가 이 문제/예산에서 실패하는 구조적 이유:

1. **building block 공급 부족**: optimal mixing 은 population 에 이미 있는
   부분해를 재조합할 뿐 **새 유전자를 만들지 않는다**. population 16으로는
   cardinality 30인 컬럼의 레벨 절반 이상이 유전자 풀에 아예 존재하지
   않는다 — 최적 레벨이 초기 pop 에 없으면 mixing 으로는 영원히 도달 불가.
   (GOMEA 이론이 population sizing 을 전제하는 이유이며, trap 함수 보장도
   subfunction 당 O(2^k) population 을 요구한다 — 800 예산으로 불가능)
2. **accept-not-worse 의 조기 수렴**: 나빠지지 않으면 채택하는 관례가
   population 을 빠르게 평범한 합의점으로 붕괴시키고, 이후 랜덤 이민자는
   member 를 이기지 못해 다양성 주입이 실패한다.
3. 교훈은 eda_tree 와 대칭적이다: eda_tree 는 "구조를 배울 표본"이 부족했고,
   gomea 는 "재조합할 재료"가 부족했다. **800 예산에서는 탐색 재료/모델을
   스스로 생성하는 method(회귀 surrogate, 분포 갱신형 ACO)가 이긴다.**

수정된 우선순위: 2순위(trust-region xgb)와 3순위(surrogate-screened GA/ACO)
를 승격. GOMEA 를 살리려면 mutation 결합 + balanced 초기화 + population 확대가
필요한데, 그 시점에서 사실상 memetic GA 가 되므로 별도 추구 가치는 낮다.

## 3. 출처

- Bounce: https://arxiv.org/abs/2307.00618 (HTML: https://arxiv.org/html/2307.00618v2)
- 고차원 이산 BO 서베이·벤치마크: https://arxiv.org/html/2406.04739v2
- GOMEA 라이브러리: https://github.com/CWI-EvolutionaryIntelligence/GOMEA
- GOMEA trap 함수 런타임 분석: https://arxiv.org/abs/2407.08335
- Parameterless GOMEA: https://arxiv.org/abs/2109.05259
- SMAC3: https://github.com/automl/SMAC3 (논문: https://arxiv.org/pdf/2109.09831)
- DB 튜닝 벤치마크(SMAC vs TPE, 상호작용): https://arxiv.org/pdf/2110.12654
- SAEA 서베이(조합공간): https://link.springer.com/article/10.1007/s40747-024-01465-5
- F-Race 개선: https://link.springer.com/chapter/10.1007/978-3-540-75514-2_9
- racing 재표본 전략: https://inria.hal.science/inria-00633006v1
- LLM-driven SAEA: https://arxiv.org/pdf/2507.02892
