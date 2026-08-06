# Chow-Liu 의존성 트리 EDA (MIMIC/COMIT 계열)

이 문서는 `optimizer.py` 의 `eda_tree` (`ChowLiuTreeEDAOptimizer`) 가 구현하는
알고리즘의 배경, 원리, 설계 결정을 소개한다.

---

## 1. EDA(Estimation of Distribution Algorithm)란

GA 가 "좋은 해들을 **교배**해서" 다음 세대를 만든다면, EDA 는 "좋은 해들의
**확률분포를 추정**해서" 다음 세대를 그 분포로부터 샘플링한다:

```
반복:
  1. 현재까지의 관측 중 상위 γ (elite) 를 고른다
  2. elite 로부터 확률모형 P̂(x) 를 추정한다
  3. P̂(x) 에서 새 후보들을 샘플링해 평가한다
```

crossover/mutation 같은 유전 연산자를 명시적 통계 모형으로 대체한 것이
핵심이며, 어떤 모형을 쓰느냐에 따라 계열이 나뉜다.

### 모형 복잡도의 사다리

| 계열 | 모형 | 표현 가능한 구조 | 대표 알고리즘 |
|---|---|---|---|
| univariate | P̂(x) = Π P(x_i) | 주효과만 | UMDA, PBIL, cGA |
| **pairwise** | 체인/트리 인수분해 | **변수 쌍 상호작용** | **MIMIC, COMIT (Chow-Liu)** |
| multivariate | 베이지안 네트워크 등 | 고차 상호작용 | BOA, hBOA, EBNA |

- **MIMIC** (De Bonet et al., 1997): 조건부 엔트로피가 작은 순서로 변수를
  이어 붙인 **체인** P(x_1)·P(x_2|x_1)·…·P(x_n|x_{n-1}) 을 탐욕적으로 만든다.
- **COMIT** (Baluja & Davies, 1997): 체인을 **트리**로 일반화 —
  최적 트리를 Chow-Liu 알고리즘으로 찾는다. 본 구현이 이 방식이다.
- 참고로 본 리포지토리의 **ACO(페로몬 테이블)와 TPE 는 사실상 univariate
  EDA** 에 해당한다. `eda_tree` 는 pool 에서 유일하게 "변수 간 결합을
  **샘플링 분포 자체로** 표현"하는 method 다 (XGB surrogate 도 상호작용을
  배우지만, 그건 예측 모형이지 생성 모형이 아니라서 후보 생성이 mutation 에
  의존한다).

---

## 2. Chow-Liu 트리: 원리

### 정리 (Chow & Liu, 1968)

트리 구조로 인수분해되는 분포

```
P_T(x) = P(x_root) · Π_{(u→v)∈T} P(x_v | x_u)
```

중에서 **참 분포와의 KL divergence 를 최소화**하는 트리 T 는, 변수 쌍의
**상호정보량(mutual information) I(X_u; X_v) 의 합을 최대화하는 신장
트리**다. 즉:

```
T* = argmax_T Σ_{(u,v)∈T} I(X_u; X_v)
```

증명 스케치: KL(P‖P_T) 를 전개하면 `-Σ_(u,v)∈T I(u;v) + (T와 무관한 항)`
이 되므로, MI 합 최대화 = KL 최소화. 따라서 "pairwise 정보만 쓸 수 있다면
이보다 나은 인수분해는 없다"는 **최적성 보장**이 있다 — MIMIC 의 탐욕적
체인보다 이론적으로 우월한 지점.

### 알고리즘 (한 세대)

```
입력: elite 표본 E (상위 γ), 컬럼 cardinality
1. 모든 컬럼 쌍 (u, v) 에 대해 joint 히스토그램 → I(u; v) 추정
   ( I(u;v) = Σ p(a,b)·log[ p(a,b) / (p(a)p(b)) ] , Laplace 평활 )
2. I 를 간선 가중치로 하는 완전그래프에서 최대 신장 트리 (Prim/Kruskal)
3. 임의 루트에서 BFS 로 방향을 정해 조건부 확률표 P(x_v | x_u) 추정
4. 루트의 marginal 부터 트리 순서대로 조건부 샘플링 → 새 batch
```

복잡도: 컬럼 n=30, elite 크기 E, 최대 cardinality k=30 기준
- MI 추정: O(n²·E) — 쌍 435개 × elite 스캔 (지배적 비용)
- MST: O(n²), 샘플링: O(batch·n)

E 를 상한(우리 구현은 400)으로 캡하면 히스토리가 커져도 세대당 비용이
상수로 유지된다 (100K 스케일 true-optimum 실행에도 사용 가능).

---

## 3. 왜 이 문제에 맞는가

본 벤치마크(특히 BM3)의 구조와 정확히 맞물린다:

1. **이산 ordinal + 작은 cardinality (2~30)**: joint 히스토그램 기반 MI
   추정이 값싸고 표본효율적이다. 연속 변수라면 필요한 이산화/커널 추정이
   여기서는 공짜다.
2. **BM3 의 교차-블록 결합 저격**: BM3 는 `cos(π(u_common − u_set))` 항으로
   "common 값이 바뀌면 set 쪽 최적이 이동"하게 설계되어 있다 (`doc` 참조:
   README 의 BM3 제작 상세). univariate 모형은 이걸 혼합분포로 뭉개지만,
   Chow-Liu 트리는 (common, set) 쌍의 MI 가 크게 잡히면 그 간선을 트리에
   넣어 **조건부로 함께 샘플링**한다.
3. **블록 구조와의 정합**: `set1 ⫫ set2 | common` 이므로 참 의존성 그래프가
   common 을 허브로 하는 성긴 구조 — 트리 근사가 크게 무리하지 않는 문제다.

### 한계 (정직하게)

- **트리는 쌍까지만**: 3변수 이상의 고차 상호작용(예: common 두 개가 함께
  set 하나를 결정)은 표현 불가. 그건 BOA 급 모형의 영역이다.
- **deceptive 성분에는 여전히 취약**: elite 통계가 가짜 오르막(u≈0.8)에
  지배되면 트리도 그쪽으로 질량을 모은다. 상호작용 표현력은 이 문제를
  직접 해결하지 못한다.
- **초기 트리 불안정**: elite 가 적을 때 MI 추정 노이즈로 트리가 세대마다
  크게 바뀔 수 있다 — startup 랜덤 구간과 Laplace 평활로 완화한다.
- **다양성 붕괴**: EDA 공통 문제. 탐험 하한(uniform 혼합)으로 완화한다.

---

## 4. 본 구현 (`eda_tree`) 의 설계 결정

ask-tell stateless 구조(§`optimizer.py` 모듈 docstring)에 맞춘 결정들:

| 결정 | 내용 | 이유 |
|---|---|---|
| elite 선정 | 매 ask 마다 **전체 히스토리** 상위 γ=25% (하한 30, 상한 400) | 세대 개념 없이 히스토리 재정규화(매 tell 점수 갱신)와 자연 정합. 상한 400 은 100K 스케일 대비 |
| MI/조건부 평활 | joint 히스토그램에 Laplace α=0.5 | 미관측 조합의 확률 0 방지, 초기 MI 노이즈 완화 |
| 탐험 하한 | 샘플링 분포 = (1−ε)·모형 + ε·uniform, ε=0.05 | 확률 포화(0/1)로 인한 조기 수렴 방지 — PBIL 의 mutation, ACO 의 explore_floor 와 같은 역할 |
| 루트 선택 | MI 합(트리 간선 기준)이 가장 큰 노드 | 정보가 많은 허브(대개 common 컬럼)에서 샘플링을 시작 |
| startup | 처음 40 관측은 균등 랜덤 | MI 추정에 최소한의 무편향 표본 확보 + 초기 스케일러 안정화 |
| 상태 | 히스토리 스냅샷 + RNG 만 | 모형은 매 ask 재구축 (TPE 와 동일한 순수 함수형 접근 — stateless 에 최적) |

ordinal smoothing(TPE 의 이웃-레벨 커널)은 조건부 확률표에는 적용하지
않았다 — 2차원 표에 커널을 두 번 접으면 상호작용 신호 자체가 흐려질 수
있어, 1차 구현은 Laplace 만 쓰고 필요성이 확인되면 추가한다.

---

## 5. 참고문헌

- C. Chow, C. Liu (1968). *Approximating discrete probability distributions
  with dependence trees.* IEEE Trans. Information Theory.
- J. De Bonet, C. Isbell, P. Viola (1997). *MIMIC: Finding optima by
  estimating probability densities.* NIPS.
- S. Baluja, S. Davies (1997). *Using optimal dependency-trees for
  combinatorial optimization (COMIT).* CMU-CS-97-107.
- M. Pelikan, D. Goldberg, E. Cantú-Paz (1999). *BOA: The Bayesian
  optimization algorithm.* GECCO. — 트리 너머(고차 상호작용)가 필요할 때.
- P. Larrañaga, J. Lozano (2002). *Estimation of Distribution Algorithms:
  A New Tool for Evolutionary Computation.* — EDA 전반의 표준 교과서.
