# 반응표면 기반 최적화 검증 환경

30 ordinal 컬럼(≈10^15 조합), 다목적 6개. 실제 TEST 가 비싸므로 **이미 확보한
관측으로 세운 반응표면**을 측정기 대역으로 두고 그 위에서 알고리즘을 돌려본다.

목적은 성능 비교가 아니라 두 가지 검증이다 — 알고리즘이 **완주하는가**,
그리고 **근거 없는 X 를 추천하지 않는가**.

```bash
pip install numpy xgboost              # xgboost 는 xgb_tr 에만 필요

python make_dataset.py --out obs.jsonl        # 1. 관측 만들기
python accept.py                              # 2. 완료조건 4항목 판정
python run.py --algo xgb_tr --obs obs.jsonl --budget 800   # 3. 알고리즘 구동
```

## 파일

| 파일 | 역할 |
|---|---|
| `space.py` | 탐색 공간 표준 명세 (signed 정수 범위, 블록 구조) |
| `calculator.py` | **반응표면** `SurfaceCalculator` + `obs.jsonl` 입출력 |
| `algo.py` | 측정→점수(`convert_y_raw`/`get_scores`) + 알고리즘 |
| `run.py` | 루프 (algo → 측정 → obs.jsonl append) |
| `make_dataset.py` | 관측 설계(one-hot/무작위/반복) + 시뮬레이션 장치 |
| `accept.py` | 완료조건 4항목 (참값 대조) |
| `lesson_learned.md` | 합성 벤치마크 시절에서 건진 것 |

## 알고리즘 추가하기

함수 하나다. 클래스도 상속도 없다:

```python
from algo import algorithm, sample, mutate, top_k

@algorithm("my_algo", state={"cur": None}, step=2)
def my_algo(data):
    X, s, st = data["X"], data["scores"], data["state"]
    if len(X) == 0:
        return sample(data["space"], data["rng"], 1)     # 첫 호출 — 히스토리가 비었다
    i = len(s) - 1                                        # 방금 평가된 점
    if st["cur"] is None or s[i] > s[st["cur"]]:
        st["cur"] = i
    return [mutate(data["space"], data["rng"], X[st["cur"]],
                   rate=data["cfg"]["step"] / data["space"].n_cols)]
```

`data` 에 `X`(n,30) · `Y`(n,6) · `scores`(n,) · `state` · `rng` · `space` ·
`cfg` · `budget` 이 들어 있다. **`scores` 는 이미 정규화·sense 통일·
scalarization 이 끝난 [0,1] 값**이라 직접 만들면 안 된다 — 알고리즘마다 다르게
만들면 비교가 조용히 깨진다.

규칙 셋: ① 첫 호출은 히스토리가 비어 있다 ② `state` 에는 pickle 가능한 값만
③ 난수는 `rng` 만 (전역 `np.random` 을 쓰면 재개 궤적이 달라진다).

`state` 에 뭘 넣나 — **히스토리만 보고 복원할 수 있으면 넣지 마라.** SA 의
'현재 해'나 신뢰영역 반경처럼 관측에 안 적히는 것만 넣는다.

## 루프는 측정하는 쪽이 소유한다 (ask-and-tell)

optimizer 가 측정 함수를 호출하지 않는다. 측정하는 쪽이 "다음에 뭘 재볼까"를
묻는다. 이 인터페이스의 표준 명칭이 **ask-and-tell** 이다 (CMA-ES 유래;
Optuna·Nevergrad·scikit-optimize·Ax 가 같은 용어를 쓴다). 반대쪽 —
`scipy.optimize.minimize(f, ...)` 처럼 optimizer 가 루프를 소유하고 당신 함수를
콜백으로 부르는 방식 — 은 보통 **objective-function interface** 라 부르고
구조적으로는 제어 역전이다. 측정이 이 프로세스 밖에 있으면 못 쓴다.

측정이 밖에 있으면 `run.py` 대신 `algo.propose` 를 직접 부르면 된다:

```python
from algo import ALGORITHMS, propose, convert_y_raw
from calculator import load_observations, save_observations

fn, state = ALGORITHMS["xgb_tr"], ALGORITHMS["xgb_tr"].init_state()
d = load_observations("obs.jsonl")
for x in propose(fn, d["X"], d["Y"], state, rng, space, budget):
    y_raw = 어딘가에서_측정(x)                      # ← 루프의 주인은 당신
    save_observations("obs.jsonl", x[None, :], convert_y_raw(y_raw), append=True)
```

**관측 파일이 곧 상태다.** 중단해도 파일만 있으면 이어서 돌고, 사람이 손으로
한 줄 추가해도 즉시 반영된다.

## 관측 파일 (`obs.jsonl`)

append-only 텍스트. 한 줄이 관측 하나이고 마스크 원형까지 그 줄에 들어간다:

```json
{"i":0,"block":"one_hot","X":[...30개...],"Y":[...6개...],
 "mask1":{"shape":[128,128],"runs":[[55,63,2],[56,62,5],...]},"mask2":{...}}
```

마스크는 행 단위 run-length(`[row, col0, len]`) — 무손실이고, blob 은 행마다
연속 구간이 한두 개뿐이라 조밀하며(111점 242KB; base64 비트팩이면 634KB)
무엇보다 눈으로 읽힌다.

## 반응표면이 하는 일

예측 모델이 아니라 **재생기 + 국소 보간기**다. 관측점은 실측을 그대로 내고,
관측에서 멀면 값을 지어내지 않고 "데이터 없음"으로 표시한다. 판정은 **목적
그룹별로 독립**이다 (y1x 는 데이터가 있고 y2x 는 없는 X 가 존재한다):

| 상태 | 조건 | 값 |
|---|---|---|
| `exact` | 30컬럼 전부 일치하는 관측이 있다 | 그 실측값 그대로 (마스크는 바이트 그대로) |
| `exact_block` | 해당 목적의 의존 블록이 전부 일치 | 그 관측들을 합성 |
| `interp` | 최근접 거리 ≤ `d_gate`, 이웃 불일치 ≤ `spread_gate` | k최근접 역거리가중 보간 |
| `no_data` | 그 외 | 지어내지 않음 (`flag`/`pessimistic`/`strict`) |

거리는 **블록 제한 정규화 해밍** — 목적이 의존하지 않는 컬럼의 차이는 세지
않는다. 근거가 여럿이면 스칼라는 가중평균, 마스크는 **blob 형상 보간**
(무게중심 + 각도별 반경을 가중평균 후 재래스터화).

`neighbors()`/`explain()` 으로 근거 관측을 되짚고, `report()` 로 알고리즘이
커버리지를 얼마나 벗어났는지, `loo_report()` 로 보간의 정직도를 본다.

## 한계

관측이 소량이면 자유 탐색의 `no_data_rate` 는 1.00 에 가깝다 (10^15 공간에
관측 수백). 이건 고장이 아니라 검증 2의 답 자체다 — 다만 그래서 이 표면을
**최적화 성능 비교에 쓰면 안 된다.** 지형 대부분이 근거 없는 채움값이다.
