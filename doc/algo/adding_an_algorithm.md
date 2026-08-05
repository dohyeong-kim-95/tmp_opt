# 알고리즘 추가하기 — 함수 하나로

새 최적화 알고리즘을 넣는 데 필요한 건 **함수 하나**다. 클래스도, 상속도,
`super()` 도 없다. 복사해서 시작할 파일: [`algo_template.py`](../../algo_template.py)

```python
from optimizer import simple_algorithm

@simple_algorithm("my_hill", state={"cur": None}, step=2)
def my_hill(data):
    X, s, st, rng, ctx = (data["X"], data["scores"], data["state"],
                          data["rng"], data["ctx"])
    if len(X) == 0:
        return ctx.sample(rng, 1)                 # 첫 호출 — 히스토리가 비었다
    i = len(s) - 1                                 # 방금 평가된 점
    if st["cur"] is None or s[i] > s[st["cur"]]:
        st["cur"] = i
    return [ctx.mutate(rng, X[st["cur"]], rate=data["cfg"]["step"] / ctx.space.n_cols)]
```

**인자 하나, 반환 하나.** `examples/owner_minimal.py` 와 같은 모양이고, 거기서
직접 쓰던 루프를 프레임워크가 돌려주는 것만 다르다. 인자를 dict 대신 이름으로
받고 싶으면 `@algorithm` 을 쓴다 (§2 참조 — 계약은 동일하다).

```bash
python runner.py --plugin algo_template --optimizer my_hill --surface-data obs.jsonl
```

---

## 1. ask-tell 이 뭔가 (한 문단)

**최적화기가 측정 함수를 호출하지 않는다.** 호출자가 측정하고, 결과를
최적화기에 알려준다. 흔한 반대쪽 API 는 `optimize(f, budget=800)` 처럼
최적화기가 루프를 소유하는데, 그러면 중간 저장·재개가 안 되고 측정을 다른
기계에서 못 하며 사람이 끼어들 수 없다. 그래서 루프를 밖으로 꺼냈다:

```python
x = opt.ask(state)          # "다음에 뭘 재볼까?"
y = measure(x)              # 측정은 호출자가
state = opt.tell(state, x, y)   # "이렇게 나왔다"
```

`@algorithm` 은 이 둘을 **한 함수로 합쳐서** 보여준다. 함수는 라운드마다 한 번
불리고, 그때까지의 전체 히스토리를 받아 다음 X 들을 돌려준다. 즉 인자가 tell,
반환값이 ask 다.

## 2. 함수 계약

`@simple_algorithm` 은 인자를 `data` dict 하나로 준다. 키는 아래 표와 같다
(`data["X"]`, `data["scores"]`, …). `@algorithm` 을 쓰면 같은 것들을 **선언한
것만** 이름으로 받는다 — 필요 없으면 안 적으면 된다. 둘은 완전히 같은 계약이고
아무 차이 없이 섞어 쓸 수 있다.

| 인자 | 형상 | 내용 |
|---|---|---|
| `X` | (n, 30) int64 | 지금까지 평가된 X (평가 순서) |
| `Y` | (n, 6) float | 대응 raw 관측 (열 = y11,y12,y13,y21,y22,y23) |
| `scores` | (n,) float | **[0,1] 점수, 클수록 좋음** ← 보통 이걸 쓴다 |
| `state` | dict | 라운드 간 기억 (`@algorithm(state=...)` 이 초기값) |
| `rng` | Generator | 난수 — **반드시 이것만** |
| `ctx` | Ctx | `ctx.space` / `ctx.budget` / `ctx.cfg[하이퍼파라미터]` + 헬퍼 |

`ctx` 헬퍼: `ctx.sample(rng, n)` 균등 랜덤 · `ctx.mutate(rng, x, rate)` ordinal
이웃(±1 위주, 가끔 점프, 범위 클램프 포함) · `ctx.top_k(X, scores, k)` 상위 k 개 ·
`ctx.clip(x)` 범위 클램프.

반환은 `xs` 또는 `(xs, state)`. `xs` 는 `(30,)` 하나여도, 리스트여도, `(b, 30)`
배열이어도 된다. `state` 는 dict 라 제자리 수정도 반영되므로 굳이 반환하지 않아도 된다.

### `scores` 를 직접 만들지 말 것

목적이 6개고 은닉 스케일이 5~6자릿수 차이 난다. 그래서 정규화(robust p5–p95) →
sense 통일(최소화 목적 뒤집기) → scalarization(chebyshev 기본) 이 필요한데,
이건 이미 되어서 `scores` 로 온다. 알고리즘마다 다시 구현하면 **알고리즘 간
비교가 조용히 무의미해진다** (실행이 깨지지 않으니 아무도 눈치채지 못한다).
원시값이 꼭 필요하면 `Y` 를 쓰되, 순위 판단은 `scores` 로 하라.

점수 파이프라인의 단일 진입점은 `get_scores` 다. `OptimizerBase.tell` 도 이걸
쓰므로, 밖에서 직접 루프를 돌 때 같은 함수를 부르면 결과가 **정확히 일치**한다
(3개 scorer 전부 오차 0으로 검증):

```python
from optimizer import convert_y_raw, get_scores

Y = convert_y_raw(calc.evaluate(X))
scores = get_scores(Y)                       # 기본 chebyshev
scores = get_scores(Y, "owa", k=3)           # scalarization·파라미터 변경
```

## 3. 규칙 세 개

**(1) 첫 호출은 히스토리가 비어 있다.** `len(X) == 0` 을 반드시 처리하라.
초기점은 보통 `ctx.sample(rng, n)`.

**(2) `state` 에는 pickle 가능한 값만.** 프로세스 분리 실행(`--serve-step`)은
매 스텝이 **새 프로세스**라 `state.pkl` 이 스텝 간 유일한 기억이다. 모델 객체는
pickle 되면 넣어도 되고(기존 `xgb_surrogate` 가 그렇게 한다), 파일 핸들·락·
제너레이터는 안 된다.

> 그래서 제너레이터 스타일(`y = yield x`)은 쓸 수 없다. 읽기엔 제일 좋지만
> 실행 상태(프로그램 카운터·지역변수)가 직렬화되지 않아 재개가 불가능하다.

**(3) 난수는 `rng` 만.** 전역 `np.random` 을 쓰면 중단·재개 시 궤적이 달라진다.
`algo_template.py` 의 자가 점검이 이걸 실제로 잡아낸다.

### state 에 뭘 넣나 — 판단 기준

> **히스토리(X, Y, scores)만 보고 복원할 수 있나?** 있으면 state 에 넣지 마라.

기존 11종을 조사하면 이렇게 갈린다:

| state 불필요 | state 필요 (그리고 무엇이) |
|---|---|
| `random`, `tpe`, `eda_tree` | `sa` 현재 해 · `pso` 속도 · `aco` 페로몬 · `xgb_tr` 신뢰영역 반경/카운터 · `gomea_block` population |

예: SA 의 "현재 해" 는 히스토리에 안 적혀 있다 — 어떤 제안을 수락했는지는
주사위 결과에 달렸기 때문이다. 이걸 `best` 로 대체하면 언덕을 내려갈 수 없어
SA 가 아니라 언덕오르기가 된다 (실측: best 0.785 vs 0.968).

## 4. 배치 크기

`ask` 는 원하는 만큼 돌려줄 수 있다. SA 처럼 1개, GA 처럼 세대 단위 4~20개.
runner 가 예산 초과분을 잘라내므로 마지막 라운드는 요청보다 적게 평가될 수 있다
— 배치 전부가 평가된다고 가정하지 말고, 다음 호출 때 `len(X)` 로 확인하라.

## 5. 외부 파일에 두기 (`--plugin`)

기여 코드를 `optimizer.py` 에 넣을 필요는 없다. 별도 모듈에 두고:

```bash
python runner.py --plugin my_algos --optimizer my_hill --surface-data obs.jsonl
python runner.py --plugin my_algos --optimizer my_hill --surface-data obs.jsonl --separate xchg/
```

프로세스 분리 실행에서도 `--plugin` 이 서브프로세스로 전달되어 매 스텝 다시
import 된다. 플러그인을 로드하지 않으면 `OPTIMIZERS` 는 기존 11종 그대로다
(완전성 게이트가 깨지지 않는다).

## 6. 스스로 점검하기

`algo_template.py` 의 `__main__` 이 세 가지를 검사한다. 새 알고리즘에도 그대로
복사해 쓰면 된다:

1. ask-tell 사이클이 도는가 + `state` 가 pickle 되는가
2. **중단·재개 궤적이 무중단과 동일한가** (규칙 3 위반이 여기서 잡힌다)
3. 실제 반응표면에서 예산을 완주하는가

```bash
python algo_template.py
```

## 7. OptimizerBase 없이 직접 루프를 돌아도 되나

된다. `OptimizerBase` 는 네 가지를 대신해 줄 뿐이고, 앞의 셋은 직접 해도 된다:

| 하는 일 | 직접 하면 |
|---|---|
| 마스크 → 숫자 6개 | `convert_y_raw(raw)` 를 부르면 끝 |
| 숫자 → 점수 | `get_scores(Y)` 를 부르면 끝 (재구현 금지) |
| 히스토리 누적 | `np.vstack` 몇 줄 (매번 O(N) 복사라는 점만 유의) |
| **체크포인트 호환 state** | ← 이건 `OptimizerBase` 가 필요하다 |

즉 **한 알고리즘을 한 번 굴려보는 실험**이면 `OptimizerBase` 없이 100줄이면
충분하다. 필요해지는 시점은 두 가지다: 중단·재개(또는 프로세스 분리)를 쓸 때,
그리고 여러 알고리즘을 **같은 조건에서 비교**할 때.

## 8. 클래스로 쓰고 싶다면

`OptimizerBase` 를 상속해도 된다 — 함수 스타일은 그 위에 얹은 얇은 어댑터일
뿐이고 둘은 완전히 같은 계약을 만족한다.

기존 11종 중 **9종이 함수**이고, 클래스로 남은 것은 `blockwise_coord`(phase/
sweep 커서 + 평가 캐시)와 `gomea_block`(FOS 순회 상태) 둘뿐이다. 이 둘은
상태 키가 10개 이상이라 함수 하나에 넣으면 오히려 읽기 나쁘다 — 그게 클래스를
고를 기준이다.
그 경우 구현할 것은 `init_state(seed)` / `ask(state)` / `_update(state, X, scores)`
셋이고, **인스턴스에 탐색 상태를 저장하지 말 것** (재개가 깨진다).
