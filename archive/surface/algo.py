"""algo.py — y_raw 를 점수로 바꾸고, 다음 X 를 고른다.

두 부분뿐이다:

  1. 측정 → 점수   `convert_y_raw` (마스크 → 숫자 6개) → `get_scores` (→ 점수 1개)
  2. 알고리즘      `algo(data) -> [다음 X, ...]`

알고리즘은 함수 하나다. 클래스도 상속도 없다:

    @algorithm("my_algo", state={...초기 기억...}, 하이퍼파라미터=기본값)
    def my_algo(data):
        return [다음_x, 다음_x]

`data` 에 필요한 게 전부 들어 있다:

    data["X"]      (n, 30) int64   지금까지 평가된 X (평가 순서)
    data["Y"]      (n, 6)  float   대응 raw 관측
    data["scores"] (n,)    float   [0,1] 점수, 클수록 좋음  ← 보통 이걸 본다
    data["state"]  dict            라운드 간 기억. **제자리 수정**하면 반영된다
    data["rng"]    Generator       난수 — 반드시 이것만 (재현성)
    data["space"]  SearchSpace     x_min / x_max / n_cols / sample / clip
    data["cfg"]    dict            @algorithm 에 넘긴 하이퍼파라미터
    data["budget"] int             총 예산

규칙 셋:
  1. 첫 호출은 히스토리가 비어 있다 (`len(X) == 0`) — 초기점을 낼 것
  2. `state` 에는 pickle 가능한 값만 (중단·재개가 그걸로 된다)
  3. 난수는 `rng` 만 — 전역 np.random 을 쓰면 재개 시 궤적이 달라진다

`state` 에 뭘 넣나: **히스토리만 보고 복원할 수 있으면 넣지 마라.**
SA 의 '현재 해'나 신뢰영역 반경처럼 관측에 안 적히는 것만 넣는다.
"""

from __future__ import annotations

import numpy as np

from space import SearchSpace

#: 목적 이름 / 방향 (+1 = 최대화, -1 = 최소화)
OBJECTIVE_NAMES: tuple[str, ...] = ("y11", "y12", "y13", "y21", "y22", "y23")
OBJECTIVE_SENSES: tuple[int, ...] = (+1, +1, -1, +1, +1, -1)

#: 이름 → 알고리즘. @algorithm 이 채운다
ALGORITHMS: dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# 1. 측정 → 점수
# ──────────────────────────────────────────────────────────────────────────────

def convert_y_raw(y_raw) -> np.ndarray:
    """관측 원형 → (n, 6) float64. **측정 정의는 여기 한 곳뿐이다.**

    mask1 → y11 = max height(열별 True 개수의 최대), y12 = max width(행별)
    mask2 → y21, y22 (같은 방식) · y13/y23 은 그대로 통과.
    개수 기반이라 가장자리 픽셀이 흔들려도 ±수 픽셀로만 반응한다.
    NaN/inf 는 즉시 raise — 조용한 대체 금지.
    """
    if not isinstance(y_raw, dict):
        Y = np.atleast_2d(np.asarray(y_raw, dtype=np.float64))
    else:
        for k in ("mask1", "mask2", "y13", "y23"):
            if k not in y_raw:
                raise ValueError(f"y_raw 에 {k!r} 없음 — keys={list(y_raw)}")
        m1 = np.asarray(y_raw["mask1"], dtype=bool)
        m2 = np.asarray(y_raw["mask2"], dtype=bool)
        if m1.ndim != 3 or m1.shape != m2.shape:
            raise ValueError(f"마스크 형상 불일치 — {m1.shape} vs {m2.shape}")
        y13 = np.asarray(y_raw["y13"], dtype=np.float64).reshape(-1)
        y23 = np.asarray(y_raw["y23"], dtype=np.float64).reshape(-1)
        if not (len(m1) == len(y13) == len(y23)):
            raise ValueError(f"batch 불일치 — mask {len(m1)}, y13 {len(y13)}")
        Y = np.column_stack([m1.sum(axis=1).max(axis=1), m1.sum(axis=2).max(axis=1),
                             y13,
                             m2.sum(axis=1).max(axis=1), m2.sum(axis=2).max(axis=1),
                             y23]).astype(np.float64)
    if Y.ndim != 2 or Y.shape[1] != len(OBJECTIVE_NAMES):
        raise ValueError(f"형상 {Y.shape} — (n, {len(OBJECTIVE_NAMES)}) 여야 함")
    if not np.isfinite(Y).all():
        r, c = map(int, np.argwhere(~np.isfinite(Y))[0])
        raise ValueError(f"y_raw 에 비유한값 — 행 {r} 목적 {c}: {Y[r, c]}")
    return Y


def get_scores(Y: np.ndarray, kind: str = "chebyshev", rho: float = 0.01,
               k: int = 2) -> np.ndarray:
    """(n, 6) raw 관측 → (n,) 점수. 클수록 좋음. **점수 정의는 여기 한 곳뿐이다.**

    값 범위를 모르므로 robust quantile(p5–p95)로 정규화하고, 최소화 목적을
    뒤집어 "1 = best" 로 통일한 뒤 스칼라화한다. 목적 간 스케일이 5~6자릿수
    차이 나므로 이 단계 없이는 비교 자체가 성립하지 않는다.

    알고리즘마다 다시 구현하면 비교가 조용히 무의미해진다 — 실행은 안 깨지고
    아무도 눈치채지 못한다. 그래서 하나뿐이다.

    kind: "chebyshev"(기본, 최악 목적 방어) / "sum"(평균) / "owa"(최악 k개 평균)
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    lo, hi = np.quantile(Y, 0.05, axis=0), np.quantile(Y, 0.95, axis=0)
    z = np.empty_like(Y)
    for j in range(Y.shape[1]):
        if hi[j] - lo[j] < 1e-15:          # 퇴화(전부 같은 값) → 중립
            z[:, j] = 0.5
            continue
        zj = (Y[:, j] - lo[j]) / (hi[j] - lo[j])
        z[:, j] = np.clip(1.0 - zj if OBJECTIVE_SENSES[j] < 0 else zj, 0.0, 1.0)
    if kind == "sum":
        return z.mean(axis=1)
    if kind == "owa":
        return np.sort(z, axis=1)[:, :k].mean(axis=1)
    if kind == "chebyshev":                # augmented Chebyshev (ideal = 1)
        return (z.min(axis=1) + rho * z.mean(axis=1)) / (1.0 + rho)
    raise ValueError(f"알 수 없는 scorer {kind!r}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. 알고리즘 — 헬퍼 + 등록 데코레이터
# ──────────────────────────────────────────────────────────────────────────────

def sample(space: SearchSpace, rng, n: int = 1) -> np.ndarray:
    """균등 랜덤 X (n, 30)."""
    return space.sample(rng, n)


def mutate(space: SearchSpace, rng, x, rate: float = 1.0 / 30) -> np.ndarray:
    """ordinal 이웃: 대체로 ±1 스텝, 가끔 랜덤 점프. 최소 1컬럼은 변한다."""
    x = np.asarray(x, dtype=np.int64).copy()
    m = rng.random(space.n_cols) < rate
    if not m.any():
        m[rng.integers(space.n_cols)] = True
    for c in np.flatnonzero(m):
        if rng.random() < 0.8:
            x[c] += rng.choice([-1, 1])
        else:
            x[c] = rng.integers(space.x_min[c], space.x_max[c] + 1)
    return space.clip(x)


def top_k(X: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """점수 상위 k 개의 X (내림차순)."""
    return X[np.argsort(scores)[::-1][:k]]


def algorithm(name: str, state: dict | None = None, **cfg):
    """함수 하나를 알고리즘으로 등록한다 (`ALGORITHMS[name]`).

    Args:
        name  : 이름 (run.py --algo 값)
        state : 라운드 간 기억의 초깃값 (매 run 깊은 복사)
        **cfg : 하이퍼파라미터 기본값 → data["cfg"]
    """
    import copy

    if state is not None and not isinstance(state, dict):
        raise TypeError("state 는 dict 여야 한다 (pickle 가능한 값만)")

    def decorate(fn):
        if name in ALGORITHMS:
            raise ValueError(f"알고리즘 이름 중복: {name!r}")
        fn.algo_name = name
        fn.init_state = lambda: copy.deepcopy(state) if state else {}
        fn.cfg = dict(cfg)
        ALGORITHMS[name] = fn
        return fn

    return decorate


def propose(fn, X, Y, state, rng, space, budget, cfg=None) -> np.ndarray:
    """알고리즘을 한 번 호출해 (b, 30) 정수 배치를 얻는다. 형상·범위를 검증한다."""
    data = {"X": X, "Y": Y, "scores": get_scores(Y) if len(Y) else np.empty(0),
            "state": state, "rng": rng, "space": space, "budget": budget,
            "cfg": {**fn.cfg, **(cfg or {})}}
    out = fn(data)
    batch = np.atleast_2d(np.asarray(out, dtype=np.int64))
    if batch.ndim != 2 or batch.shape[1] != space.n_cols:
        raise ValueError(f"{fn.algo_name}: 반환 형상 {batch.shape} "
                         f"— (b, {space.n_cols}) 여야 함")
    if len(batch) == 0:
        raise ValueError(f"{fn.algo_name}: 빈 배치를 반환했다")
    return space.clip(batch)


# ──────────────────────────────────────────────────────────────────────────────
# 3. 알고리즘 구현
# ──────────────────────────────────────────────────────────────────────────────

@algorithm("random", batch=10)
def random_search(data):
    """균등 랜덤. baseline — 이걸 못 이기면 문제가 있는 것이다."""
    return sample(data["space"], data["rng"], data["cfg"]["batch"])


@algorithm("xgb_tr",
           state={"models": None, "seen_n": 0, "radius": 8, "succ": 0, "fail": 0,
                  "traj": 0, "reseed": False, "rounds": 0, "seen": None},
           n_startup=30, batch=4, n_cand=300, kappa=1.0, r_init=8, r_max=15,
           succ_tol=3, fail_tol=8, refit=4, n_ens=4, max_train=4000)
def xgb_trust_region(data):
    """XGB 앙상블 surrogate + 해밍 신뢰영역 (CASMOPOLITAN-lite).

    - 모델은 전체 히스토리로 학습(전역 정보 유지)하되, 후보는 현 trajectory 의
      best 로부터 해밍 반경 R 안에서만 생성한다(국소 탐색).
    - R 은 연속 개선이면 2배(최대 r_max), 연속 정체면 절반. R < 1 이면 restart —
      새 랜덤 지점에서 trajectory 를 다시 시작한다(모델·히스토리는 유지).
      "나쁜 초기 basin 고착" 이라는 전역 surrogate 의 실패 모드를 이게 끊는다.
    - acquisition = UCB(μ + κσ). σ 는 시드/부표본이 다른 앙상블의 예측 표준편차.
    """
    X, s, st, rng, sp, cfg = (data["X"], data["scores"], data["state"],
                              data["rng"], data["space"], data["cfg"])
    n = len(s)
    if st["seen"] is None:
        st["seen"] = set()

    # ── 새 관측 반영: 기관측 set + 신뢰영역 반경 + 주기적 재학습 ──
    if n > st["seen_n"]:
        for i in range(st["seen_n"], n):
            st["seen"].add(X[i].tobytes())
        st["rounds"] += 1
        traj = np.arange(st["traj"], n)
        if len(traj) and n >= cfg["n_startup"]:
            best_i = traj[int(np.argmax(s[traj]))]
            if best_i >= st["seen_n"]:                     # 이번 배치에서 개선
                st["succ"] += 1
                st["fail"] = 0
                if st["succ"] >= cfg["succ_tol"]:
                    st["radius"] = min(st["radius"] * 2, cfg["r_max"])
                    st["succ"] = 0
            else:                                          # 정체
                st["fail"] += 1
                st["succ"] = 0
                if st["fail"] >= cfg["fail_tol"]:
                    st["fail"] = 0
                    r = st["radius"] // 2
                    if r < 1:                              # 수렴 → restart
                        st["radius"], st["traj"] = cfg["r_init"], n
                        st["reseed"] = True
                    else:
                        st["radius"] = r
        interval = max(cfg["refit"], n // 2000)
        if n >= cfg["n_startup"] and (st["models"] is None
                                      or st["rounds"] % interval == 0):
            from xgboost import XGBRegressor
            Xt, st_ = X, s
            if len(s) > cfg["max_train"]:                  # elite 절반 + 랜덤 절반
                half = cfg["max_train"] // 2
                order = np.argsort(s)[::-1]
                keep = np.concatenate([order[:half],
                                       rng.choice(order[half:], half, replace=False)])
                Xt, st_ = X[keep], s[keep]
            st["models"] = [
                XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.1,
                             subsample=0.7, colsample_bytree=0.8, n_jobs=2,
                             verbosity=0, random_state=int(rng.integers(2**31)))
                .fit(Xt.astype(np.float32), st_.astype(np.float32))
                for _ in range(cfg["n_ens"])]
        st["seen_n"] = n

    if n < cfg["n_startup"] or st["models"] is None or st["reseed"]:
        st["reseed"] = False
        return sample(sp, rng, cfg["batch"])               # 시동 / restart 직후

    traj = np.arange(st["traj"], n)
    inc = X[traj[int(np.argmax(s[traj]))]]                 # 현 trajectory 의 best
    R = max(1, int(st["radius"]))
    cands = np.tile(inc, (cfg["n_cand"], 1))               # 신뢰영역 안 후보
    for i in range(cfg["n_cand"]):
        d = int(rng.integers(1, R + 1))
        for c in rng.choice(sp.n_cols, size=d, replace=False):
            lo, hi = int(sp.x_min[c]), int(sp.x_max[c])
            if rng.random() < 0.7:                         # ordinal 이웃 스텝
                cands[i, c] = np.clip(cands[i, c] + rng.choice([-1, 1]), lo, hi)
            else:                                          # 가끔 값 점프
                cands[i, c] = rng.integers(lo, hi + 1)
    P = np.stack([m.predict(cands.astype(np.float32)) for m in st["models"]])
    acq = P.mean(axis=0) + cfg["kappa"] * P.std(axis=0)

    seen, out = set(st["seen"]), []                        # 기평가·중복 제외
    for i in np.argsort(acq)[::-1]:
        key = cands[i].tobytes()
        if key not in seen:
            seen.add(key)
            out.append(cands[i])
        if len(out) == cfg["batch"]:
            break
    while len(out) < cfg["batch"]:
        out.append(sample(sp, rng, 1)[0])
    return out


if __name__ == "__main__":
    space = SearchSpace()
    rng = np.random.default_rng(0)
    print(f"등록된 알고리즘: {sorted(ALGORITHMS)}")

    # 점수 파이프라인
    Y = rng.normal(0, 1, (50, 6)) * [900, 1.1, 0.005, 900, 1.1, 0.005]
    for kind in ("chebyshev", "sum", "owa"):
        v = get_scores(Y, kind)
        assert v.shape == (50,) and (0 <= v).all() and (v <= 1).all()
    print(f"[OK] get_scores 3종 — 범위 [0,1], 최소화 목적 뒤집기 적용")

    # 마스크 측정
    g = 32
    m = np.zeros((2, g, g), dtype=bool)
    m[:, 10:20, 12:26] = True                      # 세로 10 × 가로 14
    Yc = convert_y_raw({"mask1": m, "mask2": m,
                        "y13": np.zeros(2), "y23": np.zeros(2)})
    assert (Yc[:, 0] == 10).all() and (Yc[:, 1] == 14).all()
    print("[OK] convert_y_raw — max height/width 측정")

    # 알고리즘: ask 사이클 + state pickle + 재개 궤적
    import pickle
    for name, fn in ALGORITHMS.items():
        def trace(cut=None):
            st, X = fn.init_state(), np.empty((0, space.n_cols), np.int64)
            Y_, r, xs = np.empty((0, 6)), np.random.default_rng(3), []
            for i in range(8):
                b = propose(fn, X, Y_, st, r, space, 100)
                xs.append(b.copy())
                X = np.vstack([X, b])
                Y_ = np.vstack([Y_, r.normal(0, 1, (len(b), 6)) * 100])
                if cut == i:
                    st = pickle.loads(pickle.dumps(st))     # 저장→로드 왕복
            return np.vstack(xs)
        assert np.array_equal(trace(), trace(cut=3)), f"{name}: 재개 궤적이 다르다"
        print(f"[OK] {name:<8} 제안·재개 궤적 동일, state pickle 가능")
