"""algo_template.py — 새 알고리즘 추가 템플릿. **이 파일을 복사해서 시작하라.**

계약은 하나다: **"지금까지의 관측을 보고, 다음에 평가할 X 들을 돌려준다."**
클래스도, 상속도, super() 도 없다. **인자 하나, 반환 하나.**

    @simple_algorithm("이름", state={...초기 기억...}, 하이퍼파라미터=기본값)
    def 내_알고리즘(data):
        ...
        return 다음_X_들

필요한 건 전부 `data` 안에 있다:

    data["X"]      (n, 30) int64   지금까지 평가된 X (평가 순서)
    data["Y"]      (n, 6)  float   대응 raw 관측 (열 = y11,y12,y13,y21,y22,y23)
    data["scores"] (n,)    float   [0,1] 점수, 클수록 좋음 ← **이걸 쓰면 된다**
                                   스케일 정규화·최소화 목적 뒤집기·scalarization
                                   이 이미 끝나 있다. 직접 다시 만들지 말 것 —
                                   알고리즘마다 다르게 만들면 비교가 깨진다.
    data["state"]  dict            라운드 간 기억 (히스토리로 복원 불가능한 것만).
                                   **제자리 수정**하면 그대로 반영된다.
    data["rng"]    Generator       난수. **반드시 이것만 쓸 것** (재현성)
    data["space"]  SearchSpace     x_min / x_max / n_cols / sample / clip
    data["cfg"]    dict            @simple_algorithm 에 넘긴 하이퍼파라미터
    data["budget"] int             총 예산 (SA 온도 스케줄 등에 쓴다)
    data["ctx"]    Ctx             헬퍼: ctx.sample(rng, n) / ctx.mutate(rng, x)
                                   / ctx.top_k(X, scores, k) / ctx.clip(x)

이건 `examples/owner_minimal.py` 와 같은 모양이다 — 거기서는 루프까지 직접
쓰고, 여기서는 루프를 프레임워크가 돌려준다는 것만 다르다.

규칙 세 개만 지키면 된다:
  1. **첫 호출은 히스토리가 비어 있다** (`len(X) == 0`) — 초기점을 내놓을 것
  2. **state 에는 pickle 가능한 값만** — 프로세스 분리 실행은 매 스텝이 새
     프로세스라 state.pkl 이 유일한 기억이다
  3. **난수는 rng 만** — 전역 np.random 을 쓰면 재개 시 궤적이 달라진다

실행:
    python runner.py --plugin algo_template --optimizer my_hill --surface-data obs.jsonl
    python algo_template.py          # 이 파일의 알고리즘들 자가 점검

(인자를 dict 대신 이름으로 받고 싶으면 `@algorithm` 을 쓴다 — 계약은 동일하고
 `X, scores, state, rng, ctx` 중 **선언한 것만** 넘어온다. 자세히는
 doc/algo/adding_an_algorithm.md)
"""

from __future__ import annotations

import numpy as np

from optimizer import simple_algorithm


# ──────────────────────────────────────────────────────────────────────────────
# 예시 1 — 상태가 필요 없는 알고리즘 (히스토리만 보면 되는 경우)
#
# `data["state"]` 를 아예 안 쓴다. 매 라운드 히스토리 상위권에서 부모를 뽑아
# 자식을 만들면 "population" 이 곧 상위 k 개다. 이런 알고리즘은
# examples/owner_minimal.py 처럼 파일만 가지고도 그대로 돌릴 수 있다.
# ──────────────────────────────────────────────────────────────────────────────

@simple_algorithm("my_ga", n_elite=8, batch=4, p_cross=0.5)
def my_ga(data):
    """상위 k 개를 부모 풀로 삼는 GA. state 불필요 — 히스토리가 곧 population."""
    X, scores, rng, ctx, cfg = (data["X"], data["scores"], data["rng"],
                                data["ctx"], data["cfg"])
    if len(X) < cfg["n_elite"]:
        return ctx.sample(rng, cfg["n_elite"])           # 시동: 랜덤으로 채운다

    elite = ctx.top_k(X, scores, cfg["n_elite"])
    children = []
    for _ in range(cfg["batch"]):
        a = elite[rng.integers(len(elite))]
        b = elite[rng.integers(len(elite))]
        child = np.where(rng.random(ctx.space.n_cols) < cfg["p_cross"], a, b)
        children.append(ctx.mutate(rng, child))          # 변이 (범위 클램프 포함)
    return children


# ──────────────────────────────────────────────────────────────────────────────
# 예시 2 — 상태가 필요한 알고리즘
#
# "현재 서 있는 지점" 은 히스토리만 보고 복원할 수 없다 (어떤 제안을 수락했는지는
# 주사위 결과에 달렸다). 그런 것만 data["state"] 에 넣는다 — 제자리 수정하면 된다.
# ──────────────────────────────────────────────────────────────────────────────

@simple_algorithm("my_hill", state={"cur": None}, step=2, restart_after=25)
def my_hill(data):
    """1-hop 언덕오르기 + 정체 시 재시작. state 는 '현재 지점' 인덱스뿐."""
    X, scores, st, rng, ctx, cfg = (data["X"], data["scores"], data["state"],
                                    data["rng"], data["ctx"], data["cfg"])
    if len(X) == 0:
        return ctx.sample(rng, 1)                        # 첫 호출: 히스토리가 비었다

    i = len(scores) - 1                                  # 방금 평가된 점
    if st["cur"] is None or scores[i] > scores[st["cur"]]:
        st["cur"] = i                                    # 개선 → 이동
        st["stall"] = 0
    else:
        st["stall"] = st.get("stall", 0) + 1

    if st["stall"] >= cfg["restart_after"]:              # 정체 → 재시작
        st["cur"], st["stall"] = None, 0
        return ctx.sample(rng, 1)

    return [ctx.mutate(rng, X[st["cur"]], rate=cfg["step"] / ctx.space.n_cols)]


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle

    from calculator import SurfaceCalculator
    from optimizer import OPTIMIZERS
    from runner import run_single
    from space import SearchSpace

    names = ["my_ga", "my_hill"]
    space = SearchSpace()
    print(f"등록됨: {[n for n in names if n in OPTIMIZERS]}")

    # 1) 관측 없이도 인터페이스가 성립하는가 (임의 raw 관측으로 ask-tell 사이클)
    rng = np.random.default_rng(0)
    for nm in names:
        opt = OPTIMIZERS[nm](space, total_budget=100)
        st = opt.init_state(7)
        for _ in range(6):
            batch, st = opt.ask(st)
            assert batch.ndim == 2 and batch.shape[1] == space.n_cols
            assert (batch >= space.x_min).all() and (batch <= space.x_max).all()
            st = opt.tell(st, batch, rng.normal(0, 1, (len(batch), 6)) * 100.0)
        pickle.loads(pickle.dumps(st))                   # 체크포인트 가능해야 한다
        print(f"  [OK] {nm:<9} ask-tell {st['n_evals']} evals, state pickle 가능")

    # 2) 중단·재개가 무중단과 같은 궤적인가 (규칙 3 위반 시 여기서 깨진다)
    for nm in names:
        def trace(cut=None):
            o = OPTIMIZERS[nm](space, total_budget=60)
            s, xs, r = o.init_state(1), [], np.random.default_rng(5)
            for k in range(12):
                b, s = o.ask(s)
                xs.append(b.copy())
                s = o.tell(s, b, r.normal(0, 1, (len(b), 6)) * 100.0)
                if cut == k:
                    s.update(pickle.loads(pickle.dumps(
                        {kk: vv for kk, vv in s.items()
                         if kk not in ("X_hist", "Y_raw_hist", "scores_hist")})))
            return np.vstack(xs)
        assert np.array_equal(trace(), trace(cut=5)), f"{nm}: 재개 궤적이 다르다"
        print(f"  [OK] {nm:<9} 중단·재개 궤적 동일")

    # 3) 실제 반응표면 위에서 완주하는가
    from pathlib import Path
    obs = Path(__file__).resolve().parent / "obs.jsonl"
    if obs.exists():
        for nm in names:
            calc = SurfaceCalculator.from_jsonl(obs, policy="pessimistic")
            r = run_single(nm, calc, seed=0, budget=80, source=str(obs))
            print(f"  [OK] {nm:<9} 반응표면 {len(r.X)} evals, "
                  f"best={float(r.final_state['scores_hist'].max()):.4f}")
    else:
        print(f"  (건너뜀) {obs} 없음 — python make_dataset.py --out obs.jsonl")
