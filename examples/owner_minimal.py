"""owner_minimal.py — **가장 단순한 형태.** 알고리즘은 함수 하나, 상태는 파일.

    def dummy_algo(data) -> [다음 X, 다음 X, ...]

`data` 는 obs.jsonl 을 읽은 그대로다 (`calculator.load_observations` 결과 +
점수). 알고리즘은 그걸 보고 다음에 잴 X 들을 돌려주기만 한다. 클래스도,
state 도, ask/tell 도 없다 — **관측 파일이 곧 상태**이기 때문이다.

    측정 → obs.jsonl 에 append → 다시 읽어서 algo 에 넘김

그래서 중단해도 파일만 있으면 이어서 돌릴 수 있고, 사람이 손으로 한 줄
추가해도 즉시 반영된다.

쓸 수 있는 조건: **알고리즘이 히스토리만 보고 판단할 수 있을 것.**
(random / TPE / EDA / top-k 기반 GA 등이 여기 해당한다. SA 의 "현재 해",
 PSO 의 속도처럼 히스토리에 안 적히는 기억이 필요하면 `owner_inprocess.py`
 의 Session 을 쓸 것 — 거기서도 루프의 주인은 여전히 당신이다.)

    python examples/owner_minimal.py
"""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from calculator import SurfaceCalculator, load_observations, save_observations
from optimizer import convert_y_raw, get_scores
from space import SearchSpace

BUDGET = 800
OBS = _ROOT / "obs.jsonl"
space = SearchSpace()
rng = np.random.default_rng(0)


# ── 알고리즘 — 이 함수만 바꾸면 된다 ─────────────────────────────────────
def dummy_algo(data):
    """관측을 보고 다음에 잴 X 들을 돌려준다.

    data["X"]      (n, 30) int   지금까지 잰 X
    data["Y"]      (n, 6)  float 대응 관측 (y11,y12,y13,y21,y22,y23)
    data["scores"] (n,)    float [0,1] 점수, 클수록 좋음  ← 보통 이걸 본다
    data["mask1"] / data["mask2"]  (n,128,128) bool  마스크 원형 (필요하면)
    """
    X, s = data["X"], data["scores"]
    elite = X[np.argsort(s)[::-1][:8]]                    # 상위 8개
    out = []
    for _ in range(4):                                    # 자식 4개
        a, b = elite[rng.integers(8)], elite[rng.integers(8)]
        child = np.where(rng.random(space.n_cols) < 0.5, a, b)   # 교차
        c = rng.integers(space.n_cols)                           # 변이
        child[c] = rng.integers(space.x_min[c], space.x_max[c] + 1)
        out.append(space.clip(child))
    return out


# ── 루프 — 주인은 측정하는 쪽(당신) ──────────────────────────────────────
def measure(x):
    """실제로는 장비/외부 서비스. 여기서는 반응표면으로 대역한다."""
    return calc.evaluate(np.atleast_2d(x))


if __name__ == "__main__":
    calc = SurfaceCalculator.from_jsonl(OBS, policy="pessimistic")
    work = _ROOT / "run_obs.jsonl"                 # 원본을 건드리지 않는 사본
    work.write_text(OBS.read_text())

    data = load_observations(work)
    data["scores"] = get_scores(data["Y"])
    start = len(data["X"])

    while len(data["X"]) - start < BUDGET:
        for x in dummy_algo(data):                 # ← 알고리즘 호출
            y_raw = measure(x)                     # ← 측정
            save_observations(work, np.atleast_2d(x), convert_y_raw(y_raw),
                              block=["run"], masks={"mask1": y_raw["mask1"],
                                                    "mask2": y_raw["mask2"]},
                              append=True)         # ← 파일에 append (상태 저장)
            # 메모리 쪽도 갱신 (매번 파일 전체를 다시 읽지 않으려고)
            data["X"] = np.vstack([data["X"], np.atleast_2d(x)])
            data["Y"] = np.vstack([data["Y"], convert_y_raw(y_raw)])
            data["scores"] = get_scores(data["Y"])
            if len(data["X"]) - start >= BUDGET:
                break

    i = int(np.argmax(data["scores"]))
    print(f"{BUDGET} evals (관측 {start} → {len(data['X'])}), "
          f"best_score={data['scores'][i]:.4f}")
    print(f"best_x = {data['X'][i].tolist()}")
    print(f"best_y = {data['Y'][i].round(4).tolist()}")
    print(f"이어서 돌리려면: {work.name} 를 그대로 다시 넘기면 된다")
