"""run.py — 루프. 측정하는 쪽이 주인이다.

    while 예산 남음:
        for x in algo(data):        # 다음에 잴 X 들
            y_raw = 측정(x)          # ← 루프의 주인은 여기
            obs.jsonl 에 append      # 관측 파일이 곧 상태

**관측 파일이 상태다.** 중단해도 파일만 있으면 이어서 돌고, 사람이 손으로 한 줄
추가해도 즉시 반영된다. 알고리즘 자체의 기억(신뢰영역 반경 등)은 히스토리로
복원되지 않으므로 `<obs>.state.pkl` 에 따로 남긴다 — 없으면 처음부터 시작한다.

측정이 이 프로세스 밖에 있으면(장비·다른 서비스) `step()` 을 직접 부르면 된다:

    xs, state = step(algo, data, state, rng, space, budget)
    for x in xs: 측정하고 obs.jsonl 에 append

실행:
    python run.py --algo xgb_tr --obs obs.jsonl --budget 800
    python run.py --algo xgb_tr --obs obs.jsonl --budget 800 --resume
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from algo import ALGORITHMS, OBJECTIVE_NAMES, convert_y_raw, get_scores, propose
from calculator import SurfaceCalculator, load_observations, save_observations
from space import SearchSpace


def _state_path(obs_path) -> Path:
    """obs.jsonl → obs.state.pkl (알고리즘 기억. 없으면 처음부터)."""
    p = Path(obs_path)
    return p.with_name(p.stem + ".state.pkl")


def run(algo_name: str, calc, obs_path, budget: int = 800, seed: int = 0,
        resume: bool = False, verbose: bool = True) -> dict:
    """알고리즘 하나를 budget 회 평가할 때까지 돌린다.

    Args:
        calc     : `evaluate(X) -> y_raw dict` 를 만족하는 측정기
        obs_path : 관측 파일. 여기에 append 하며 이게 곧 상태다
        resume   : True 면 기존 <obs>.state.pkl 에서 알고리즘 기억을 이어받는다
    """
    if algo_name not in ALGORITHMS:
        raise ValueError(f"알 수 없는 알고리즘 {algo_name!r} — {sorted(ALGORITHMS)}")
    fn = ALGORITHMS[algo_name]
    space = SearchSpace()
    obs_path, sp_path = Path(obs_path), _state_path(obs_path)

    d = load_observations(obs_path)
    X, Y = d["X"], d["Y"]
    start = len(X)

    state, rng = fn.init_state(), np.random.default_rng(seed)
    if resume and sp_path.exists():
        saved = pickle.loads(sp_path.read_bytes())
        state, rng = saved["state"], np.random.default_rng()
        rng.bit_generator.state = saved["rng"]
        start = saved["start"]
        if verbose:
            print(f"[재개] {sp_path.name} — 관측 {len(X)}행, 이 run 에서 "
                  f"{len(X) - start}/{budget} 완료")

    t0 = time.perf_counter()
    while len(X) - start < budget:
        for x in propose(fn, X, Y, state, rng, space, budget):
            raw = calc.evaluate(x[None, :])
            y = convert_y_raw(raw)
            save_observations(obs_path, x[None, :], y, block=[algo_name],
                              masks={"mask1": raw["mask1"], "mask2": raw["mask2"]},
                              append=True)
            X, Y = np.vstack([X, x[None, :]]), np.vstack([Y, y])
            if len(X) - start >= budget:
                break
        sp_path.write_bytes(pickle.dumps(
            {"state": state, "rng": rng.bit_generator.state, "start": start}))

    scores = get_scores(Y)
    i = int(np.argmax(scores))
    out = {"best_x": X[i].copy(), "best_score": float(scores[i]),
           "best_y": Y[i].copy(), "X": X, "Y": Y, "scores": scores,
           "n_evals": len(X) - start, "elapsed_sec": time.perf_counter() - t0}
    if verbose:
        print(f"{algo_name}: {out['n_evals']} evals, {out['elapsed_sec']:.1f}s, "
              f"관측 {start} → {len(X)}행, best={out['best_score']:.4f}")
        if hasattr(calc, "report"):
            rep = calc.report()
            print(f"  커버리지: no_data {rep['no_data_rate']:.0%} / "
                  f"exact {rep['exact_rate']:.0%} · 관측까지 해밍거리 중앙값 "
                  f"{rep['d_hamming']['median']:.3f} (게이트 {rep['d_gate']:.3f})")
            if rep["no_data_rate"] > 0.5:
                print("  ⚠ 제안의 절반 이상이 관측 커버리지 밖 — 이 run 의 y 는 "
                      "대부분 근거가 없다")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="반응표면 위에서 알고리즘 하나를 구동")
    ap.add_argument("--algo", default="xgb_tr", choices=sorted(ALGORITHMS))
    ap.add_argument("--obs", type=Path, default=Path("obs.jsonl"),
                    help="관측 파일 (여기에 append 한다)")
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="<obs>.state.pkl 에서 알고리즘 기억을 이어받는다")
    ap.add_argument("--policy", default="pessimistic",
                    choices=("flag", "pessimistic", "strict"),
                    help="커버리지 밖 처리 (기본: 최악 관측치로 채워 관측 영역으로 유도)")
    args = ap.parse_args()

    calc = SurfaceCalculator.from_jsonl(args.obs, policy=args.policy)
    r = run(args.algo, calc, args.obs, args.budget, args.seed, args.resume)
    print("  best_x =", r["best_x"].tolist())
    print("  best_y =", dict(zip(OBJECTIVE_NAMES, r["best_y"].round(4).tolist())))


if __name__ == "__main__":
    main()
