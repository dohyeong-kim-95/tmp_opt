"""runner.py — calculator 와 optimizer 를 반복 호출하는 실행 루프.

이 파일은 의도적으로 '기계'다. 한 run 은 아래 세 줄의 반복일 뿐이다:
    1. batch, state = opt.ask(state)                 # 후보 제안
    2. batch 를 **순차** 평가 (병렬 불가 가정)        # calculator 호출
    3. state = opt.tell(state, batch, Y_raw_batch)   # 증분 raw 관측 통보

역할 분담:
    - calculator.py : X → raw y0 계산 (문제 정의·노이즈)
    - optimizer.py  : 나머지 전부 — 히스토리 누적, 스케일링/sense 통일,
                      scalarization, 알고리즘. runner 는 점수를 전혀 모른다.
    - benchmark.py  : 여러 run 의 비교 — pooled 재점수, 토너먼트/격자, 시각화.

실행 예:
    python runner.py --optimizer sa --benchmark bm1_easy --seed 0 --budget 800
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from calculator import BENCHMARKS
from optimizer import OPTIMIZERS, OptimizerBase, SCORERS
from space import SearchSpace


@dataclass
class RunResult:
    """한 번의 (optimizer × benchmark × seed) 실행 결과."""

    optimizer: str
    benchmark: str
    seed: int
    X: np.ndarray        # (N, 30) 평가된 해들 (평가 순서대로)
    Y_raw: np.ndarray    # (N, 6)  raw 관측값 (노이즈 포함)
    elapsed_sec: float
    final_state: dict = field(repr=False, default_factory=dict)


def run_single(
    optimizer_name: str,
    benchmark_name: str,
    seed: int,
    budget: int,
    scorer_name: str = "chebyshev",
    state_file: Path | None = None,
) -> RunResult:
    """optimizer 하나를 벤치마크 하나에서 budget 회 평가할 때까지 실행한다.

    Args:
        scorer_name: optimizer 내부 score 파이프라인의 scalarization 선택.
        state_file : 지정하면 매 tell 후 상태를 pickle 로 저장한다 (체크포인트.
                     히스토리도 상태 안에 있으므로 이 파일 하나로 재개 가능).
    """
    space = SearchSpace()
    calc = BENCHMARKS[benchmark_name](noise_seed=seed)
    opt: OptimizerBase = OPTIMIZERS[optimizer_name](
        space, total_budget=budget, scorer_name=scorer_name
    )
    # 배선(dispatch) 판별: 요청한 이름과 생성된 optimizer 가 일치해야 한다.
    assert opt.name == optimizer_name, \
        f"dispatch 불일치: 요청 {optimizer_name!r} → 생성 {opt.name!r}"

    state = opt.init_state(seed)
    t0 = time.perf_counter()
    n = 0
    while n < budget:
        batch, state = opt.ask(state)
        batch = batch[: budget - n]  # 예산 초과분은 잘라낸다

        # 순차 평가 (실전 가정: 병렬 불가) — 한 행씩 평가해 모은다
        Y_raw = np.array([calc.evaluate(x) for x in batch])
        state = opt.tell(state, batch, Y_raw)
        n += len(batch)

        if state_file is not None:  # 선택적 체크포인트
            with open(state_file, "wb") as f:
                pickle.dump(state, f)

    assert state["n_evals"] == budget  # 히스토리 무결성 (optimizer 소유)
    return RunResult(
        optimizer=optimizer_name,
        benchmark=benchmark_name,
        seed=seed,
        X=state["X_hist"].copy(),
        Y_raw=state["Y_raw_hist"].copy(),
        elapsed_sec=time.perf_counter() - t0,
        final_state=state,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="단일 run 실행기 (여러 run 비교/시각화는 benchmark.py)"
    )
    parser.add_argument("--optimizer", choices=list(OPTIMIZERS), default="random")
    parser.add_argument("--benchmark", choices=list(BENCHMARKS), default="bm1_easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=800,
                        help="run 당 평가 횟수 (기본 800)")
    parser.add_argument("--scorer", choices=list(SCORERS), default="chebyshev",
                        help="optimizer 내부 scalarization (기본 chebyshev)")
    parser.add_argument("--state-file", type=Path, default=None,
                        help="매 tell 후 상태를 저장할 pickle 경로 (체크포인트)")
    args = parser.parse_args()

    r = run_single(args.optimizer, args.benchmark, args.seed, args.budget,
                   args.scorer, args.state_file)
    drive_best = float(r.final_state["scores_hist"].max())
    print(f"{r.optimizer} on {r.benchmark} seed={r.seed}: "
          f"{len(r.X)} evals, {r.elapsed_sec:.1f}s, "
          f"drive best={drive_best:.4f} "
          f"(per-run 스케일러 기준 — run 간 비교는 benchmark.py 의 pooled 점수로)")


if __name__ == "__main__":
    main()
