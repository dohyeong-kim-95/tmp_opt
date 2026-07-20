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
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from calculator import BENCHMARKS
from optimizer import (OPTIMIZERS, OptimizerBase, SCORERS, append_history,
                       load_history, save_state)
from space import SearchSpace

_HERE = Path(__file__).resolve().parent


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
    checkpoint_dir: Path | None = None,
) -> RunResult:
    """optimizer 하나를 벤치마크 하나에서 budget 회 평가할 때까지 실행한다.

    Args:
        scorer_name   : optimizer 내부 score 파이프라인의 scalarization 선택.
        checkpoint_dir: 지정하면 매 tell 후 history.jsonl(관측 append-only) +
                        state.pkl(알고리즘 상태)을 기록한다. 이 두 파일로
                        optimizer.load_state 재개가 가능하다.
    """
    space = SearchSpace()
    calc = BENCHMARKS[benchmark_name](noise_seed=seed)
    opt: OptimizerBase = OPTIMIZERS[optimizer_name](
        space, total_budget=budget, scorer_name=scorer_name
    )
    # 배선(dispatch) 판별: 요청한 이름과 생성된 optimizer 가 일치해야 한다.
    assert opt.name == optimizer_name, \
        f"dispatch 불일치: 요청 {optimizer_name!r} → 생성 {opt.name!r}"

    if checkpoint_dir is not None:  # 새 run 은 이전 히스토리를 이어 쓰지 않는다
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "history.jsonl").unlink(missing_ok=True)

    state = opt.init_state(seed)
    t0 = time.perf_counter()
    n = 0
    while n < budget:
        batch, state = opt.ask(state)
        batch = batch[: budget - n]  # 예산 초과분은 잘라낸다

        # 평가 (calc 는 배치를 받아도 내부적으로 순차 의미 유지 — 병렬 불가 가정).
        # 반환은 구조화 raw 관측(마스크+스칼라) — 수치화는 tell 내부의
        # convert_y_raw(optimizer 소유 이음새)가 담당한다. runner 는 형태를 모른다.
        raw = calc.evaluate(batch)
        state = opt.tell(state, batch, raw)

        if checkpoint_dir is not None:  # 선택적 체크포인트 (관측 + 상태 분리)
            # 히스토리에는 변환된 (b, K) 측정치를 기록한다 (마스크 원형은
            # 용량·가독성 문제로 보존하지 않음 — 변환은 결정적이다)
            append_history(checkpoint_dir / "history.jsonl", batch,
                           state["Y_raw_hist"][n:n + len(batch)], eval_index=n)
            save_state(checkpoint_dir / "state.pkl", state)
        n += len(batch)

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


def run_separated(
    optimizer_name: str,
    benchmark_name: str,
    seed: int,
    budget: int,
    exchange_dir: Path,
    verbose: bool = True,
) -> RunResult:
    """**프로세스 분리** 실행 — optimizer 와 calculator 를 별도 서브프로세스로
    번갈아 띄우고 교환 디렉토리의 파일(x.txt / y_raw.bin)로만 통신한다.

    두 프로세스는 공유 메모리가 없으므로 in-process 우회가 물리적으로 불가능하다
    — 파일 교환이 실제로 일어나야만 한 스텝이 진행된다. runner 는 순서만 강제:
        opt-step (x.txt 쓰거나 done) → calc-eval (y_raw.bin 씀) → 반복

    한 handshake 라운드 = optimizer 배치 하나. dispatch/배선은 서브프로세스
    exit code + done 마커로 강제된다.
    """
    d = Path(exchange_dir)
    d.mkdir(parents=True, exist_ok=True)
    for f in ("state.pkl", "history.jsonl", "x.txt", "y_raw.bin", "done"):
        (d / f).unlink(missing_ok=True)

    opt_cmd = [sys.executable, str(_HERE / "optimizer.py"), "--serve-step",
               "--optimizer", optimizer_name, "--dir", str(d),
               "--seed", str(seed), "--budget", str(budget)]
    calc_cmd = [sys.executable, str(_HERE / "calculator.py"), "--serve-eval",
                "--benchmark", benchmark_name, "--dir", str(d), "--seed", str(seed)]

    t0 = time.perf_counter()
    rounds = 0
    while True:
        subprocess.run(opt_cmd, check=True, cwd=_HERE,
                       capture_output=not verbose)
        if (d / "done").exists():
            break
        subprocess.run(calc_cmd, check=True, cwd=_HERE,
                       capture_output=not verbose)
        rounds += 1
        if rounds > budget + 5:  # 안전장치 (batch=1 이라도 budget 라운드면 끝)
            raise RuntimeError("handshake 라운드가 예산을 초과 — 진행 안 됨")

    # 산출물(history.jsonl)에서 결과 복원 — runner 는 optimizer 내부 상태를 안 본다
    X, Y = load_history(d / "history.jsonl", space=SearchSpace())
    assert len(X) == budget, f"history 길이 {len(X)} ≠ 예산 {budget}"
    if verbose:
        print(f"[separated] {optimizer_name} on {benchmark_name}: "
              f"{rounds} handshake 라운드, {len(X)} evals, "
              f"{time.perf_counter() - t0:.1f}s")
    return RunResult(optimizer_name, benchmark_name, seed, X, Y,
                     time.perf_counter() - t0)


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
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="매 tell 후 history.jsonl + state.pkl 을 기록할 디렉토리")
    parser.add_argument("--separate", type=Path, default=None, metavar="DIR",
                        help="프로세스 분리 실행 — optimizer/calculator 를 별도 "
                             "서브프로세스로 띄우고 지정 디렉토리의 파일로만 통신")
    args = parser.parse_args()

    if args.separate is not None:  # 파일 기반 프로세스 분리 실행
        r = run_separated(args.optimizer, args.benchmark, args.seed,
                          args.budget, args.separate)
        print(f"{r.optimizer} on {r.benchmark} seed={r.seed}: "
              f"{len(r.X)} evals via 파일 교환 ({args.separate})")
        return

    r = run_single(args.optimizer, args.benchmark, args.seed, args.budget,
                   args.scorer, args.checkpoint_dir)
    drive_best = float(r.final_state["scores_hist"].max())
    print(f"{r.optimizer} on {r.benchmark} seed={r.seed}: "
          f"{len(r.X)} evals, {r.elapsed_sec:.1f}s, "
          f"drive best={drive_best:.4f} "
          f"(per-run 스케일러 기준 — run 간 비교는 benchmark.py 의 pooled 점수로)")


if __name__ == "__main__":
    main()
