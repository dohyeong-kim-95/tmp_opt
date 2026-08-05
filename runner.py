"""runner.py — calculator 와 optimizer 를 반복 호출하는 실행 루프.

이 파일은 의도적으로 '기계'다. 한 run 은 아래 세 줄의 반복일 뿐이다:
    1. batch, state = opt.ask(state)                 # 후보 제안
    2. batch 를 **순차** 평가 (병렬 불가 가정)        # calculator 호출
    3. state = opt.tell(state, batch, Y_raw_batch)   # 증분 raw 관측 통보

역할 분담:
    - calculator.py : X → raw y0 계산 (문제 정의 = 실측 반응표면)
    - optimizer.py  : 나머지 전부 — 히스토리 누적, 스케일링/sense 통일,
                      scalarization, 알고리즘. runner 는 점수를 전혀 모른다.

실행 예:
    python runner.py --optimizer sa --surface-data obs.jsonl --seed 0 --budget 800

calculator 는 이름으로 조회하지 않고 **인스턴스를 직접 받는다** — 합성 벤치마크
레지스트리(BENCHMARKS)를 걷어내면서 그 간접층이 필요 없어졌다. 계약은
`evaluate(X) -> y_raw dict` 하나뿐이라 다른 측정기를 물려도 그대로 동작한다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from calculator import SurfaceCalculator
from optimizer import (OPTIMIZERS, OptimizerBase, SCORERS, append_history,
                       load_history, load_plugins, save_state)
from space import SearchSpace

_HERE = Path(__file__).resolve().parent


@dataclass
class RunResult:
    """한 번의 (optimizer × calculator × seed) 실행 결과."""

    optimizer: str
    source: str          # 관측 출처 라벨 (보통 obs.jsonl 경로)
    seed: int
    X: np.ndarray        # (N, 30) 평가된 해들 (평가 순서대로)
    Y_raw: np.ndarray    # (N, 6)  raw 관측값 (노이즈 포함)
    elapsed_sec: float
    final_state: dict = field(repr=False, default_factory=dict)


def run_single(
    optimizer_name: str,
    calc,
    seed: int,
    budget: int,
    scorer_name: str = "chebyshev",
    checkpoint_dir: Path | None = None,
    source: str = "surface",
) -> RunResult:
    """optimizer 하나를 calculator 하나에서 budget 회 평가할 때까지 실행한다.

    Args:
        calc          : `evaluate(X) -> y_raw dict` 를 만족하는 계산기 인스턴스.
        source        : 결과에 남길 관측 출처 라벨 (보통 obs.jsonl 경로).
        scorer_name   : optimizer 내부 score 파이프라인의 scalarization 선택.
        checkpoint_dir: 지정하면 매 tell 후 history.jsonl(관측 append-only) +
                        state.pkl(알고리즘 상태)을 기록한다. 이 두 파일로
                        optimizer.load_state 재개가 가능하다.
    """
    space = SearchSpace()
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
        source=source,
        seed=seed,
        X=state["X_hist"].copy(),
        Y_raw=state["Y_raw_hist"].copy(),
        elapsed_sec=time.perf_counter() - t0,
        final_state=state,
    )


def optimize(optimizer_name: str, calc, budget: int = 800, seed: int = 0,
             scorer_name: str = "chebyshev", **kw) -> dict:
    """**한 번 호출로 끝나는 진입점** — `scipy.optimize.minimize(f, ...)` 스타일.

    ask-tell 이 익숙하지 않다면 이것부터 쓰면 된다. 루프·상태·점수화를 전부
    숨기고 결과만 돌려준다:

        from calculator import SurfaceCalculator
        from runner import optimize

        calc = SurfaceCalculator.from_jsonl("obs.jsonl")
        res = optimize("xgb_tr", calc, budget=800)
        print(res["best_x"], res["best_score"])

    Returns:
        {"best_x": (30,) int64, "best_score": float, "best_y": (6,) float,
         "X": (N,30), "Y": (N,6), "scores": (N,), "elapsed_sec": float,
         "result": RunResult}
    """
    r = run_single(optimizer_name, calc, seed, budget, scorer_name, **kw)
    s = r.final_state["scores_hist"]
    i = int(np.argmax(s))
    return {"best_x": r.X[i].copy(), "best_score": float(s[i]),
            "best_y": r.Y_raw[i].copy(), "X": r.X, "Y": r.Y_raw,
            "scores": np.asarray(s).copy(),
            "elapsed_sec": r.elapsed_sec, "result": r}


class Session:
    """**루프를 당신이 소유하는 진입점** — 한 점씩 묻고 답한다.

    측정이 이 프로세스 밖에 있을 때(실측 장비, 사람의 수작업, 다른 서비스)
    쓴다. optimizer 는 측정 함수를 전혀 모른다:

        sess = Session("xgb_tr", budget=800, seed=0)
        while not sess.done:
            x = sess.next_x()              # 다음에 측정할 X 하나
            y_raw = 어딘가에서_측정(x)      # 루프의 주인은 당신
            sess.report(x, y_raw)          # 결과 통보
        print(sess.best())

    배치는 내부 큐가 흡수하므로 호출자는 배치 개념을 몰라도 된다. 다만 큐에
    남은 후보는 갱신 전 정보로 만들어진 것이라, 한 점씩 report 해도 알고리즘이
    그 즉시 반응하지는 않는다 (population 계열은 원래 그렇게 동작한다).
    """

    def __init__(self, optimizer_name: str, budget: int = 800, seed: int = 0,
                 space: SearchSpace | None = None, **kw) -> None:
        self.opt: OptimizerBase = OPTIMIZERS[optimizer_name](
            space or SearchSpace(), total_budget=budget, **kw)
        self.state = self.opt.init_state(seed)
        self.budget = budget
        self._queue: list = []

    @property
    def n_evals(self) -> int:
        return int(self.state["n_evals"])

    @property
    def done(self) -> bool:
        return self.n_evals >= self.budget

    def next_x(self) -> np.ndarray:
        """다음에 평가할 X 하나 (30,)."""
        if self.done:
            raise RuntimeError(f"예산 {self.budget} 소진 — done 을 먼저 확인할 것")
        if not self._queue:
            batch, self.state = self.opt.ask(self.state)
            self._queue = [row for row in batch[: self.budget - self.n_evals]]
        return self._queue.pop(0)

    def report(self, x: np.ndarray, y_raw) -> None:
        """방금 측정한 (x, y_raw) 를 통보한다. y_raw 는 calculator 계약과 동일."""
        self.state = self.opt.tell(self.state, np.atleast_2d(x), y_raw)

    def best(self) -> dict:
        if self.n_evals == 0:
            raise RuntimeError("아직 관측이 없다")
        s = self.state["scores_hist"]
        i = int(np.argmax(s))
        return {"best_x": self.state["X_hist"][i].copy(),
                "best_score": float(s[i]),
                "best_y": self.state["Y_raw_hist"][i].copy()}


def run_separated(
    optimizer_name: str,
    surface_data: str | Path,
    seed: int,
    budget: int,
    exchange_dir: Path,
    verbose: bool = True,
    plugins: list[str] | None = None,
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
    for f in ("state.pkl", "history.jsonl", "x.txt", "y_raw.bin", "done",
              "coverage.jsonl"):
        (d / f).unlink(missing_ok=True)

    opt_cmd = [sys.executable, str(_HERE / "optimizer.py"), "--serve-step",
               "--optimizer", optimizer_name, "--dir", str(d),
               "--seed", str(seed), "--budget", str(budget)]
    for m in plugins or ():  # 외부 파일 알고리즘은 서브프로세스에서 다시 import
        opt_cmd += ["--plugin", m]
    calc_cmd = [sys.executable, str(_HERE / "calculator.py"), "--serve-eval",
                "--surface-data", str(surface_data), "--dir", str(d),
                "--seed", str(seed)]

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
        print(f"[separated] {optimizer_name} on {surface_data}: "
              f"{rounds} handshake 라운드, {len(X)} evals, "
              f"{time.perf_counter() - t0:.1f}s")
    return RunResult(optimizer_name, str(surface_data), seed, X, Y,
                     time.perf_counter() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="단일 run 실행기 — 반응표면 위에서 optimizer 하나를 구동"
    )
    # choices 미지정 — 플러그인은 파서 생성 뒤에 등록된다 (검증은 아래에서)
    parser.add_argument("--optimizer", default="random")
    parser.add_argument("--plugin", action="append", default=[], metavar="MODULE",
                        help="알고리즘이 정의된 모듈 (예: algo_template). 반복 지정 가능")
    parser.add_argument("--surface-data", type=Path, required=True, metavar="JSONL",
                        help="실측 관측 obs.jsonl (make_dataset.py 산출)")
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
    load_plugins(args.plugin)
    if args.optimizer not in OPTIMIZERS:
        raise SystemExit(f"알 수 없는 optimizer {args.optimizer!r}. "
                         f"사용 가능: {sorted(OPTIMIZERS)}")

    if args.separate is not None:  # 파일 기반 프로세스 분리 실행
        r = run_separated(args.optimizer, args.surface_data, args.seed,
                          args.budget, args.separate, plugins=args.plugin)
        print(f"{r.optimizer} on {r.source} seed={r.seed}: "
              f"{len(r.X)} evals via 파일 교환 ({args.separate})")
        return

    calc = SurfaceCalculator.from_jsonl(args.surface_data, noise_seed=args.seed)
    r = run_single(args.optimizer, calc, args.seed, args.budget,
                   args.scorer, args.checkpoint_dir, source=str(args.surface_data))
    drive_best = float(r.final_state["scores_hist"].max())
    print(f"{r.optimizer} on {r.source} seed={r.seed}: "
          f"{len(r.X)} evals, {r.elapsed_sec:.1f}s, "
          f"drive best={drive_best:.4f} (per-run 스케일러 기준)")
    rep = calc.report()
    print(f"  커버리지: no_data {rep['no_data_rate']:.1%} / "
          f"exact {rep['exact_rate']:.1%} · "
          f"관측까지 해밍거리 중앙값 {rep['d_hamming']['median']:.3f} "
          f"(최소 {rep['d_hamming']['min']:.3f}, 게이트 {rep['d_gate']:.3f})")
    if rep["no_data_rate"] > 0.5:
        print("  ⚠ 제안의 절반 이상이 관측 커버리지 밖 — 이 run 의 y 는 "
              "대부분 근거가 없다 (관측을 늘리거나 탐색 범위를 좁힐 것)")


if __name__ == "__main__":
    main()
