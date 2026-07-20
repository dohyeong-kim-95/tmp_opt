"""benchmark.py — 여러 run 을 모아 비교하는 계층 (토너먼트/격자/시각화).

runner.run_single 이 만든 RunResult 들을 모아:
1. **pooled 재점수**: 탐색 중에는 각 run 이 자기 히스토리만으로 스케일러를
   적합하지만(실전과 동일 조건, optimizer 내부), run 마다 정규화 기준이 다르면
   점수를 직접 비교할 수 없다. 같은 벤치마크의 **모든 run 의 raw y0 을 합쳐
   (pooled)** 스케일러를 다시 적합한 뒤 점수·수렴곡선을 재계산한다.
   순위와 그림은 전부 이 pooled 점수 기준이다. (구현은 optimizer.py 의
   RobustScaler/SCORERS 를 그대로 공유 — 채점 파이프라인 단일화.)
2. **토너먼트**: bm1(3 seeds) → 상위 50% → bm2(5 seeds) → 상위 50%
   → bm3(10 seeds) 순으로 진행해 챔피언 optimizer 를 가린다.
3. **matrix 모드**: 탈락 없이 (optimizer × benchmark × seed) 전 격자 비교.
4. **top-3 추천 + confirmation**: 챔피언을 서로 다른 init 으로 3회 돌려
   후보 3개를 얻고, 각 후보를 반복 재측정(노이즈 포함)해 신뢰도를 확인한다.
5. **true optimum**: 무노이즈 장기(기본 100K evals) 실행 참조값.
6. **결과물**: results/ 에 parquet/json, vis/ 에 png 시각화.

실행 예:
    python benchmark.py                     # 토너먼트 + top-3 + 시각화
    python benchmark.py --smoke             # 초소형 예산으로 빠른 동작 확인
    python benchmark.py --matrix bm1_easy,bm3_hard --seeds 5
    python benchmark.py --compute-true-optimum
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경에서 파일 저장 전용
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from calculator import BENCHMARKS, OBJECTIVE_NAMES, TOURNAMENT_ORDER
from optimizer import OPTIMIZERS, SCORERS, RobustScaler, convert_y_raw
from runner import RunResult, run_single
from space import SearchSpace

# ──────────────────────────────────────────────────────────────────────────────
# 경로 / 시각화 상수
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"
VIS_DIR = Path(__file__).parent / "vis"

# optimizer 별 고정 색상 (검증된 categorical 팔레트, 순서 고정 — 재배색 금지).
# 어떤 그림에서든 같은 optimizer 는 항상 같은 색을 쓴다.
OPTIMIZER_COLORS: dict[str, str] = {
    "random": "#2a78d6",           # blue
    "blockwise_coord": "#1baf7a",  # aqua
    "ga": "#eda100",               # yellow
    "sa": "#008300",               # green
    "pso": "#4a3aa7",              # violet
    "aco": "#e34948",              # red
    "tpe": "#e87ba4",              # magenta
    "xgb_surrogate": "#eb6834",    # orange
    # 팔레트 8슬롯이 모두 찼으므로 9번째 이후 series 는 새 hue 를 만들지 않고
    # 중립 잉크/브라운을 쓴다 — 8개 hue 어느 것과도 혼동되지 않는다.
    "eda_tree": "#333333",
    "gomea_block": "#8b5e34",
    # xgb_tr 은 xgb_surrogate 의 후속이므로 같은 hue + 점선으로 계보를 표시
    "xgb_tr": "#eb6834",
}

# 기본은 실선. 파생 method 는 점선으로 구분한다 (같은 hue 재사용 시).
OPTIMIZER_LINESTYLES: dict[str, str] = {"xgb_tr": "--"}
TRUE_OPT_COLOR = "#555555"  # 참조선(true optimum)은 중립 회색 점선

# ──────────────────────────────────────────────────────────────────────────────
# 1) 사후(pooled) 점수 — run 간 공정 비교를 위한 재정규화
# ──────────────────────────────────────────────────────────────────────────────

def fit_pooled_scaler(results: list[RunResult]) -> RobustScaler:
    """같은 벤치마크의 모든 run 관측을 합쳐 하나의 스케일러를 적합한다."""
    Y_all = np.vstack([r.Y_raw for r in results])
    return RobustScaler().fit(Y_all)


def pooled_scores(r: RunResult, scaler: RobustScaler, scorer_name: str) -> np.ndarray:
    """run 하나의 전체 관측을 pooled 스케일러 기준으로 다시 점수화한다."""
    return SCORERS[scorer_name](scaler.transform(r.Y_raw))


def best_so_far(scores: np.ndarray) -> np.ndarray:
    """평가 순서를 따라가며 '지금까지의 최고 점수' 곡선을 만든다."""
    return np.maximum.accumulate(scores)


# ──────────────────────────────────────────────────────────────────────────────
# 2) 토너먼트
# ──────────────────────────────────────────────────────────────────────────────

#: 스테이지별 (벤치마크, seed 개수, 진출 비율). 마지막 스테이지는 전원 완주.
TOURNAMENT_STAGES: list[tuple[str, int, float]] = [
    (TOURNAMENT_ORDER[0], 3, 0.5),   # bm1_easy   : 3 seeds → 상위 50% 진출
    (TOURNAMENT_ORDER[1], 5, 0.5),   # bm2_medium : 5 seeds → 상위 50% 진출
    (TOURNAMENT_ORDER[2], 10, 1.0),  # bm3_hard   : 10 seeds → 최종 순위
]


def run_stage(
    optimizer_names: list[str],
    benchmark_name: str,
    n_seeds: int,
    budget: int,
    scorer_name: str,
) -> tuple[list[RunResult], pl.DataFrame]:
    """한 스테이지의 모든 (optimizer × seed) run 을 수행하고 순위표를 만든다.

    Returns:
        results : 이 스테이지의 모든 RunResult
        ranking : optimizer 별 mean/std 최종 best score (내림차순 정렬)
    """
    results: list[RunResult] = []
    for name in optimizer_names:
        for seed in range(n_seeds):
            r = run_single(name, benchmark_name, seed, budget, scorer_name)
            results.append(r)
            print(f"    {name:>16s} seed={seed} done ({r.elapsed_sec:.1f}s)")

    # pooled 스케일러 기준으로 최종 best score 를 계산해 순위를 매긴다
    scaler = fit_pooled_scaler(results)
    rows = [
        {
            "optimizer": r.optimizer,
            "seed": r.seed,
            "final_best": float(pooled_scores(r, scaler, scorer_name).max()),
        }
        for r in results
    ]
    ranking = (
        pl.DataFrame(rows)
        .group_by("optimizer")
        .agg(
            pl.col("final_best").mean().alias("mean_best"),
            pl.col("final_best").std().alias("std_best"),
        )
        .sort("mean_best", descending=True)
    )
    return results, ranking


def run_tournament(
    budget: int, scorer_name: str
) -> tuple[dict[str, list[RunResult]], dict]:
    """3단계 토너먼트 전체를 실행한다.

    Returns:
        all_results : 벤치마크 이름 → 그 스테이지의 RunResult 목록
        summary     : 스테이지별 순위/생존자/챔피언 (json 직렬화 가능)
    """
    contenders = list(OPTIMIZERS.keys())
    all_results: dict[str, list[RunResult]] = {}
    summary: dict = {"stages": [], "scorer": scorer_name, "budget": budget}

    for stage_i, (bm, n_seeds, keep_frac) in enumerate(TOURNAMENT_STAGES, 1):
        print(f"\n[Stage {stage_i}] {bm} — {len(contenders)} optimizers × {n_seeds} seeds")
        results, ranking = run_stage(contenders, bm, n_seeds, budget, scorer_name)
        all_results[bm] = results

        ranked = ranking["optimizer"].to_list()
        n_keep = max(1, int(np.ceil(len(ranked) * keep_frac)))
        survivors = ranked[:n_keep]
        print(f"  ranking: {ranked}")
        print(f"  → 진출: {survivors}")

        summary["stages"].append(
            {
                "benchmark": bm,
                "n_seeds": n_seeds,
                "ranking": ranking.to_dicts(),
                "survivors": survivors,
            }
        )
        contenders = survivors

    summary["champion"] = contenders[0]
    print(f"\n★ champion: {summary['champion']}")
    return all_results, summary


# ──────────────────────────────────────────────────────────────────────────────
# 3) top-3 추천 + confirmation 재측정
# ──────────────────────────────────────────────────────────────────────────────

def recommend_top3(
    champion: str,
    benchmark_name: str,
    budget: int,
    scorer_name: str,
    n_confirm: int = 10,
    init_seeds: tuple[int, ...] = (101, 202, 303),
) -> dict:
    """챔피언 optimizer 를 서로 다른 init seed 로 3회 실행해 후보 3개를 뽑고,
    각 후보를 n_confirm 회 반복 재측정(노이즈 포함)해 신뢰도를 확인한다.

    - 다양성은 'init 을 다르게 한 독립 실행'으로 확보한다 (사용자 합의 사항).
    - 탐색 중에는 반복 측정을 하지 않고, confirmation 에서만 반복 측정한다.
    """
    print(f"\n[Top-3] {champion} on {benchmark_name}, init_seeds={init_seeds}")
    runs = [
        run_single(champion, benchmark_name, s, budget, scorer_name)
        for s in init_seeds
    ]
    scaler = fit_pooled_scaler(runs)

    # 각 run 의 best 해 = pooled 점수 기준 그 run 내 최고 관측
    candidates = []
    for r in runs:
        ps = pooled_scores(r, scaler, scorer_name)
        best_i = int(np.argmax(ps))
        candidates.append({"x": r.X[best_i], "search_score": float(ps[best_i]),
                           "init_seed": r.seed})

    # confirmation: 별도 노이즈 시드의 벤치마크에서 각 후보를 반복 재측정
    calc = BENCHMARKS[benchmark_name](noise_seed=999_999)
    for cand in candidates:
        Y_rep = np.vstack([convert_y_raw(calc.evaluate(cand["x"]))
                           for _ in range(n_confirm)])
        s_rep = SCORERS[scorer_name](scaler.transform(Y_rep))
        cand["confirm_scores"] = s_rep.tolist()
        cand["confirm_mean"] = float(s_rep.mean())
        cand["confirm_std"] = float(s_rep.std())
        cand["confirm_y0_mean"] = Y_rep.mean(axis=0).tolist()
        cand["confirm_z_mean"] = scaler.transform(Y_rep).mean(axis=0).tolist()

    # 최종 순위는 confirmation 평균 점수 기준 (탐색 점수는 노이즈에 낚였을 수 있음)
    candidates.sort(key=lambda c: c["confirm_mean"], reverse=True)
    for rank, c in enumerate(candidates, 1):
        c["rank"] = rank
        c["x"] = c["x"].tolist()  # json 직렬화용
        print(f"  #{rank}: confirm={c['confirm_mean']:.4f}±{c['confirm_std']:.4f} "
              f"(search={c['search_score']:.4f}, init_seed={c['init_seed']})")
    return {"benchmark": benchmark_name, "champion": champion,
            "n_confirm": n_confirm, "candidates": candidates}


# ──────────────────────────────────────────────────────────────────────────────
# 4) true optimum — 챔피언을 무노이즈로 장기 실행한 참조값
# ──────────────────────────────────────────────────────────────────────────────

#: 히스토리 전체를 매 ask/tell 마다 재처리하는 surrogate 계열은 100K 스케일에서
#: 비용이 비현실적이므로, true optimum 계산에서는 자동으로 대체한다.
_SLOW_AT_SCALE = {"tpe", "xgb_surrogate"}


def compute_true_optimum(
    champion: str,
    benchmark_name: str,
    scorer_name: str,
    n_iters: int = 100_000,
    fallback: str = "ga",
) -> dict:
    """무노이즈 벤치마크에서 n_iters 회 평가로 얻은 최고 해를 참조 최적값으로.

    주의: 매 tell 전체 히스토리를 재점수하는 기본 동작은 100K 스케일에서
    O(N²) 이므로, rescore_interval 로 스케일러 재적합을 1000 평가마다로 줄인다
    (결과 랭킹은 raw y0 로 저장하므로 참조값 품질에 영향 없음).
    """
    opt_name = fallback if champion in _SLOW_AT_SCALE else champion
    if opt_name != champion:
        print(f"  (champion '{champion}' 은 100K 스케일에 부적합 → '{opt_name}' 사용)")

    space = SearchSpace()
    calc = BENCHMARKS[benchmark_name](noise_seed=0)
    opt = OPTIMIZERS[opt_name](space, total_budget=n_iters,
                               scorer_name=scorer_name, rescore_interval=1000)

    import time
    state = opt.init_state(seed=0)
    t0 = time.perf_counter()
    n = 0
    while n < n_iters:
        batch, state = opt.ask(state)
        batch = batch[: n_iters - n]
        # 무노이즈 배치 평가 (참조값 계산 전용 — 순차 평가 가정은 실전 run 에만).
        # 구조화 raw 는 tell 내부 convert_y_raw 가 수치화한다.
        state = opt.tell(state, batch, calc.evaluate(batch, noisy=False))
        n += len(batch)

    X_hist, Y_hist = state["X_hist"], state["Y_raw_hist"]
    final_scores = SCORERS[scorer_name](RobustScaler().fit(Y_hist).transform(Y_hist))
    best_i = int(np.argmax(final_scores))
    print(f"  true optimum done: {len(X_hist)} evals, "
          f"{time.perf_counter() - t0:.0f}s")
    return {
        "benchmark": benchmark_name,
        "optimizer_used": opt_name,
        "n_iters": int(len(X_hist)),
        "best_x": X_hist[best_i].tolist(),
        # 점수는 스케일러 의존적이므로 raw y0 을 저장하고,
        # 시각화 시점에 그 실험의 pooled 스케일러로 다시 점수화한다.
        "best_y0_noiseless": Y_hist[best_i].tolist(),
    }


def polish_true_optimum(scorer_name: str = "chebyshev") -> None:
    """저장된 true optimum 들을 무노이즈 전수 폴리시로 다듬어 국소최적을 인증한다.

    절차: 각 벤치마크의 best_x 에서 출발해
      (1) 1-hop 전수: 각 컬럼의 모든 다른 레벨 (컬럼당 card−1 개) — 컬럼
          단위 골짜기(trap)는 여기서 바로 건너뛴다
      (2) 2-swap 전수: 모든 컬럼 쌍 × 레벨 조합
    개선이 없을 때까지 반복 → "2-swap 국소최적" 인증. 점수는 해당 벤치마크
    실험 히스토리(pooled scaler) 기준 — 장기 탐색이 자기 스케일러로 최적화한
    해가 리포트 스케일러 기준으로는 국소최적이 아닐 수 있어서 이 단계가
    참조선의 품질을 크게 올린다 (실측: bm4 +0.10, bm5 +0.08).

    비용: 전부 무노이즈 해석 평가라 벤치마크당 수십 초 수준.
    """
    path = RESULTS_DIR / "true_optimum.json"
    entries = json.loads(path.read_text())
    space = SearchSpace()
    for e in entries:
        bm = e["benchmark"]
        hist = RESULTS_DIR / f"history_{bm}.parquet"
        if not hist.exists():
            print(f"  [{bm}] 실험 히스토리 없음 → 건너뜀")
            continue
        df = pl.read_parquet(hist)
        scaler = RobustScaler().fit(df.select(list(OBJECTIVE_NAMES)).to_numpy())
        calc = BENCHMARKS[bm](noise_seed=0)
        scorer = SCORERS[scorer_name]

        def f(X):
            Y = convert_y_raw(calc.evaluate(np.atleast_2d(X), noisy=False))
            return scorer(scaler.transform(Y))

        x = np.asarray(e["best_x"], dtype=np.int64)
        s = float(f(x)[0])
        s0 = s
        while True:
            cands = [  # (1) 1-hop 전수 (signed 값 범위 순회)
                np.where(np.arange(space.n_cols) == c, v, x)
                for c in range(space.n_cols)
                for v in range(space.x_min[c], space.x_max[c] + 1) if v != x[c]
            ]
            sc = f(np.array(cands))
            if sc.max() > s + 1e-12:
                x, s = np.array(cands)[np.argmax(sc)], float(sc.max())
                continue
            cands = []  # (2) 2-swap 전수
            for a in range(space.n_cols):
                for b in range(a + 1, space.n_cols):
                    for va in range(space.x_min[a], space.x_max[a] + 1):
                        for vb in range(space.x_min[b], space.x_max[b] + 1):
                            if va != x[a] or vb != x[b]:
                                y = x.copy()
                                y[a], y[b] = va, vb
                                cands.append(y)
            sc = f(np.array(cands))
            if sc.max() > s + 1e-12:
                x, s = np.array(cands)[np.argmax(sc)], float(sc.max())
            else:
                break  # 1-hop·2-swap 모두 개선 불가 → 인증 완료
        e["best_x"] = x.tolist()
        e["best_y0_noiseless"] = convert_y_raw(calc.evaluate(x, noisy=False))[0].tolist()
        e["certified"] = "2swap_local_optimum"
        print(f"  [{bm}] {s0:.4f} → {s:.4f} (2-swap 국소최적 인증)")
    path.write_text(json.dumps(entries, indent=2))


def load_true_optima() -> dict[str, dict]:
    """저장된 true optimum 파일이 있으면 {benchmark: info} 로 읽어온다."""
    path = RESULTS_DIR / "true_optimum.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return {d["benchmark"]: d for d in json.load(f)}


# ──────────────────────────────────────────────────────────────────────────────
# 5) 결과 저장 (polars)
# ──────────────────────────────────────────────────────────────────────────────

def save_histories(all_results: dict[str, list[RunResult]]) -> None:
    """모든 run 의 평가 히스토리(raw y0)를 벤치마크별 parquet 으로 저장한다.

    스키마: optimizer, seed, eval(1-based), y11..y23 — X 는 용량 대비 활용도가
    낮아 저장하지 않는다 (top-3 의 X 는 top3.json 에 별도 저장됨).
    """
    for bm, results in all_results.items():
        frames = []
        for r in results:
            df = pl.DataFrame(r.Y_raw, schema=list(OBJECTIVE_NAMES))
            frames.append(
                df.with_columns(
                    pl.lit(r.optimizer).alias("optimizer"),
                    pl.lit(r.seed).alias("seed"),
                    pl.arange(1, len(r.Y_raw) + 1).alias("eval"),
                )
            )
        pl.concat(frames).write_parquet(RESULTS_DIR / f"history_{bm}.parquet")


# ──────────────────────────────────────────────────────────────────────────────
# 6) 시각화 (vis/*.png)
# ──────────────────────────────────────────────────────────────────────────────

def _style_axes(ax: plt.Axes) -> None:
    """공통 스타일: 그리드/축은 흐리게(recessive), 데이터가 주인공."""
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bbbbbb")
    ax.tick_params(colors="#555555", labelsize=9)


def _true_opt_score(
    bm: str, true_optima: dict, scaler: RobustScaler, scorer_name: str
) -> float | None:
    """true optimum 의 raw y0 을 이 실험의 pooled 스케일러로 점수화한다."""
    if bm not in true_optima:
        return None
    y0 = np.asarray(true_optima[bm]["best_y0_noiseless"])[None, :]
    return float(SCORERS[scorer_name](scaler.transform(y0))[0])


def plot_convergence(
    bm: str,
    results: list[RunResult],
    scorer_name: str,
    true_optima: dict,
) -> None:
    """수렴 곡선: 평가 횟수 대비 best-so-far pooled 점수 (seed 평균 ± std 밴드)."""
    scaler = fit_pooled_scaler(results)
    budget = max(len(r.Y_raw) for r in results)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    by_opt: dict[str, list[np.ndarray]] = {}
    for r in results:
        curve = best_so_far(pooled_scores(r, scaler, scorer_name))
        # seed 간 길이가 같지만, 방어적으로 budget 길이로 패딩(마지막 값 유지)
        if len(curve) < budget:
            curve = np.pad(curve, (0, budget - len(curve)), mode="edge")
        by_opt.setdefault(r.optimizer, []).append(curve)

    for name in OPTIMIZER_COLORS:  # 고정 순서 → 범례/색 항상 동일
        if name not in by_opt:
            continue
        curves = np.vstack(by_opt[name])
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        x = np.arange(1, budget + 1)
        color = OPTIMIZER_COLORS[name]
        ax.plot(x, mean, color=color, linewidth=1.8, label=name,
                linestyle=OPTIMIZER_LINESTYLES.get(name, "-"))
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12,
                        linewidth=0)

    t = _true_opt_score(bm, true_optima, scaler, scorer_name)
    if t is not None:
        ax.axhline(t, color=TRUE_OPT_COLOR, linestyle="--", linewidth=1.2)
        ax.annotate(f"true optimum ≈ {t:.3f}", xy=(0.99, t),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=8, color=TRUE_OPT_COLOR)

    n_seeds = len(next(iter(by_opt.values())))
    ax.set_xlabel("evaluations")
    ax.set_ylabel(f"best-so-far {scorer_name} score (pooled scaling)")
    ax.set_title(f"Convergence on {bm} (mean ± std over {n_seeds} seeds)",
                 fontsize=11)
    _style_axes(ax)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(VIS_DIR / f"convergence_{bm}.png")
    plt.close(fig)


def plot_ranking(bm: str, results: list[RunResult], scorer_name: str,
                 true_optima: dict) -> None:
    """스테이지 순위: optimizer 별 최종 best 점수 (seed 평균, 점 = 개별 seed)."""
    scaler = fit_pooled_scaler(results)
    finals: dict[str, list[float]] = {}
    for r in results:
        finals.setdefault(r.optimizer, []).append(
            float(pooled_scores(r, scaler, scorer_name).max())
        )
    names = sorted(finals, key=lambda n: np.mean(finals[n]))  # 낮은 것부터 위로

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.8), dpi=150)
    ypos = np.arange(len(names))
    for i, name in enumerate(names):
        vals = finals[name]
        color = OPTIMIZER_COLORS[name]
        # 파생 method(점선 계열)는 빗금으로 본가와 구분한다
        hatch = "//" if name in OPTIMIZER_LINESTYLES else None
        ax.barh(i, np.mean(vals), height=0.55, color=color, alpha=0.85,
                edgecolor="white" if hatch else "none", hatch=hatch)
        ax.scatter(vals, [i] * len(vals), s=14, color="#333333", zorder=3,
                   alpha=0.7)  # 개별 seed 분포를 점으로 노출 (안정성 확인용)
        # 값 라벨은 seed 점들과 겹치지 않게 막대 위쪽에 살짝 띄운다
        ax.annotate(f"{np.mean(vals):.3f}", xy=(np.mean(vals), i),
                    xytext=(6, 9), textcoords="offset points",
                    va="center", fontsize=8, color="#333333")

    t = _true_opt_score(bm, true_optima, scaler, scorer_name)
    if t is not None:
        ax.axvline(t, color=TRUE_OPT_COLOR, linestyle="--", linewidth=1.2)
        ax.annotate(f"true optimum ≈ {t:.3f}", xy=(t, len(names) - 0.4),
                    xytext=(-6, 0), textcoords="offset points", fontsize=8,
                    color=TRUE_OPT_COLOR, ha="right", va="center")

    ax.set_yticks(ypos, names, fontsize=9)
    ax.set_xlabel(f"final best {scorer_name} score (dots = individual seeds)")
    ax.set_title(f"Final ranking on {bm}", fontsize=11)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(VIS_DIR / f"ranking_{bm}.png")
    plt.close(fig)


def plot_top3(top3: dict) -> None:
    """top-3 confirmation 결과: 반복 측정 점수 분포 + 목적별 정규화 프로파일."""
    cands = top3["candidates"]
    # 후보 3개의 고정 색 (categorical 팔레트 앞 3슬롯)
    cand_colors = ["#2a78d6", "#1baf7a", "#eda100"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

    # ── 왼쪽: confirmation 반복 측정 점수 (점 = 개별 측정, 굵은 선 = 평균) ──
    for i, c in enumerate(cands):
        scores = np.asarray(c["confirm_scores"])
        jitter = (np.random.default_rng(i).uniform(-0.08, 0.08, len(scores)))
        ax1.scatter(i + jitter, scores, s=18, color=cand_colors[i], alpha=0.65)
        ax1.hlines(c["confirm_mean"], i - 0.2, i + 0.2,
                   color=cand_colors[i], linewidth=2.5)
        # 주석은 점 무리와 겹치지 않게 평균선 왼쪽 바깥에 붙인다
        ax1.annotate(f"{c['confirm_mean']:.3f}±{c['confirm_std']:.3f}",
                     xy=(i - 0.24, c["confirm_mean"]), ha="right", va="center",
                     fontsize=8, color="#333333")
    ax1.set_xticks(range(len(cands)),
                   [f"#{c['rank']} (seed {c['init_seed']})" for c in cands],
                   fontsize=9)
    ax1.set_ylabel("confirmation score")
    ax1.set_title(f"Top-3 confirmation ({top3['n_confirm']} re-measurements)",
                  fontsize=11)
    _style_axes(ax1)

    # ── 오른쪽: 목적별 정규화 값 프로파일 (어느 목적을 희생했는지 확인) ──
    n_obj = len(OBJECTIVE_NAMES)
    width = 0.25
    xs = np.arange(n_obj)
    for i, c in enumerate(cands):
        z = np.asarray(c["confirm_z_mean"])
        ax2.bar(xs + (i - 1) * width, z, width=width * 0.92,
                color=cand_colors[i], alpha=0.85,
                label=f"#{c['rank']}")
    ax2.set_xticks(xs, OBJECTIVE_NAMES, fontsize=9)
    ax2.set_ylim(0, 1.22)  # 위쪽 여백은 범례 자리 (막대 최대 1.0)
    ax2.set_ylabel("normalized objective (1 = best)")
    ax2.set_title("Objective profile of each candidate", fontsize=11)
    _style_axes(ax2)
    ax2.legend(fontsize=8, frameon=False, loc="upper center", ncol=3)

    fig.tight_layout()
    fig.savefig(VIS_DIR / "top3_confirmation.png")
    plt.close(fig)


def load_results_from_disk() -> dict[str, list[RunResult]]:
    """results/history_*.parquet 에서 RunResult 들을 복원한다 (--plots-only 용).

    parquet 에는 X 를 저장하지 않으므로 X 는 빈 배열로 채운다 —
    시각화는 raw y0 만 사용하기 때문에 문제없다.
    """
    out: dict[str, list[RunResult]] = {}
    for path in sorted(RESULTS_DIR.glob("history_*.parquet")):
        bm = path.stem.removeprefix("history_")
        df = pl.read_parquet(path)
        results = []
        for (opt, seed), g in df.group_by(["optimizer", "seed"],
                                          maintain_order=True):
            Y = g.sort("eval").select(list(OBJECTIVE_NAMES)).to_numpy()
            results.append(RunResult(str(opt), bm, int(seed),
                                     np.empty((0, 30), dtype=np.int64), Y, 0.0))
        out[bm] = results
    return out


def make_all_plots(
    all_results: dict[str, list[RunResult]],
    scorer_name: str,
    top3: dict | None,
) -> None:
    true_optima = load_true_optima()
    for bm, results in all_results.items():
        plot_convergence(bm, results, scorer_name, true_optima)
        plot_ranking(bm, results, scorer_name, true_optima)
    if top3 is not None:
        plot_top3(top3)
    print(f"\nvis/ 에 그림 저장 완료: {sorted(p.name for p in VIS_DIR.glob('*.png'))}")


# ──────────────────────────────────────────────────────────────────────────────
# 7) matrix 모드 — 난이도 요인별 벤치마크 × optimizer 격자 비교
# ──────────────────────────────────────────────────────────────────────────────

def run_matrix(
    benchmark_names: list[str],
    budget: int,
    scorer_name: str,
    n_seeds: int = 5,
    optimizer_names: list[str] | None = None,
) -> None:
    """토너먼트(탈락식)와 달리 모든 optimizer 를 모든 벤치마크에서 끝까지
    돌려, '어떤 난이도 요인에서 어떤 method 가 강한가'를 직접 본다.

    결과: 벤치마크별 convergence/ranking 그림 + results/matrix.json 순위표.
    """
    optimizer_names = optimizer_names or list(OPTIMIZERS.keys())
    matrix_summary = {}
    for bm in benchmark_names:
        print(f"\n[Matrix] {bm} — {len(optimizer_names)} optimizers × {n_seeds} seeds")
        results, ranking = run_stage(optimizer_names, bm, n_seeds, budget,
                                     scorer_name)
        matrix_summary[bm] = ranking.to_dicts()
        print(f"  ranking: {ranking['optimizer'].to_list()}")
        true_optima = load_true_optima()
        plot_convergence(bm, results, scorer_name, true_optima)
        plot_ranking(bm, results, scorer_name, true_optima)
        # 히스토리도 남긴다 (--plots-only 재생성 및 사후 분석용)
        save_histories({bm: results})

    with open(RESULTS_DIR / "matrix.json", "w") as f:
        json.dump({"budget": budget, "scorer": scorer_name,
                   "n_seeds": n_seeds, "rankings": matrix_summary}, f, indent=2)
    print(f"\nmatrix 결과 저장: results/matrix.json, 그림: vis/")


# ──────────────────────────────────────────────────────────────────────────────
# 8) 진입점
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="optimizer 비교 토너먼트/격자 (단일 run 은 runner.py)"
    )
    parser.add_argument("--budget", type=int, default=800,
                        help="run 당 평가 횟수 (기본 800)")
    parser.add_argument("--scorer", choices=list(SCORERS), default="chebyshev",
                        help="scalarization 선택 (기본 chebyshev)")
    parser.add_argument("--smoke", action="store_true",
                        help="초소형 예산(120 evals)으로 빠른 동작 확인")
    parser.add_argument("--compute-true-optimum", action="store_true",
                        help="챔피언으로 무노이즈 장기 탐색을 돌려 참조 최적값 저장")
    parser.add_argument("--true-opt-iters", type=int, default=100_000,
                        help="true optimum 계산의 평가 횟수 (기본 100K)")
    parser.add_argument("--plots-only", action="store_true",
                        help="실험 없이 results/ 의 저장 데이터로 그림만 재생성")
    parser.add_argument("--polish-true-optimum", action="store_true",
                        help="저장된 true optimum 을 무노이즈 전수 폴리시로 "
                             "다듬어 2-swap 국소최적을 인증하고 갱신")
    parser.add_argument("--matrix", type=str, default=None,
                        help="탈락식 토너먼트 대신 지정 벤치마크들(콤마 구분)에서 "
                             "전체 optimizer 격자 비교 (예: bm3_hard,bm4_deceptive)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="--matrix 모드의 벤치마크당 seed 수 (기본 5)")
    args = parser.parse_args()

    budget = 120 if args.smoke else args.budget
    RESULTS_DIR.mkdir(exist_ok=True)
    VIS_DIR.mkdir(exist_ok=True)

    if args.plots_only:  # 저장된 결과로 시각화만 다시 만든다
        top3_path = RESULTS_DIR / "top3.json"
        top3 = json.loads(top3_path.read_text()) if top3_path.exists() else None
        make_all_plots(load_results_from_disk(), args.scorer, top3)
        return

    if args.polish_true_optimum:  # 참조값 폴리시 (실험 없이 참조만 갱신)
        polish_true_optimum(args.scorer)
        return

    if args.matrix:  # 난이도 요인별 격자 비교 모드
        run_matrix([b.strip() for b in args.matrix.split(",")],
                   budget, args.scorer, n_seeds=args.seeds)
        return

    # ── 토너먼트 ──
    all_results, summary = run_tournament(budget, args.scorer)
    champion = summary["champion"]

    # ── top-3 추천 + confirmation (최종 벤치마크에서) ──
    top3 = recommend_top3(champion, TOURNAMENT_ORDER[-1], budget, args.scorer)

    # ── true optimum (선택) ──
    if args.compute_true_optimum:
        print("\n[True optimum] 무노이즈 장기 탐색 시작 (벤치마크 3종)")
        # 챔피언이 surrogate 계열이라 100K 스케일에 부적합하면, 최종 스테이지
        # 순위에서 가장 높은 non-surrogate optimizer 를 대신 사용한다.
        final_ranking = [d["optimizer"] for d in summary["stages"][-1]["ranking"]]
        fallback = next(
            (n for n in final_ranking if n not in _SLOW_AT_SCALE), "ga"
        )
        infos = [
            compute_true_optimum(champion, bm, args.scorer, args.true_opt_iters,
                                 fallback=fallback)
            for bm in TOURNAMENT_ORDER
        ]
        with open(RESULTS_DIR / "true_optimum.json", "w") as f:
            json.dump(infos, f, indent=2)

    # ── 저장 & 시각화 ──
    save_histories(all_results)
    with open(RESULTS_DIR / "tournament.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(RESULTS_DIR / "top3.json", "w") as f:
        json.dump(top3, f, indent=2)
    make_all_plots(all_results, args.scorer, top3)

    print("\n결과 파일: results/ (parquet/json), 그림: vis/ (png)")


if __name__ == "__main__":
    main()
