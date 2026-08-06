"""optimize.py — 다음에 잴 X 를 고른다. **BO, surrogate 는 surface.Surface.**

한 점이 5분이다. 그러니 "다음 한 점" 을 고르는 데 계산을 아낌없이 써도 된다.
GP-BO 는 이 규모(수십~수백 점 × 30컬럼 이산)에서 느리고, 조합공간에서 커널을
고르는 것부터가 일이다. 여기서는 XGB 배깅 surrogate(surface.py) 를 쓰고,
acquisition 만 정석대로 세운다.

──────────────────────────────────────────────────────────────────────────────
acquisition — ParEGO 스칼라화 + Monte-Carlo EI
──────────────────────────────────────────────────────────────────────────────
목적이 4~8개다. 다목적 BO 의 정석은 EHVI 지만 목적 수가 늘면 급격히 비싸진다.
ParEGO 는 매 제안마다 **가중치 λ 를 무작위로 새로 뽑아** 목적을 하나로 접는다.
제안마다 다른 λ 를 쓰므로 여러 번 돌리면 파레토 전선 전체를 훑는다.

    f_λ(y) = min_j λ_j·ỹ_j  +  ρ·Σ_j λ_j·ỹ_j        (augmented Chebyshev, 최대화)

ỹ 는 관측 범위로 [0,1] 정규화하고 sense 를 반영해 **클수록 좋게** 맞춘 값이다.
min 항이 파레토 전선의 오목한 부분까지 집어내고, ρ 항이 약한 파레토 해를 걸러낸다.

EI 는 **Monte-Carlo 로 잰다**:

    y_s ~ N(μ(x), σ(x))   목적별 독립, S개 표본
    EI(x) = mean_s max(0, f_λ(y_s) − f*)

f_λ 가 min 을 포함해서 y 가 정규분포여도 f_λ 는 아니다. 해석적 EI 공식을 쓰면
그 자리에서 틀린다. MC 는 그 비선형성을 그대로 통과시키고, 목적 수가 늘어도
비용이 선형이다.

f* — **관측된 최대가 아니라 그 점들의 사후평균 최대**를 쓴다. 노이즈가 있는
관측에서 최댓값은 위로 편향돼 있어(운 좋게 크게 나온 점) 그대로 쓰면 EI 가
과도하게 보수적이 된다. noisy BO 의 표준 교정이다.

──────────────────────────────────────────────────────────────────────────────
acquisition 최적화 — 후보 풀
──────────────────────────────────────────────────────────────────────────────
10^15 이산공간이라 경사법이 없다. 후보를 만들어 놓고 EI 최댓점을 고른다:

  · 무작위          — 전역 탐험
  · 파레토 전선 변이 — 좋은 점 주변 (변이 폭을 섞어 국소·중거리 둘 다)
  · **블록 교차**    — set1 ⫫ set2 | common 이므로, common 이 같으면 한 점의
                      set1 과 다른 점의 set2 를 붙여도 각 그룹의 성질이 보존된다.
                      문제 구조가 실제로 허락하는 조작이고, 무작위 교차보다
                      훨씬 자주 좋은 점을 만든다.

실행:
    python optimize.py --obs data/obs.jsonl            # 다음 1점 제안
    python optimize.py --obs data/obs.jsonl --q 4      # 4점 (λ 를 다르게 뽑는다)
    python optimize.py --selfcheck                     # 시뮬레이터에서 BO vs 무작위
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import score
from space import SearchSpace
from surface import Surface

#: augmented Chebyshev 의 ρ — 약한 파레토 해를 걸러내는 정도
RHO = 0.05
#: EI 의 Monte-Carlo 표본 수
N_MC = 128


# ──────────────────────────────────────────────────────────────────────────────
# 스칼라화
# ──────────────────────────────────────────────────────────────────────────────


def normalize(Y: np.ndarray, senses, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """목적값 → [0,1], **클수록 좋게**. 범위 밖은 잘리지 않고 그대로 넘는다.

    (자르면 관측 범위를 넘는 개선이 EI 에 안 잡힌다 — 그게 우리가 찾는 것이다)
    """
    sen = np.asarray(senses, dtype=np.float64)
    span = np.maximum(hi - lo, 1e-12)
    z = (np.asarray(Y, dtype=np.float64) - lo) / span
    return np.where(sen > 0, z, 1.0 - z)


def chebyshev(Yn: np.ndarray, lam: np.ndarray, rho: float = RHO) -> np.ndarray:
    """augmented Chebyshev — 클수록 좋다. Yn 은 정규화된 (…, n_obj)."""
    w = lam * Yn
    return w.min(axis=-1) + rho * w.sum(axis=-1)


def sample_weights(rng: np.random.Generator, n_obj: int) -> np.ndarray:
    """단체(simplex) 위 균등 — Dirichlet(1,…,1)."""
    return rng.dirichlet(np.ones(n_obj))


# ──────────────────────────────────────────────────────────────────────────────
# 파레토
# ──────────────────────────────────────────────────────────────────────────────


def pareto_mask(Y: np.ndarray, senses) -> np.ndarray:
    """비지배 해 마스크. Y 는 원래 목적값(정규화 전)."""
    Z = np.asarray(Y, dtype=np.float64) * np.asarray(senses, dtype=np.float64)
    n = len(Z)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = (Z >= Z[i]).all(axis=1) & (Z > Z[i]).any(axis=1)
        if dom.any():
            keep[i] = False
    return keep


def hypervolume_mc(Yn: np.ndarray, rng: np.random.Generator, n: int = 20000) -> float:
    """[0,1]^m 단위 상자 기준 하이퍼볼륨을 Monte-Carlo 로.

    목적 수가 몇이든 같은 비용이고, 정확한 HV 알고리즘 없이 비교에 충분하다.
    (레퍼런스 점은 원점 — 정규화가 관측 범위 기준이므로 상대 비교용이다)
    """
    m = Yn.shape[1]
    P = rng.random((n, m))
    Yc = np.clip(Yn, 0.0, 1.0)
    dominated = np.zeros(n, dtype=bool)
    for i in range(0, len(Yc), 64):
        dominated |= (P[:, None, :] <= Yc[None, i:i + 64, :]).all(axis=2).any(axis=1)
    return float(dominated.mean())


# ──────────────────────────────────────────────────────────────────────────────
# 후보 풀
# ──────────────────────────────────────────────────────────────────────────────


def _mutate(X: np.ndarray, k: np.ndarray, space: SearchSpace,
            rng: np.random.Generator) -> np.ndarray:
    """행마다 k[i] 개 컬럼을 무작위 값으로 바꾼다."""
    X = X.copy()
    n, c = X.shape
    for i in range(n):
        cols = rng.choice(c, size=min(int(k[i]), c), replace=False)
        X[i, cols] = rng.integers(space.x_min[cols], space.x_max[cols] + 1)
    return X


def candidates(surf: Surface, rng: np.random.Generator, *, n_random: int = 12000,
               n_mutate: int = 6000, n_cross: int = 4000) -> np.ndarray:
    """EI 를 잴 후보 풀. 관측점은 뺀다(같은 X 를 또 재자는 제안이 되지 않게)."""
    space = surf.space
    front = surf.X[pareto_mask(surf.Y, surf.scorer.senses)]
    seeds = front if len(front) else surf.X

    parts = [space.sample(rng, n_random)]

    # 변이 — 폭을 섞는다. 1~2컬럼은 국소 미세조정, 그 이상은 중거리 탐색.
    if n_mutate:
        base = seeds[rng.integers(0, len(seeds), n_mutate)]
        k = rng.choice([1, 2, 3, 5, 8], size=n_mutate, p=[.3, .25, .2, .15, .1])
        parts.append(_mutate(base, k, space, rng))

    # 블록 교차 — set1 ⫫ set2 | common 을 실제로 이용한다.
    # common 을 한쪽에서 통째로 가져와야 조건부 독립이 성립한다(common 을 섞으면
    # 두 그룹 모두 근거 없는 점이 된다).
    if n_cross and len(seeds) >= 2:
        a = seeds[rng.integers(0, len(seeds), n_cross)]
        b = seeds[rng.integers(0, len(seeds), n_cross)]
        cross = a.copy()
        s2 = space.block_cols("set2")
        cross[:, s2] = b[:, s2]
        # 절반은 common 도 b 쪽에서 (그러면 set1 만 a 것이 된다)
        half = rng.random(n_cross) < 0.5
        cm = space.block_cols("common")
        cross[np.ix_(half, cm)] = b[np.ix_(half, cm)]
        parts.append(cross)

    C = np.vstack(parts)
    C = np.unique(C, axis=0)
    seen = {r.tobytes() for r in surf.X}
    keep = np.array([r.tobytes() not in seen for r in C])
    return C[keep]


# ──────────────────────────────────────────────────────────────────────────────
# 제안
# ──────────────────────────────────────────────────────────────────────────────


def mc_ei(mu: np.ndarray, sigma: np.ndarray, lam: np.ndarray, best_f: float,
          lo, hi, senses, rng: np.random.Generator, n_mc: int = N_MC) -> np.ndarray:
    """(n_cand,) Monte-Carlo Expected Improvement.

    f_λ 는 min 을 품고 있어 정규분포가 아니다 — 해석적 EI 를 쓰면 틀린다.
    """
    nc, m = mu.shape
    out = np.empty(nc)
    step = max(1, (2_000_000) // max(n_mc * m, 1))    # 표본 메모리 상한
    for s in range(0, nc, step):
        e = min(s + step, nc)
        draw = (mu[None, s:e, :]
                + sigma[None, s:e, :] * rng.standard_normal((n_mc, e - s, m)))
        f = chebyshev(normalize(draw, senses, lo, hi), lam)
        out[s:e] = np.maximum(f - best_f, 0.0).mean(axis=0)
    return out


def propose(surf: Surface, q: int = 1, rng: np.random.Generator | None = None,
            *, n_mc: int = N_MC, verbose: bool = False, **cand_kw) -> np.ndarray:
    """다음에 잴 X 를 q 개. 각 제안마다 λ 를 새로 뽑는다(ParEGO).

    λ 가 달라 제안끼리 파레토 전선의 다른 구역을 노린다 — 배치가 한 곳에
    몰리지 않는다. 이미 고른 점은 다음 제안의 후보에서 뺀다.
    """
    rng = rng or np.random.default_rng(0)
    senses = np.asarray(surf.scorer.senses)
    lo, hi = surf.Y.min(axis=0), surf.Y.max(axis=0)

    C = candidates(surf, rng, **cand_kw)
    if not len(C):
        raise RuntimeError("후보가 0개 — 탐색 공간이 이미 다 측정됐거나 풀 설정이 이상하다")
    pred = surf.predict(C)

    # f* 는 관측 최댓값이 아니라 **관측점의 사후평균** 최댓값 (noisy BO 교정)
    mu_obs = surf.predict(surf.X).mu

    picked, taken = [], np.zeros(len(C), dtype=bool)
    for t in range(q):
        lam = sample_weights(rng, surf.n_obj)
        best_f = float(chebyshev(normalize(mu_obs, senses, lo, hi), lam).max())
        ei = mc_ei(pred.mu, pred.sigma, lam, best_f, lo, hi, senses, rng, n_mc)
        ei[taken] = -np.inf
        i = int(np.argmax(ei))
        taken[i] = True
        picked.append(C[i])
        if verbose:
            print(f"  #{t} λ={np.round(lam, 3).tolist()} EI={ei[i]:.4g} "
                  f"status={pred.status[i].tolist()} d={pred.d_min[i, 0]:.3f}")
    return np.array(picked, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────


def _selfcheck(budget: int = 40, seed: int = 0, n_random_ctrl: bool = True) -> None:
    """시뮬레이터에서 BO 가 무작위 탐색을 실제로 이기는가.

    이게 유일한 증명이다 — acquisition 이 옳게 짜였는지는 수렴으로만 드러난다.
    """
    from sim import Simulator, design

    sc = score.get()
    senses = np.asarray(sc.senses)
    sim0 = Simulator(seed=303, noise_level=0.03)

    # ─ 공정한 비교를 위한 두 가지 통제 ─
    # (1) 정규화 기준을 **전 run 공통**으로 고정한다. run 마다 자기 범위로
    #     정규화하면 HV 가 "자기 범위 안에서의 전선 폭" 이 되어 run 간 비교가
    #     성립하지 않는다 (좋은 점을 찾아 범위를 넓힌 쪽이 오히려 손해를 본다).
    ref = sc(sim0.records(sim0.space.sample(np.random.default_rng(7), 1500),
                          np.random.default_rng(8)))
    REF_LO, REF_HI = ref.min(axis=0), ref.max(axis=0)
    # (2) 초기 설계를 **전 arm 공통**으로 고정한다. 출발점이 다르면 무엇이
    #     이겼는지 알 수 없다.
    INIT = design(sim0, np.random.default_rng(1234), n_random=25, n_repeat=6,
                  n_screen=10)

    def hv(recs) -> float:
        Y = sc(recs)
        f = pareto_mask(Y, senses)
        return hypervolume_mc(normalize(Y[f], senses, REF_LO, REF_HI),
                              np.random.default_rng(999))

    def run(kind: str, seed: int) -> list[float]:
        rng = np.random.default_rng(seed)
        sim = Simulator(seed=303, noise_level=0.03)
        recs = list(INIT)
        curve = []
        surf = None
        for t in range(budget):
            curve.append(hv(recs))
            if kind == "random":
                x = sim.space.sample(rng, 1)
            else:
                # 표면 재적합. 캘리브레이션(LOO)은 비싸서 주기적으로만.
                if surf is None or t % 5 == 0:
                    surf = Surface.fit(recs, scorer=sc, n_ensemble=6,
                                       loo_ensemble=3, seed=seed)
                    alpha = surf.alpha
                else:
                    surf = Surface.fit(recs, scorer=sc, n_ensemble=6,
                                       seed=seed, alpha=alpha)
                x = propose(surf, 1, rng, n_random=6000, n_mutate=3000, n_cross=2000)
            recs = recs + sim.records(x, rng)
        curve.append(hv(recs))
        return curve

    import time
    print(f"[비교] 예산 {budget} evals · 목적 {sc.n_obj}개 — 하이퍼볼륨(클수록 좋음)")
    print(f"    통제: 초기 설계 {len(INIT)}점 공통 · 정규화 기준 공통(무작위 1500점)")
    t0 = time.perf_counter()
    bo = run("bo", seed)
    t_bo = time.perf_counter() - t0
    rnd = [run("random", seed + 100 + k) for k in range(5)] if n_random_ctrl else []

    marks = [0, budget // 4, budget // 2, 3 * budget // 4, budget]
    print(f"    {'eval':>6} {'BO':>10} {'무작위(5회 평균)':>18}")
    for m in marks:
        r = np.mean([c[m] for c in rnd]) if rnd else float("nan")
        print(f"    {m:>6} {bo[m]:>10.4f} {r:>18.4f}")
    if rnd:
        r_end = np.mean([c[-1] for c in rnd])
        r_sd = np.std([c[-1] for c in rnd])
        gain = (bo[-1] - r_end) / max(r_sd, 1e-9)
        print(f"    최종 BO {bo[-1]:.4f} vs 무작위 {r_end:.4f}±{r_sd:.4f} "
              f"— 무작위 산포의 {gain:.1f}배 우위")
        assert bo[-1] > r_end, "BO 가 무작위를 못 이겼다 — acquisition 을 의심할 것"
    print(f"    BO {budget} evals 소요 {t_bo:.0f}s "
          f"({t_bo / budget:.1f}s/eval — 실측 300s/eval 대비 무시할 수준)")


def main() -> None:
    ap = argparse.ArgumentParser(description="다음에 잴 X 를 고른다 (XGB surrogate BO)")
    ap.add_argument("--obs", type=Path, default=Path("data/obs.jsonl"))
    ap.add_argument("--q", type=int, default=1, help="제안 개수")
    ap.add_argument("--scorer", default=None, help=f"점수 정의 — {sorted(score.SCORERS)}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=None,
                    help="거리 팽창 계수 (생략 시 LOO 캘리브레이션)")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--budget", type=int, default=40, help="--selfcheck 의 예산")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck(budget=args.budget, seed=args.seed)
        return

    surf = Surface.from_jsonl(args.obs, scorer=score.get(args.scorer),
                              seed=args.seed, alpha=args.alpha)
    rep = surf.report()
    print(f"[표면] 관측 {rep['n_obs']}점 (유일 {rep['n_unique_x']}) · "
          f"목적 {surf.scorer.describe()}")
    print(f"       α={rep['alpha']} · d_ref={rep['d_ref']} · "
          f"커버리지 {rep['coverage']}")
    if "calibration" in rep:
        c = rep["calibration"]
        print(f"       캘리브레이션 std(z)={c['z_std']} (1.0 목표) · "
              f"2σ 안 {c['within_2sigma']:.0%}")
    for w in rep["warnings"]:
        print("  ⚠", w)

    rng = np.random.default_rng(args.seed)
    X = propose(surf, args.q, rng, verbose=True)
    p = surf.predict(X)
    print(f"\n[제안] 다음에 잴 X {len(X)}점")
    for t in range(len(X)):
        print(f"  {X[t].tolist()}")
        print("    " + "  ".join(
            f"{n}={p.mu[t, j]:.4g}±{p.sigma[t, j]:.3g}"
            for j, n in enumerate(surf.scorer.names)))


if __name__ == "__main__":
    main()
