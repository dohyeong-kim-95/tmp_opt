"""make_dataset.py — 반응표면의 재료가 될 관측 데이터셋을 만든다.

실제 TEST 는 비싸다. 그래서 **무엇을 몇 점 측정할지**를 먼저 설계해 두고,
그 설계대로 뽑은 (X, y_raw) 를 **obs.jsonl** 로 남긴다 (append-only 텍스트,
마스크 원형까지 한 줄 안에 — 형식은 calculator.save_observations).
`calculator.SurfaceCalculator.from_jsonl` 이 이걸 그대로 먹는다.

설계 (세 블록) — 각각 다른 질문에 답한다:

  A. one-hot 스크리닝 (1 + n_cols 점)
     기준점 = 전 컬럼 x_min. 거기서 **한 컬럼만 x_max** 로 올린 점을 컬럼마다
     하나씩. "이 컬럼을 끝까지 올리면 y 가 얼마나 움직이나" = 주효과 크기를
     컬럼당 1점으로 잰다. 30차원에서 무작위 표본보다 훨씬 싸게 얻는 정보다.
     (상호작용은 못 본다 — 그건 B 의 몫)

  B. 무작위 표본 (기본 60점)
     지형 전반의 눈금. 보간의 재료이자, A 가 못 보는 조합 효과의 단서.

  C. 기본값 반복측정 (기본 20점)
     같은 X 를 반복해 **관측 노이즈 바닥**을 직접 잰다. 이 값이 없으면
     "모델이 틀린 건지 측정이 시끄러운 건지" 구분할 수 없다. 반응표면의
     spread_gate 와 d_gate 를 정하는 근거이자, 보간 오차의 비교 기준이다.
     기본값은 전 컬럼 0 (signed 범위 [-(c//2), ...] 의 중앙 부근).

관측 장치는 `SimulatedInstrument` — **다이아몬드에 가까운 blob** 을 낸다.
실제 장치가 확보되면 이 클래스만 갈아끼우면 되고, 하류(obs.jsonl → 반응표면)는
그대로다.

실행:
    python make_dataset.py --out obs.jsonl
    python make_dataset.py --out obs.jsonl --n-random 120 --n-repeat 30 --seed 1
    python make_dataset.py --out obs.jsonl --append   # 기존 관측에 이어쓰기
    python make_dataset.py --selfcheck               # 설계·노이즈 요약만 출력
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from calculator import (OBJECTIVE_NAMES, _GROUP1_COLS, _GROUP2_COLS,
                        save_observations)
from space import SearchSpace

#: 관측 마스크 격자 (calculator.SurfaceCalculator.RASTER_GRID 와 같아야 한다)
RASTER_GRID = 128


# ──────────────────────────────────────────────────────────────────────────────
# 관측 장치 (시뮬레이션) — 실제 장치 확보 시 이 클래스만 교체
# ──────────────────────────────────────────────────────────────────────────────

class SimulatedInstrument:
    """X → y_raw. 다이아몬드에 가까운 blob 2장 + 스칼라 2개.

    형상: 반경 프로파일이 `r(θ) = 1 / (|cosθ|/b + |sinθ|/a)` — 이게 정확히
    반축 (a, b) 인 **다이아몬드**(L1 공)다. 여기에 완만한 각도 변조를 얹어
    딱 떨어지는 다각형이 아니라 덩어리(blob) 로 만든다. 변조는 X 에 대해
    결정적이라 같은 X 는 같은 형상을 낸다.

    지형: 컬럼별 단봉 항의 가중합 + 목적 충돌 축. 목적 의존성은 블록 구조를
    따른다 (y1x ← common+set1, y2x ← common+set2).

    노이즈: 스칼라에만 가우시안 (`noise_level` × latent std). blob 은
    격자 양자화만 — 가장자리 낱개 픽셀이 튀는 노이즈는 형상 자체를 왜곡해서
    넣지 않는다 (lesson_learned.md 교훈 7).
    """

    def __init__(self, seed: int = 0, noise_level: float = 0.05,
                 grid: int = RASTER_GRID) -> None:
        self.space = SearchSpace()
        self.grid = grid
        self.noise_level = noise_level
        n = self.space.n_cols
        r = np.random.default_rng(seed)

        self._peaks = r.uniform(0.0, 1.0, (6, n))
        w = r.uniform(0.5, 1.5, (6, n))
        mask = np.zeros((6, n))
        mask[0:3, list(_GROUP1_COLS)] = 1.0
        mask[3:6, list(_GROUP2_COLS)] = 1.0
        w *= mask
        self._w = w / w.sum(axis=1, keepdims=True)

        # 목적 충돌 축: 커지면 6목적 전부 커진다 → 최소화 목적엔 손해
        cw = r.uniform(0.5, 1.5, len(self.space.block_cols("common")))
        self._c_w = cw / cw.sum()
        self._c_gain = r.uniform(0.35, 0.5, 6)
        # blob 각도 변조 (형상을 다각형에서 덩어리로) — 목적쌍마다 다른 위상
        self._mod_phase = r.uniform(0, 2 * np.pi, 2)

        self._rng = np.random.default_rng(seed + 1)
        # latent → 픽셀 캘리브레이션은 **설계가 실제로 방문할 범위** 로 잡는다.
        # 무작위 표본의 mu±3sd 로 잡으면 one-hot 기준점(전 컬럼 x_min) 같은
        # 공간의 모서리에서 포화해, 그 설계의 주효과가 통째로 0 이 되어버린다.
        probe_X = np.vstack([self.space.sample(np.random.default_rng(12345), 4096),
                             design_one_hot(self.space)])
        probe = self._latent(self.space.to_unit(probe_X))
        self._lo, self._hi = probe.min(axis=0), probe.max(axis=0)
        self._sd = probe.std(axis=0)

    # ─── 지형 ──────────────────────────────────────────────────────────────

    def _latent(self, u: np.ndarray) -> np.ndarray:
        base = np.einsum("nkc,kc->nk",
                         1.0 - (u[:, None, :] - self._peaks[None]) ** 2, self._w)
        c = u[:, self.space.block_cols("common")] @ self._c_w
        return base + c[:, None] * self._c_gain[None, :]

    def _semi_px(self, lat: np.ndarray, k: int) -> np.ndarray:
        """latent → 반축 픽셀 (단조 — 측정치 최대화 = latent 최대화).

        설계 범위 [lo, hi] 를 픽셀 [8, grid/2−6] 에 선형으로 편다. 포화가
        없으므로 어떤 설계 점에서도 주효과가 측정된다.
        """
        half = self.grid // 2
        t = (lat - self._lo[k]) / (self._hi[k] - self._lo[k] + 1e-12)
        return np.clip(np.rint(8 + (half - 14) * t), 6, half - 4)

    # ─── 렌더 ──────────────────────────────────────────────────────────────

    def _render(self, semi_h: np.ndarray, semi_w: np.ndarray,
                phase: float) -> np.ndarray:
        """다이아몬드형 blob 래스터 (b, G, G)."""
        g = self.grid
        c = (g - 1) / 2.0
        rr, cc = np.mgrid[0:g, 0:g]
        dy, dx = rr - c, cc - c
        rad = np.hypot(dy, dx)
        ang = np.arctan2(dy, dx)
        ca, sa = np.abs(np.cos(ang)), np.abs(np.sin(ang))
        # 완만한 각도 변조 → 딱 떨어지는 다각형이 아니라 덩어리
        mod = 1.0 + 0.08 * np.sin(3.0 * ang + phase)
        out = np.empty((len(semi_h), g, g), dtype=bool)
        for i in range(len(semi_h)):
            a, b = float(semi_h[i]), float(semi_w[i])
            r_theta = mod / (ca / b + sa / a + 1e-12)  # L1 공의 반경 프로파일
            out[i] = rad <= r_theta
        return out

    # ─── 공개 API — calculator 계약과 동일 ─────────────────────────────────

    def evaluate(self, X: np.ndarray, noisy: bool = True) -> dict:
        X = np.atleast_2d(np.asarray(X, dtype=np.int64))
        if X.shape[1] != self.space.n_cols:
            raise ValueError(f"X 는 {self.space.n_cols}컬럼이어야 함: {X.shape}")
        lat = self._latent(self.space.to_unit(X))
        sc = lat[:, [2, 5]]
        if noisy:
            sc = sc + self._rng.normal(0.0, 1.0, sc.shape) * (
                self.noise_level * self._sd[[2, 5]])
        return {
            "mask1": self._render(self._semi_px(lat[:, 0], 0),
                                  self._semi_px(lat[:, 1], 1), self._mod_phase[0]),
            "mask2": self._render(self._semi_px(lat[:, 3], 3),
                                  self._semi_px(lat[:, 4], 4), self._mod_phase[1]),
            "y13": sc[:, 0] * 0.005 + 0.02,
            "y23": sc[:, 1] * 0.005 + 0.02,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 설계 (X 만 만든다 — 측정은 장치가)
# ──────────────────────────────────────────────────────────────────────────────

def design_one_hot(space: SearchSpace) -> np.ndarray:
    """기준점(전 컬럼 x_min) + 컬럼마다 그 하나만 x_max 로 올린 점."""
    base = space.x_min.copy()
    rows = [base]
    for c in range(space.n_cols):
        v = base.copy()
        v[c] = space.x_max[c]
        rows.append(v)
    return np.array(rows, dtype=np.int64)


def design_random(space: SearchSpace, n: int, rng: np.random.Generator) -> np.ndarray:
    """균등 무작위 n 점."""
    return space.sample(rng, n)


def design_repeat(space: SearchSpace, n: int,
                  default: np.ndarray | None = None) -> np.ndarray:
    """기본값 X 를 n 번 — 관측 노이즈 바닥 측정용."""
    x = np.zeros(space.n_cols, dtype=np.int64) if default is None else \
        np.asarray(default, dtype=np.int64)
    if (x < space.x_min).any() or (x > space.x_max).any():
        raise ValueError("기본값이 탐색 공간 범위를 벗어남")
    return np.repeat(x[None, :], n, axis=0)


def build_dataset(instrument, space: SearchSpace, n_random: int = 60,
                  n_repeat: int = 20, seed: int = 0,
                  default: np.ndarray | None = None) -> dict:
    """세 블록을 붙여 측정하고 저장용 페이로드를 만든다."""
    rng = np.random.default_rng(seed)
    blocks = {
        "one_hot": design_one_hot(space),
        "random": design_random(space, n_random, rng),
        "repeat": design_repeat(space, n_repeat, default),
    }
    X = np.vstack([blocks[k] for k in ("one_hot", "random", "repeat")])
    raw = instrument.evaluate(X)          # 노이즈 포함 = 실측 대역

    from optimizer import convert_y_raw   # 측정 정의는 optimizer 소유
    Y = convert_y_raw(raw)
    # 블록 라벨 — 어느 설계에서 나온 점인지 나중에 되짚을 수 있게
    labels = np.concatenate([[k] * len(blocks[k])
                             for k in ("one_hot", "random", "repeat")])
    return {"X": X, "Y": Y, "block": labels,
            "mask1": raw["mask1"], "mask2": raw["mask2"]}


# ──────────────────────────────────────────────────────────────────────────────
# 요약 리포트 — 설계가 실제로 무엇을 알려줬는지
# ──────────────────────────────────────────────────────────────────────────────

def summarize(payload: dict) -> dict:
    """노이즈 바닥(C블록)과 주효과 크기(A블록)를 요약한다."""
    X, Y, block = payload["X"], payload["Y"], payload["block"]
    rep = Y[block == "repeat"]
    oh = Y[block == "one_hot"]
    rnd = Y[block == "random"]

    noise_sd = rep.std(axis=0, ddof=1)                 # 관측 노이즈 바닥
    span = rnd.max(axis=0) - rnd.min(axis=0) + 1e-12   # 신호 범위
    # 주효과: 기준점(one_hot 첫 행) 대비 각 컬럼 1개 변경의 효과.
    # 두 측정치의 **차이** 이므로 노이즈 sd 가 √2 배로 커진다 — 문턱을 그에
    # 맞춰야 무관 컬럼을 과검출하지 않는다.
    effects = np.abs(oh[1:] - oh[0])                   # (n_cols, 6)
    thresh = 3.0 * np.sqrt(2.0) * np.maximum(noise_sd, 1e-12)
    return {
        "n_total": len(X),
        "noise_sd": dict(zip(OBJECTIVE_NAMES, noise_sd.round(6))),
        "noise_over_span": dict(zip(OBJECTIVE_NAMES,
                                    (noise_sd / span).round(4))),
        "effect_median": dict(zip(OBJECTIVE_NAMES,
                                  np.median(effects, axis=0).round(4))),
        # 노이즈보다 확실히 큰 효과를 낸 컬럼 수 (주효과 스크리닝 결과).
        # 노이즈 sd 가 0 인 목적(픽셀 측정 등)에서는 "효과 > 0" 으로 퇴화하는데,
        # 픽셀은 그 자체가 양자화 단위라 그게 곧 유의미한 문턱이다.
        "cols_above_noise": dict(zip(
            OBJECTIVE_NAMES,
            (effects > thresh).sum(axis=0).tolist())),
        "detected_cols": {n: np.flatnonzero(effects[:, k] > thresh[k]).tolist()
                          for k, n in enumerate(OBJECTIVE_NAMES)},
    }


def print_summary(payload: dict) -> None:
    s = summarize(payload)
    b, cnt = np.unique(payload["block"], return_counts=True)
    print(f"[설계] 총 {s['n_total']}점 — " +
          ", ".join(f"{k} {v}" for k, v in zip(b.tolist(), cnt.tolist())))
    print(f"{'목적':>6} {'노이즈 sd':>12} {'/신호범위':>10} "
          f"{'주효과 중앙값':>14} {'노이즈 초과 컬럼':>16}")
    for n in OBJECTIVE_NAMES:
        print(f"{n:>6} {s['noise_sd'][n]:>12.6g} {s['noise_over_span'][n]:>10.4f} "
              f"{s['effect_median'][n]:>14.4g} "
              f"{s['cols_above_noise'][n]:>14d}/30")
    print("  · 노이즈/신호범위가 크면 그 목적은 반응표면 보간이 어렵다")
    print("  · 노이즈 초과 컬럼 수 = 주효과가 확실히 잡힌 컬럼 (one-hot 스크리닝)")


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="반응표면용 관측 데이터셋 생성")
    ap.add_argument("--out", type=Path, default=None,
                    help="저장할 obs.jsonl 경로 (생략 시 저장 없이 요약만)")
    ap.add_argument("--append", action="store_true",
                    help="--out 파일에 이어쓴다 (관측을 늘릴 때)")
    ap.add_argument("--n-random", type=int, default=60, help="무작위 표본 수")
    ap.add_argument("--n-repeat", type=int, default=20,
                    help="기본값 반복측정 수 (노이즈 바닥 추정용)")
    ap.add_argument("--seed", type=int, default=0, help="설계/노이즈 시드")
    ap.add_argument("--instrument-seed", type=int, default=303,
                    help="시뮬레이션 장치의 지형 시드 (문제 자체를 고정)")
    ap.add_argument("--selfcheck", action="store_true",
                    help="설계·노이즈 요약만 출력하고 종료")
    args = ap.parse_args()

    space = SearchSpace()
    inst = SimulatedInstrument(seed=args.instrument_seed)
    payload = build_dataset(inst, space, n_random=args.n_random,
                            n_repeat=args.n_repeat, seed=args.seed)
    print_summary(payload)

    if args.selfcheck:
        # 설계 불변식 확인
        oh = payload["X"][payload["block"] == "one_hot"]
        assert np.array_equal(oh[0], space.x_min), "one-hot 기준점이 x_min 이 아님"
        assert all((oh[i + 1] != space.x_min).sum() == 1 for i in range(len(oh) - 1)), \
            "one-hot 점이 정확히 한 컬럼만 다르지 않음"
        rp = payload["X"][payload["block"] == "repeat"]
        assert (rp == rp[0]).all(), "반복측정 블록의 X 가 서로 다름"
        assert (payload["Y"][payload["block"] == "repeat"][:, [2, 5]].std(axis=0) > 0).all(), \
            "반복측정인데 스칼라 노이즈가 0 — 노이즈 바닥을 못 잰다"
        print("[OK] 설계 불변식 통과 (one-hot / 무작위 / 반복 블록)")

        # one-hot 스크리닝이 블록 구조를 실제로 복원하는지 — 이 설계의 존재 이유
        det = summarize(payload)["detected_cols"]
        g1, g2 = set(_GROUP1_COLS.tolist()), set(_GROUP2_COLS.tolist())
        # 스크리닝은 통계적 판정이라 오검출이 0 이어야 할 이유는 없다 —
        # 3√2σ 문턱에서 무관 컬럼당 오검출 확률은 ~0.2%, 목적당 기대값 <1 개.
        fp = {n: sorted(set(det[n]) - (g1 if n[1] == "1" else g2))
              for n in OBJECTIVE_NAMES}
        n_fp = sum(len(v) for v in fp.values())
        assert n_fp <= 3, f"오검출 과다 — {fp}"
        print(f"[OK] one-hot 스크리닝이 블록 구조 복원 — 오검출 {n_fp}개(허용 3), "
              + ", ".join(f"{n} {len(det[n])}/{len(g)}"
                          for n, g in (("y11", g1), ("y21", g2))))
        return

    if args.out is not None:
        save_observations(args.out, payload["X"], payload["Y"],
                          block=payload["block"],
                          masks={"mask1": payload["mask1"],
                                 "mask2": payload["mask2"]},
                          append=args.append)
        n_lines = sum(1 for _ in args.out.open())
        kb = args.out.stat().st_size / 1e3
        print(f"[저장] {args.out} — {n_lines}줄 누적, {kb:.0f} KB"
              f"{' (이어쓰기)' if args.append else ''}")
        print(f"       다음: python runner.py --optimizer sa "
              f"--surface-data {args.out}")


if __name__ == "__main__":
    main()
