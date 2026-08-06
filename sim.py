"""sim.py — 시뮬레이션 측정기. **개발·검증 전용이다.**

REAL 반응표면은 우리가 모른다. 실측은 5분/점이고 지금 수십 점뿐이라, 반응표면이
제대로 도는지를 실측만으로 확인할 수 없다. 그래서 **성질을 아는 가짜 표면**을
하나 두고 거기서 표면 코드를 검증한다 — 정답을 알아야 "얼마나 틀렸나" 를 잴 수
있다.

이건 REAL 표면의 근사가 아니다. 다음 성질만 흉내 낸다:

- **블록 구조** — group1 은 common+set1(15컬럼), group2 는 common+set2(25컬럼)
  에만 의존한다. 무관 컬럼을 넣으면 예측이 나빠져야 정상이다.
- **blob 형상** — 다이아몬드와 타원 중간(`|Δy/a|^p + |Δx/b|^p ≤ 1`, p=1.5)의
  덩어리 하나. 조각나지 않는다.
- **크기가 다른 두 장** — mask1 과 mask2 의 격자가 서로 다르다.
- **관측 노이즈** — 반경에 곱셈 노이즈, 스칼라에 덧셈 노이즈. 같은 X 를 다시
  재면 다른 값이 나온다(반복측정으로 노이즈 바닥을 잴 수 있어야 하므로).
  픽셀 단위 flip 노이즈는 넣지 않는다 — 그건 형상 자체를 왜곡한다.
- **목적 충돌** — blob 을 키우는 방향이 스칼라도 키운다. trade-off 가 없으면
  다목적이 의미가 없다.

실행:
    python sim.py                       # 자가 점검
    python sim.py --out measurements/   # npz 측정 파일 생성 (ingest 로 먹일 수 있다)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from record import N_SCALARS, Blob, Record
from space import SearchSpace

#: 두 마스크의 격자 — 서로 다르다
DEFAULT_SHAPES: tuple[tuple[int, int], tuple[int, int]] = ((64, 48), (96, 120))


class Simulator:
    """X → raw. 성질을 아는 가짜 측정기.

    Args:
        seed        : 지형 시드 — **문제 자체**를 고정한다 (측정 노이즈와 별개)
        noise_level : 반경/스칼라 노이즈 크기 (0 이면 결정적)
        shapes      : 두 마스크의 (H, W)
    """

    def __init__(self, seed: int = 303, noise_level: float = 0.03,
                 shapes=DEFAULT_SHAPES, space: SearchSpace | None = None) -> None:
        self.space = space or SearchSpace()
        self.shapes = tuple(tuple(int(v) for v in s) for s in shapes)
        self.noise_level = float(noise_level)
        n = self.space.n_cols
        r = np.random.default_rng(seed)

        # 그룹별 의존 컬럼 — 이게 이 시뮬레이터의 존재 이유(블록 구조)
        common = self.space.block_cols("common")
        self.cols = {1: np.concatenate([common, self.space.block_cols("set1")]),
                     2: np.concatenate([common, self.space.block_cols("set2")])}

        # latent 항 3종(반높이 · 반너비 · 스칼라) × 그룹 2개 = 6개 지형.
        # 각 지형은 컬럼별 단봉 항의 가중합이다.
        self._peak = r.uniform(0.15, 0.85, (2, 3, n))
        w = r.uniform(0.5, 1.5, (2, 3, n))
        self._w = w / w.sum(axis=2, keepdims=True)
        self._width = r.uniform(0.25, 0.55, (2, 3, n))

    # ─── latent ────────────────────────────────────────────────────────────

    def _latent(self, u: np.ndarray, g: int, k: int) -> np.ndarray:
        """(b, n_cols) unit 좌표 → (b,) [0,1] latent. 의존 컬럼만 본다."""
        cols = self.cols[g]
        gi = g - 1
        d = (u[:, cols] - self._peak[gi, k, cols]) / self._width[gi, k, cols]
        w = self._w[gi, k, cols]
        return (np.exp(-0.5 * d ** 2) * w).sum(axis=1) / w.sum()

    # ─── 측정 ──────────────────────────────────────────────────────────────

    def measure(self, x: np.ndarray, rng: np.random.Generator | None = None) -> dict:
        """X 하나 → {"x", "masks", "scalars"} (ingest 의 reader 계약과 같은 형태).

        `rng` 를 주면 측정 노이즈가 실린다. 같은 X 라도 매번 다르다 —
        반복측정이 노이즈 바닥을 재는 근거가 되려면 그래야 한다.
        """
        x = np.asarray(x, dtype=np.int64).reshape(1, -1)
        u = self.space.to_unit(x)
        masks, scalars = [], []
        for g in (1, 2):
            h, w = self.shapes[g - 1]
            fh, fw, fs = (self._latent(u, g, k)[0] for k in range(3))
            # 반축: latent 를 격자의 8%~45% 로 매핑
            ah = (0.08 + 0.37 * fh) * h / 2
            aw = (0.08 + 0.37 * fw) * w / 2
            # 목적 충돌 — blob 을 키우는 방향이 스칼라도 키운다(스칼라는 최소화)
            s = 0.35 * fs + 0.65 * (fh + fw) / 2
            if rng is not None and self.noise_level > 0:
                ah *= 1.0 + rng.normal(0.0, self.noise_level)
                aw *= 1.0 + rng.normal(0.0, self.noise_level)
                s += rng.normal(0.0, self.noise_level * 0.5)
            masks.append(_blob(h, w, max(ah, 1.0), max(aw, 1.0)))
            scalars.append(float(s))
        return {"x": x[0], "masks": masks,
                "scalars": np.asarray(scalars, dtype=np.float64)}

    def record(self, x, rng=None, src: str = "") -> Record:
        raw = self.measure(x, rng)
        return Record(0, raw["x"], tuple(Blob.from_mask(m) for m in raw["masks"]),
                      raw["scalars"], src)

    def records(self, X, rng=None, src_prefix: str = "sim") -> list[Record]:
        X = np.atleast_2d(np.asarray(X, dtype=np.int64))
        return [self.record(X[i], rng, f"{src_prefix}{i:05d}") for i in range(len(X))]


def _blob(h: int, w: int, ah: float, aw: float, p: float = 1.5) -> np.ndarray:
    """다이아몬드(p=1)와 타원(p=2) 중간의 덩어리 하나."""
    yy, xx = np.ogrid[:h, :w]
    return ((np.abs(yy - (h - 1) / 2) / ah) ** p
            + (np.abs(xx - (w - 1) / 2) / aw) ** p) <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 설계된 관측 — 실측을 흉내 낸 표본 (스크리닝 + 무작위 + 반복)
# ──────────────────────────────────────────────────────────────────────────────


def design(sim: Simulator, rng: np.random.Generator, n_random: int = 40,
           n_repeat: int = 8, n_screen: int | None = None) -> list[Record]:
    """반응표면 개발에 쓸 표본을 뽑는다.

    세 블록은 각각 다른 질문에 답한다:
      · 스크리닝 — 기준점에서 한 컬럼만 끝까지 올린 점. 주효과를 컬럼당 1점으로
      · 무작위   — 지형 전반의 눈금
      · 반복측정 — 같은 X 를 여러 번. **관측 노이즈 바닥**을 재는 유일한 근거
    """
    ss = sim.space
    base = ss.x_min.copy()
    X = [base]
    cols = range(ss.n_cols) if n_screen is None else range(min(n_screen, ss.n_cols))
    for c in cols:
        xi = base.copy()
        xi[c] = ss.x_max[c]
        X.append(xi)
    X.extend(ss.sample(rng, n_random))
    x_rep = ss.sample(rng, 1)[0]
    X.extend([x_rep] * n_repeat)
    return sim.records(np.array(X), rng)


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="시뮬레이션 측정기 (개발·검증 전용)")
    ap.add_argument("--out", type=Path, help="npz 측정 파일을 쓸 디렉토리")
    ap.add_argument("--n", type=int, default=60, help="--out 일 때 생성할 점 수")
    ap.add_argument("--seed", type=int, default=303)
    ap.add_argument("--noise", type=float, default=0.03)
    args = ap.parse_args()

    sim = Simulator(seed=args.seed, noise_level=args.noise)
    rng = np.random.default_rng(0)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        X = sim.space.sample(rng, args.n)
        for i in range(args.n):
            raw = sim.measure(X[i], rng)
            np.savez(args.out / f"sim{i:05d}.npz", x=raw["x"],
                     mask1=raw["masks"][0], mask2=raw["masks"][1],
                     scalars=raw["scalars"])
        print(f"[생성] {args.out} — 측정 파일 {args.n}개")
        print(f"       다음: python ingest.py --src {args.out} --out data/obs.jsonl")
        raise SystemExit(0)

    import score

    ss = sim.space

    # ─ 블록 구조: 무관 블록을 바꿔도 그 그룹의 목적은 안 움직여야 한다 ─
    x0 = ss.sample(rng, 1)[0]
    r0 = sim.record(x0)                       # 노이즈 없이 (rng=None)
    set2_cols = ss.block_cols("set2")
    x1 = x0.copy()
    x1[set2_cols] = ss.x_max[set2_cols]       # group2 컬럼만 전부 바꾼다
    r1 = sim.record(x1)
    assert r0.blobs[0].area == r1.blobs[0].area, "set2 를 바꿨는데 group1 blob 이 변함"
    assert r0.scalars[0] == r1.scalars[0], "set2 를 바꿨는데 group1 스칼라가 변함"
    assert r0.blobs[1].area != r1.blobs[1].area, "set2 를 바꿨는데 group2 가 안 변함"
    print("[OK] 블록 구조 — set2 변경이 group1 에 영향 없음, group2 에는 영향 있음")

    x2 = x0.copy()
    common = ss.block_cols("common")
    x2[common] = ss.x_max[common]
    r2 = sim.record(x2)
    assert r2.blobs[0].area != r0.blobs[0].area and r2.blobs[1].area != r0.blobs[1].area
    print("[OK] 블록 구조 — common 변경이 두 그룹 모두에 영향")

    # ─ blob 성질: 덩어리 하나, 크기가 다른 두 장 ─
    for k, (h, w) in enumerate(sim.shapes):
        b = r0.blobs[k]
        assert b.shape == (h, w)
        rows = np.flatnonzero(b.mask.any(axis=1))
        assert len(b.runs) == len(rows), "행당 run 이 1개가 아님 — 덩어리가 조각남"
        assert rows[-1] - rows[0] + 1 == len(rows), "행이 끊김 — 덩어리가 하나가 아님"
    assert sim.shapes[0] != sim.shapes[1]
    print(f"[OK] blob — 덩어리 하나(행당 run 1개), 두 장 크기 {sim.shapes[0]} ≠ {sim.shapes[1]}")

    # ─ 결정성 vs 노이즈 ─
    assert sim.record(x0).blobs[0].area == r0.blobs[0].area, "rng 없이도 값이 흔들림"
    rep = [sim.record(x0, rng) for _ in range(30)]
    areas = np.array([r.blobs[0].area for r in rep], dtype=float)
    ss1 = np.array([r.scalars[0] for r in rep])
    assert areas.std() > 0 and ss1.std() > 0, "반복측정인데 노이즈가 0 — 바닥을 못 잰다"
    print(f"[OK] 노이즈 — 결정적 재생 OK, 반복 30회 area1 sd={areas.std():.1f} "
          f"({areas.std() / areas.mean():.1%}), s1 sd={ss1.std():.4f}")

    # ─ 목적 충돌: blob 이 큰 해가 스칼라도 크다(trade-off 가 실재) ─
    recs = sim.records(ss.sample(rng, 300), rng)
    Y = score.get("area")(recs)
    c = np.corrcoef(Y[:, 0], Y[:, 2])[0, 1]
    assert c > 0.3, f"blob 과 스칼라가 충돌하지 않음 (상관 {c:.2f}) — 다목적이 무의미"
    print(f"[OK] 목적 충돌 — corr(area1, s1) = {c:+.2f} (클수록 trade-off 가 뚜렷)")

    # ─ 설계 표본 ─
    ds = design(sim, np.random.default_rng(1))
    xs = np.array([r.x for r in ds])
    uniq = len({r.x.tobytes() for r in ds})
    print(f"[OK] design — {len(ds)}점 (유일 x {uniq}, 반복 {len(ds) - uniq + 1}회 1점)")

    Yd = score.get("area")(ds)
    print(f"[참고] 설계 표본 목적 범위 — "
          + "  ".join(f"{n}[{Yd[:, k].min():.3g}, {Yd[:, k].max():.3g}]"
                      for k, n in enumerate(score.get('area').names)))
