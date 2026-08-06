"""record.py — 관측 저장 형식. **raw 가 유일한 진실이다.**

한 줄이 관측(측정) 하나인 append-only jsonl 이다.

    {"i":0, "x":[...30개...],
     "m1":{"shape":[H1,W1],"runs":[[row,col0,len],...]},
     "m2":{"shape":[H2,W2],"runs":[...]},
     "s":[s1,s2], "src":"meas_00042.dat"}

┌──────────────────────────────────────────────────────────────────────────┐
│ 목적값(y)을 저장하지 않는다                                                │
│   점수 정의(면적이냐 max 폭·높이냐 …)는 언제든 바뀐다. y 를 파일에 넣으면  │
│   바뀌는 순간 전부 stale 이 되고, 어느 줄이 어느 정의로 쓰인 건지 알 수    │
│   없어진다. raw(blob 원형 + 스칼라)만 남기면 점수는 언제든 다시 재고,      │
│   반응표면도 언제든 reset 된다. 점수 정의는 score.py 소관.                 │
└──────────────────────────────────────────────────────────────────────────┘

왜 이 형식인가:

- **행 단위 run-length.** blob 은 다이아몬드와 타원 중간의 덩어리 **하나**라
  행마다 연속 구간이 하나뿐이다 — RLE 가 거의 최적이고 무손실이다. 무엇보다
  `[row, col0, len]` 은 눈으로 읽어도 의미가 보인다(base64 는 불투명한 덩어리).
- **shape 를 줄마다 기록한다.** 마스크 두 장은 **서로 크기가 다르다**. 전역
  상수 하나로 격자를 가정하면 그 자리에서 깨진다. 측정끼리는 크기가 같으므로
  파일 전체의 일관성은 `validate()` 가 검사한다.
- **면적·폭·높이를 RLE 에서 바로 잰다.** 마스크를 펼치지 않고 계산되므로
  (`Blob.area` 등) 점수 정의가 바뀌어 전량 재채점해도 비용이 거의 없다.
- **append-only.** 관측이 늘어도 전체 재작성이 없고, 사람이 손으로 한 줄
  추가해도 유효하다. 파일이 곧 상태다.

- `src` 는 출처 측정 파일명이다. 측정 메타데이터가 아니라 **머지 운영 정보**로,
  ingest 가 이미 넣은 파일을 건너뛰는 근거다(멱등). 없어도 형식은 유효하다.

자가 점검:
    python record.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from space import SearchSpace

#: 한 측정이 내는 것 — boolean blob 2장 + 스칼라 2개
N_MASKS = 2
N_SCALARS = 2


# ──────────────────────────────────────────────────────────────────────────────
# Blob — RLE 로 들고 있는 boolean 덩어리. 마스크는 필요할 때만 펼친다.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Blob:
    """boolean 마스크 한 장 (행 단위 RLE).

    속성:
        shape : (H, W)
        runs  : (n, 3) int64 — 각 행이 [row, col0, len]

    지표(area/height/width)는 **RLE 에서 직접** 계산한다. 마스크를 펼치지
    않으므로 수천 점 전량 재채점도 즉시 끝난다.
    """

    shape: tuple[int, int]
    runs: np.ndarray

    # ─── 지표 (마스크를 펼치지 않는다) ──────────────────────────────────────

    @property
    def area(self) -> int:
        """True 픽셀 수."""
        return int(self.runs[:, 2].sum()) if len(self.runs) else 0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(row0, col0, row1, col1) — end 는 exclusive. 빈 blob 은 전부 0."""
        if not len(self.runs):
            return (0, 0, 0, 0)
        r, c0, ln = self.runs[:, 0], self.runs[:, 1], self.runs[:, 2]
        return (int(r.min()), int(c0.min()), int(r.max()) + 1, int((c0 + ln).max()))

    @property
    def height(self) -> int:
        """bounding box 높이."""
        r0, _, r1, _ = self.bbox
        return r1 - r0

    @property
    def width(self) -> int:
        """bounding box 너비."""
        _, c0, _, c1 = self.bbox
        return c1 - c0

    # ─── 마스크 왕복 ───────────────────────────────────────────────────────

    @property
    def mask(self) -> np.ndarray:
        """(H, W) boolean 으로 펼친다 (무손실)."""
        h, w = self.shape
        m = np.zeros((h, w), dtype=bool)
        for r, c0, ln in self.runs:
            m[r, c0:c0 + ln] = True
        return m

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "Blob":
        """(H, W) boolean → Blob (무손실)."""
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"마스크는 2차원이어야 함: {mask.shape}")
        h, w = mask.shape
        # 행마다 양옆을 0 으로 패딩 → run 이 행 경계를 넘지 않고 시작/끝이 1:1
        pad = np.zeros((h, w + 2), dtype=np.int8)
        pad[:, 1:-1] = mask
        d = np.diff(pad.ravel().astype(np.int16))
        s = np.flatnonzero(d == 1) + 1
        e = np.flatnonzero(d == -1) + 1
        runs = np.stack([s // (w + 2), s % (w + 2) - 1, e - s], axis=1).astype(np.int64)
        return cls((h, w), runs)

    # ─── 직렬화 ────────────────────────────────────────────────────────────

    def to_json(self) -> dict:
        return {"shape": [int(self.shape[0]), int(self.shape[1])],
                "runs": self.runs.tolist()}

    @classmethod
    def from_json(cls, enc: dict) -> "Blob":
        h, w = (int(v) for v in enc["shape"])
        runs = np.asarray(enc["runs"], dtype=np.int64).reshape(-1, 3)
        if len(runs):
            r, c0, ln = runs[:, 0], runs[:, 1], runs[:, 2]
            bad = (r < 0) | (r >= h) | (c0 < 0) | (ln <= 0) | (c0 + ln > w)
            if bad.any():
                raise ValueError(
                    f"run 이 형상 {h}x{w} 을 벗어남: {runs[bad][0].tolist()}")
        return cls((h, w), runs)


# ──────────────────────────────────────────────────────────────────────────────
# Record — 관측 한 점
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Record:
    """관측 하나 — X 와 그때 나온 raw.

    같은 x 가 여러 줄에 나올 수 있다. 그게 **반복측정**이고, 관측 노이즈
    바닥을 재는 유일한 근거다(TRUE 반응표면에 노이즈가 있으므로).
    """

    i: int
    x: np.ndarray            # (n_cols,) int64
    blobs: tuple[Blob, ...]  # 길이 N_MASKS — 서로 크기가 다를 수 있다
    scalars: np.ndarray      # (N_SCALARS,) float64
    src: str = ""

    def to_json(self) -> dict:
        rec = {"i": int(self.i), "x": self.x.tolist()}
        for k, b in enumerate(self.blobs, start=1):
            rec[f"m{k}"] = b.to_json()
        rec["s"] = [float(v) for v in self.scalars]
        if self.src:
            rec["src"] = self.src
        return rec

    @classmethod
    def from_json(cls, rec: dict) -> "Record":
        blobs = tuple(Blob.from_json(rec[f"m{k}"]) for k in range(1, N_MASKS + 1))
        s = np.asarray(rec["s"], dtype=np.float64)
        if s.shape != (N_SCALARS,):
            raise ValueError(f"스칼라는 {N_SCALARS}개여야 함: {s.shape}")
        if not np.isfinite(s).all():
            raise ValueError(f"스칼라에 비유한값: {s.tolist()}")
        return cls(int(rec["i"]), np.asarray(rec["x"], dtype=np.int64),
                   blobs, s, str(rec.get("src", "")))


# ──────────────────────────────────────────────────────────────────────────────
# 파일 입출력 — append-only
# ──────────────────────────────────────────────────────────────────────────────


def append(path, records, *, space: SearchSpace | None = None) -> int:
    """관측을 파일 끝에 이어 쓴다. 반환값은 쓴 뒤의 총 줄 수.

    `i` 는 호출자가 뭘 넣었든 **현재 줄 수에서 이어지도록 다시 매긴다** —
    인덱스는 파일이 소유하는 것이지 레코드가 들고 다니는 게 아니다.
    """
    path = Path(path)
    records = list(records)
    space = space or SearchSpace()
    base = count(path)
    for k, r in enumerate(records):
        _check_x(r.x, space)
        _check_blobs(r.blobs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for k, r in enumerate(records):
            fh.write(json.dumps(
                Record(base + k, r.x, r.blobs, r.scalars, r.src).to_json()) + "\n")
    return base + len(records)


def iter_records(path) -> Iterator[Record]:
    """한 줄씩 흘려 읽는다 (전량을 메모리에 올리지 않는다)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"관측 파일 없음: {path}")
    with path.open() as fh:
        for ln, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} JSON 파싱 실패 — {e}") from e
            try:
                yield Record.from_json(rec)
            except (KeyError, ValueError) as e:
                raise ValueError(f"{path}:{ln} 레코드 형식 오류 — {e}") from e


def read(path) -> list[Record]:
    """전량 로드."""
    return list(iter_records(path))


def count(path) -> int:
    """줄 수 (없는 파일은 0). 파싱하지 않는다."""
    path = Path(path)
    if not path.exists():
        return 0
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())


def sources(path) -> set[str]:
    """이미 들어간 `src` 집합 — ingest 가 멱등하려고 쓴다."""
    if not Path(path).exists():
        return set()
    return {r.src for r in iter_records(path) if r.src}


# ──────────────────────────────────────────────────────────────────────────────
# 검증
# ──────────────────────────────────────────────────────────────────────────────


def _check_x(x: np.ndarray, space: SearchSpace) -> None:
    x = np.asarray(x)
    if x.shape != (space.n_cols,):
        raise ValueError(f"x 는 {space.n_cols}개 정수여야 함: {x.shape}")
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError(f"x 는 정수여야 함: dtype={x.dtype}")
    if (x < space.x_min).any() or (x > space.x_max).any():
        bad = np.flatnonzero((x < space.x_min) | (x > space.x_max))
        raise ValueError(
            f"x 가 탐색 공간을 벗어남 — 컬럼 {bad.tolist()}, "
            f"값 {x[bad].tolist()}, 허용 "
            f"{list(zip(space.x_min[bad].tolist(), space.x_max[bad].tolist()))}")


def _check_blobs(blobs) -> None:
    if len(blobs) != N_MASKS:
        raise ValueError(f"blob 은 {N_MASKS}장이어야 함: {len(blobs)}")


def validate(path, *, space: SearchSpace | None = None) -> dict:
    """파일 전체를 훑어 형식·일관성을 확인한다. 어긋나면 즉시 raise.

    측정끼리 마스크 크기는 **같아야 한다**(장치가 고정). 장 사이의 크기는
    달라도 된다 — 실제로 두 장은 서로 크기가 다르다.
    """
    space = space or SearchSpace()
    shapes: list[tuple[int, int] | None] = [None] * N_MASKS
    n, dup_src = 0, set()
    seen_src: set[str] = set()
    x_seen: dict[bytes, int] = {}
    for r in iter_records(path):
        if r.i != n:
            raise ValueError(f"{path}: i 가 어긋남 — {n}번째 줄의 i={r.i}")
        _check_x(r.x, space)
        _check_blobs(r.blobs)
        for k, b in enumerate(r.blobs):
            if shapes[k] is None:
                shapes[k] = b.shape
            elif b.shape != shapes[k]:
                raise ValueError(
                    f"{path}:{n + 1} m{k + 1} 크기가 다름 — {b.shape} ≠ {shapes[k]} "
                    "(측정끼리 마스크 크기는 같아야 한다)")
        if r.src:
            if r.src in seen_src:
                dup_src.add(r.src)
            seen_src.add(r.src)
        key = r.x.tobytes()
        x_seen[key] = x_seen.get(key, 0) + 1
        n += 1
    if dup_src:
        raise ValueError(f"{path}: src 중복 {sorted(dup_src)[:5]} — 머지가 겹쳤다")
    if n == 0:
        raise ValueError(f"{path}: 관측이 0개")
    n_rep = sum(c for c in x_seen.values() if c > 1)
    return {"n": n, "n_unique_x": len(x_seen), "n_repeated": n_rep,
            "mask_shapes": [tuple(s) for s in shapes if s is not None],
            "n_with_src": len(seen_src)}


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    ss = SearchSpace()

    def _diamond(h, w, ah, aw, cy=None, cx=None):
        """다이아몬드와 타원 중간 덩어리 하나 — 실제 blob 성격에 맞춘 표본."""
        cy = h / 2 if cy is None else cy
        cx = w / 2 if cx is None else cx
        yy, xx = np.ogrid[:h, :w]
        u = np.abs(yy - cy) / max(ah, 1e-9)
        v = np.abs(xx - cx) / max(aw, 1e-9)
        return (u ** 1.5 + v ** 1.5) <= 1.0        # p=1 다이아몬드, p=2 타원

    # ─ Blob 왕복 무손실 + 지표가 마스크와 일치 ─
    for _ in range(20):
        h, w = int(rng.integers(20, 90)), int(rng.integers(20, 90))
        m = _diamond(h, w, rng.uniform(3, h / 2), rng.uniform(3, w / 2))
        b = Blob.from_mask(m)
        assert np.array_equal(b.mask, m), "RLE 왕복이 무손실이 아님"
        assert b.area == int(m.sum())
        rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        assert b.height == rows[-1] - rows[0] + 1 and b.width == cols[-1] - cols[0] + 1
        # 덩어리 하나 → 행당 run 1개 (RLE 가 조밀하다는 근거)
        assert len(b.runs) == len(rows), f"행당 run 이 1개가 아님: {len(b.runs)} vs {len(rows)}"
    print("[OK] Blob — RLE 왕복 무손실, 지표(area/height/width) 일치, 행당 run 1개")

    # 빈 blob / 전체 True 같은 극단도 왕복
    for m in (np.zeros((7, 5), bool), np.ones((7, 5), bool)):
        assert np.array_equal(Blob.from_mask(m).mask, m)
    assert Blob.from_mask(np.zeros((7, 5), bool)).area == 0
    print("[OK] Blob — 빈 blob / 전체 True 극단 왕복")

    # ─ 파일 왕복 + append + 검증 ─
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "obs.jsonl"
        H1, W1, H2, W2 = 64, 48, 96, 120      # 두 장의 크기가 서로 다르다

        def _make(i, x=None):
            x = ss.sample(rng, 1)[0] if x is None else x
            b1 = Blob.from_mask(_diamond(H1, W1, rng.uniform(5, 25), rng.uniform(5, 20)))
            b2 = Blob.from_mask(_diamond(H2, W2, rng.uniform(5, 40), rng.uniform(5, 50)))
            return Record(i, x, (b1, b2), rng.uniform(0, 1, N_SCALARS), f"m{i:04d}.dat")

        n = append(p, [_make(i) for i in range(10)])
        assert n == 10 and count(p) == 10
        n = append(p, [_make(i) for i in range(10, 15)])   # 이어쓰기
        assert n == 15 and count(p) == 15

        back = read(p)
        assert [r.i for r in back] == list(range(15)), "i 가 파일 순서대로 매겨지지 않음"
        assert back[0].blobs[0].shape == (H1, W1) and back[0].blobs[1].shape == (H2, W2)
        print(f"[OK] 파일 — append 10→15줄, 마스크 크기 {(H1, W1)} / {(H2, W2)} 공존")

        rep = validate(p)
        assert rep["n"] == 15 and rep["mask_shapes"] == [(H1, W1), (H2, W2)]
        assert len(sources(p)) == 15

        # 반복측정(같은 x 두 줄)이 형식상 유효하고 집계에 잡히는가
        x_rep = back[0].x
        append(p, [Record(0, x_rep, back[0].blobs, back[0].scalars, "rep.dat")])
        rep2 = validate(p)
        assert rep2["n"] == 16 and rep2["n_unique_x"] == 15 and rep2["n_repeated"] == 2
        print(f"[OK] 검증 — {rep2['n']}줄, 유일 x {rep2['n_unique_x']}, 반복측정 {rep2['n_repeated']}줄")

        # fail-loud: 크기가 다른 측정 / 범위 밖 x
        try:
            append(p, [Record(0, ss.sample(rng, 1)[0],
                              (Blob.from_mask(np.zeros((8, 8), bool)), back[0].blobs[1]),
                              np.zeros(N_SCALARS), "bad.dat")])
            validate(p)
            raise AssertionError("마스크 크기 불일치를 잡지 못했다")
        except ValueError as e:
            assert "크기가 다름" in str(e)
        try:
            append(p, [Record(0, ss.x_max + 1, back[0].blobs, np.zeros(N_SCALARS))])
            raise AssertionError("범위 밖 x 를 잡지 못했다")
        except ValueError as e:
            assert "탐색 공간을 벗어남" in str(e)
        print("[OK] fail-loud — 마스크 크기 불일치 / 범위 밖 x 즉시 거부")

        kb = p.stat().st_size / 1e3
        px = (H1 * W1 + H2 * W2) * 16
        print(f"[크기] 16줄 {kb:.0f} KB — 원시 boolean {px / 1e3:.0f} KB 대비 "
              f"{kb / (px / 1e3):.2f}배 (덩어리 하나라 RLE 가 조밀하다)")
