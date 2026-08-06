"""score.py — raw → 목적값. **교체 가능한 계층이다.**

점수 정의는 확정된 게 아니다. 지금은 면적이지만 max 폭·높이일 수도 있고 앞으로
또 달라질 수 있다. 그래서 정의를 여기 한 곳에 가두고, 저장 계층(record.py)에는
raw 만 남긴다 — 정의가 바뀌면 **파일은 그대로 두고 여기만 갈아끼운 뒤 표면을
reset** 한다.

목적 개수도 scorer 가 정한다. 면적 기준이면 4목적(blob 2 + 스칼라 2), 폭·높이
기준이면 6목적이다. 하류(반응표면·optimizer)는 `names` / `senses` 만 보고 돌아야
하고, 6목적을 하드코딩하면 안 된다.

방향: blob 지표는 **최대화**(+1), 스칼라는 **최소화**(−1).

모든 지표는 `Blob` 의 RLE 에서 바로 계산된다 — 마스크를 펼치지 않으므로 정의를
바꿔 전량 재채점해도 사실상 공짜다.

사용:
    from score import SCORERS
    sc = SCORERS["area"]()
    Y = sc(records)              # (n, sc.n_obj) float64
    sc.names, sc.senses          # ("area1","area2","s1","s2"), (+1,+1,−1,−1)

자가 점검:
    python score.py
"""

from __future__ import annotations

import numpy as np

from record import N_SCALARS, Record


class Scorer:
    """raw → 목적값. 하위 클래스는 `names`/`senses`/`_row` 만 정의한다."""

    #: 목적 이름 — 하류가 이것만 보고 돌아야 한다
    names: tuple[str, ...] = ()
    #: +1 최대화 / −1 최소화, `names` 와 같은 길이
    senses: tuple[int, ...] = ()

    def __init_subclass__(cls, **kw) -> None:
        super().__init_subclass__(**kw)
        if len(cls.names) != len(cls.senses):
            raise TypeError(f"{cls.__name__}: names 와 senses 길이가 다름")
        if not set(cls.senses) <= {1, -1}:
            raise TypeError(f"{cls.__name__}: senses 는 +1 / −1 만 허용")

    @property
    def n_obj(self) -> int:
        return len(self.names)

    def _row(self, r: Record) -> list[float]:
        raise NotImplementedError

    def __call__(self, records) -> np.ndarray:
        """Record 하나 또는 여럿 → (n, n_obj) float64."""
        if isinstance(records, Record):
            records = [records]
        Y = np.array([self._row(r) for r in records], dtype=np.float64)
        Y = Y.reshape(-1, self.n_obj)
        if not np.isfinite(Y).all():
            raise ValueError(f"{type(self).__name__}: 목적값에 비유한값")
        return Y

    def describe(self) -> str:
        return "  ".join(f"{n}{'↑' if s > 0 else '↓'}"
                         for n, s in zip(self.names, self.senses))


class AreaScorer(Scorer):
    """면적 기준 — 현재 기본. blob 의 True 픽셀 수를 최대화한다.

    blob 이 "크다" 를 면적으로 읽는다. 폭·높이와 달리 형상 전체를 반영하므로
    같은 bounding box 안에서 더 꽉 찬 덩어리를 선호한다.
    """

    names = ("area1", "area2", "s1", "s2")
    senses = (1, 1, -1, -1)

    def _row(self, r: Record) -> list[float]:
        return [r.blobs[0].area, r.blobs[1].area, *r.scalars]


class ExtentScorer(Scorer):
    """max 폭·높이 기준 — 기존 repo 의 6목적 정의(y11..y23)와 같은 뜻.

    면적과 달리 덩어리가 얼마나 **멀리 뻗었는지**를 본다. 속이 빈 형상도
    높게 나오므로 면적과 다른 해를 고른다.
    """

    names = ("h1", "w1", "s1", "h2", "w2", "s2")
    senses = (1, 1, -1, 1, 1, -1)

    def _row(self, r: Record) -> list[float]:
        b1, b2 = r.blobs
        return [b1.height, b1.width, r.scalars[0],
                b2.height, b2.width, r.scalars[1]]


class AreaExtentScorer(Scorer):
    """면적 + 폭·높이 둘 다 — 어느 지표가 맞는지 아직 못 고를 때.

    목적이 6개로 늘어 파레토 전선이 넓어진다(수렴이 느려진다). 지표를 고르기
    전 탐색용이지, 이걸 최종 정의로 쓸 이유는 없다.
    """

    names = ("area1", "h1", "w1", "s1", "area2", "h2", "w2", "s2")
    senses = (1, 1, 1, -1, 1, 1, 1, -1)

    def _row(self, r: Record) -> list[float]:
        b1, b2 = r.blobs
        return [b1.area, b1.height, b1.width, r.scalars[0],
                b2.area, b2.height, b2.width, r.scalars[1]]


#: 이름 → scorer. 점수 정의를 바꾼다는 건 여기서 다른 걸 고른다는 뜻이다.
SCORERS: dict[str, type[Scorer]] = {
    "area": AreaScorer,
    "extent": ExtentScorer,
    "area+extent": AreaExtentScorer,
}

#: 현재 기본 정의
DEFAULT = "area"


def get(name: str | None = None) -> Scorer:
    name = DEFAULT if name is None else name
    if name not in SCORERS:
        raise ValueError(f"알 수 없는 scorer {name!r} — {sorted(SCORERS)}")
    return SCORERS[name]()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    from record import Blob
    from space import SearchSpace

    rng = np.random.default_rng(0)
    ss = SearchSpace()

    def _diamond(h, w, ah, aw):
        yy, xx = np.ogrid[:h, :w]
        return ((np.abs(yy - h / 2) / ah) ** 1.5
                + (np.abs(xx - w / 2) / aw) ** 1.5) <= 1.0

    recs = []
    for i in range(200):
        b1 = Blob.from_mask(_diamond(64, 48, rng.uniform(4, 28), rng.uniform(4, 22)))
        b2 = Blob.from_mask(_diamond(96, 120, rng.uniform(4, 44), rng.uniform(4, 56)))
        recs.append(Record(i, ss.sample(rng, 1)[0], (b1, b2),
                           rng.uniform(0, 1, N_SCALARS)))

    for name in SCORERS:
        sc = get(name)
        Y = sc(recs)
        assert Y.shape == (200, sc.n_obj)
        print(f"[OK] {name:<12} {sc.n_obj}목적  {sc.describe()}")

    # 지표가 실제 마스크와 일치하는가 (RLE 계산이 옳은지 — 여기가 틀리면 전부 틀린다)
    sc = get("area")
    Y = sc(recs[:20])
    for k, r in enumerate(recs[:20]):
        assert Y[k, 0] == r.blobs[0].mask.sum() and Y[k, 1] == r.blobs[1].mask.sum()
    print("[OK] 면적이 펼친 마스크의 True 픽셀 수와 일치 (RLE 직접 계산 검증)")

    # 방향 규약 — 하류가 senses 만 보고 돌 수 있어야 한다
    for name in SCORERS:
        sc = get(name)
        assert len(sc.names) == len(sc.senses) == sc.n_obj
        assert set(sc.senses) <= {1, -1}
        assert sum(1 for s in sc.senses if s < 0) == N_SCALARS, "스칼라는 최소화 목적"
    print("[OK] names/senses 규약 — 스칼라 2개가 모두 최소화 방향")

    # 정의 교체 비용 — 전량 재채점이 사실상 공짜여야 raw-only 저장이 성립한다
    t0 = time.perf_counter()
    for _ in range(50):
        get("area+extent")(recs)
    dt = (time.perf_counter() - t0) / 50
    print(f"[비용] 200점 × 8목적 전량 재채점 {dt * 1e3:.1f} ms "
          f"— 정의를 바꿔도 파일 재작성 없이 즉시 다시 잰다")

    # 면적과 폭·높이는 실제로 다른 해를 고른다 (지표 선택이 의미 있다는 확인)
    a = get("area")(recs)[:, 0]
    e = get("extent")(recs)[:, 0] * get("extent")(recs)[:, 1]
    print(f"[참고] area1 최댓점 idx={int(a.argmax())}, "
          f"h1×w1 최댓점 idx={int(e.argmax())} "
          f"— {'다름' if a.argmax() != e.argmax() else '같음'}")
