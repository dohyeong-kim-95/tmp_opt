"""space.py — signed 정수 범위 탐색 공간 (SearchSpace).

각 column 은 [x_min[c], x_max[c]] 구간의 **정수 값**을 갖는다 (음수 허용,
unit step). cardinality 는 저장하지 않고 `x_max − x_min + 1` 로 유도한다.

┌──────────────────────────────────────────────────────────────────────────┐
│ ⚠️  signed 범위 — [0, card) 산술 금지                                     │
│   X 는 레벨 인덱스가 아니라 **값 자체**다. 다음이 항상 성립해야 한다:      │
│     · sample 결과 ∈ [x_min, x_max]  (0 하한 아님)                         │
│     · clip 은 [x_min, x_max] 로 클램프 (0, card−1 아님)                   │
│     · to_unit(x_min) == 0, to_unit(x_max) == 1                           │
│   값을 배열 인덱스로 쓰려면 반드시 `x − x_min` 오프셋을 거칠 것            │
│   (예: arr[c, x[c] − x_min[c]]).                                          │
└──────────────────────────────────────────────────────────────────────────┘

사용 예:
    ss = SearchSpace()                 # 기본 30컬럼 문제 기하
    x  = ss.sample(rng, n=10)          # (10, n_cols) 정수, 각 열 ∈ [x_min, x_max]
    xc = ss.clip(x + noise)            # 범위 밖 값을 클램프
    xu = ss.to_unit(x)                 # [0,1] 등간격 매핑 (모델/optimizer 입력용)
"""

from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 기본 문제 기하 — 30컬럼, 곱 ≈ 1.1×10^15 (cardinality 는 기존 tmp_opt 와 동일,
# 값 범위만 0 기준이 아니라 0 을 중심으로 한 signed 구간으로 정의한다)
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CARDINALITIES: tuple[int, ...] = (
    # common: col 0–9
    30, 6, 5, 4, 4, 3, 3, 2, 2, 2,
    # set1: col 10–14
    4, 4, 4, 4, 4,
    # set2: col 15–29
    4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
)

# 중심 부근 signed 구간: card c → [−(c//2), −(c//2)+c−1]. 예: 30 → [−15, 14],
# 4 → [−2, 1], 2 → [−1, 0]. 음수 절반이 실제로 존재해 signed 처리 누락이
# 기본 설정에서 바로 드러난다.
_DEFAULT_X_MIN: tuple[int, ...] = tuple(-(c // 2) for c in _DEFAULT_CARDINALITIES)
_DEFAULT_X_MAX: tuple[int, ...] = tuple(
    lo + c - 1 for lo, c in zip(_DEFAULT_X_MIN, _DEFAULT_CARDINALITIES)
)

#: 블록별 (start, end) 인덱스 범위 — end 는 exclusive.
_DEFAULT_BLOCKS: dict[str, tuple[int, int]] = {
    "common": (0, 10),
    "set1": (10, 15),
    "set2": (15, 30),
}


class SearchSpace:
    """signed 정수 범위 탐색 공간.

    속성:
        n_cols        : int — column 개수
        x_min / x_max : (n_cols,) int64 — 각 column 의 최소/최대값 (음수 허용)
        cardinalities : (n_cols,) int64 — x_max − x_min + 1 (유도값)
        blocks        : dict — {이름: (start, end)}, end 는 exclusive
        log10_size    : float — 전체 조합 공간 크기의 log10
    """

    def __init__(
        self,
        x_min: "np.ndarray | tuple[int, ...] | list[int] | None" = None,
        x_max: "np.ndarray | tuple[int, ...] | list[int] | None" = None,
        blocks: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        self.x_min = np.asarray(
            _DEFAULT_X_MIN if x_min is None else x_min, dtype=np.int64
        )
        self.x_max = np.asarray(
            _DEFAULT_X_MAX if x_max is None else x_max, dtype=np.int64
        )
        assert self.x_min.ndim == 1 and self.x_min.shape == self.x_max.shape, \
            "x_min/x_max 는 같은 길이의 1차원 배열이어야 함"
        assert (self.x_max >= self.x_min).all(), "x_max ≥ x_min 이어야 함"

        self.n_cols = int(len(self.x_min))
        self.cardinalities = self.x_max - self.x_min + 1  # 유도값 (저장 금지)

        self.blocks = dict(_DEFAULT_BLOCKS if blocks is None else blocks)
        for name, (start, end) in self.blocks.items():
            assert 0 <= start < end <= self.n_cols, \
                f"블록 {name!r} 범위 ({start}, {end}) 가 [0, {self.n_cols}] 를 벗어남"

    # ─── 속성 ──────────────────────────────────────────────────────────────

    @property
    def log10_size(self) -> float:
        """전체 조합 공간 크기의 log10."""
        return float(np.sum(np.log10(self.cardinalities)))

    def block_cols(self, name: str) -> np.ndarray:
        """블록의 (start, end) 범위를 컬럼 인덱스 배열로 편다."""
        start, end = self.blocks[name]
        return np.arange(start, end, dtype=np.int64)

    # ─── 샘플링 / 변환 ─────────────────────────────────────────────────────

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """각 열이 [x_min[c], x_max[c]] 균등 랜덤인 (n, n_cols) 정수 배열."""
        return rng.integers(
            self.x_min[None, :], self.x_max[None, :] + 1, size=(n, self.n_cols)
        )

    def clip(self, x: np.ndarray) -> np.ndarray:
        """범위 밖 값을 [x_min, x_max] 로 클램프한다.

        dtype 은 입력을 따른다 — float 변이(x + noise)를 넣으면 float 로
        클램프만 하므로, 정수 격자가 필요하면 호출자가 반올림 후 다시 clip:
            ss.clip(np.rint(x + noise).astype(np.int64))
        """
        return np.clip(x, self.x_min, self.x_max)

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        """정수 값 → [0, 1] float 등간격 매핑. to_unit(x_min)=0, to_unit(x_max)=1.

        card=1 인 퇴화 column(구간 폭 0)은 중립값 0.5 로 매핑한다.
        """
        span = (self.x_max - self.x_min).astype(np.float64)
        x = np.asarray(x, dtype=np.float64)
        return np.where(span > 0, (x - self.x_min) / np.maximum(span, 1.0), 0.5)


if __name__ == "__main__":
    # 자가 점검: spec 의 불변식들을 기본 기하에서 확인한다.
    ss = SearchSpace()
    rng = np.random.default_rng(42)

    assert ss.n_cols == 30
    assert np.array_equal(ss.cardinalities, ss.x_max - ss.x_min + 1)
    assert (ss.x_min < 0).any(), "기본 기하는 signed 범위를 실제로 포함해야 함"
    assert ss.blocks == {"common": (0, 10), "set1": (10, 15), "set2": (15, 30)}

    # sample: shape / dtype / 범위 (0 하한이 아니라 x_min 하한)
    x = ss.sample(rng, n=1000)
    assert x.shape == (1000, ss.n_cols) and np.issubdtype(x.dtype, np.integer)
    assert (x >= ss.x_min).all() and (x <= ss.x_max).all()
    assert ss.sample(rng).shape == (1, ss.n_cols)  # n 생략 → (1, n_cols)
    # 각 열이 자기 범위의 양 끝을 실제로 친다 (1000 샘플이면 card≤30 에서 충분)
    assert np.array_equal(x.min(axis=0), ss.x_min)
    assert np.array_equal(x.max(axis=0), ss.x_max)

    # clip: 정수/실수 모두 [x_min, x_max] 클램프
    assert np.array_equal(ss.clip(ss.x_max + 5), ss.x_max)
    assert np.array_equal(ss.clip(ss.x_min - 5), ss.x_min)
    xf = ss.clip(x + rng.normal(0, 3, x.shape))
    assert (xf >= ss.x_min).all() and (xf <= ss.x_max).all()

    # to_unit: 등간격, 끝점 매핑
    u = ss.to_unit(x)
    assert u.shape == x.shape and (u >= 0).all() and (u <= 1).all()
    assert np.allclose(ss.to_unit(ss.x_min[None, :]), 0.0)
    assert np.allclose(ss.to_unit(ss.x_max[None, :]), 1.0)
    # signed 오프셋 검증: 값 x 의 unit 좌표 == (x − x_min) / (card − 1)
    c0 = 0  # card 30, 범위 [−15, 14]
    assert np.isclose(ss.to_unit(ss.x_min)[c0], 0.0)
    assert np.isclose(ss.to_unit(ss.x_min + 1)[c0], 1.0 / 29.0)

    # 파라미터화: 사용자 지정 기하 (card=1 퇴화 column 포함)
    tiny = SearchSpace(x_min=[-3, 0, 5], x_max=[3, 0, 9],
                       blocks={"a": (0, 2), "b": (2, 3)})
    assert np.array_equal(tiny.cardinalities, [7, 1, 5])
    assert np.allclose(tiny.to_unit(np.array([[0, 0, 7]])), [[0.5, 0.5, 0.5]])
    assert np.array_equal(tiny.block_cols("a"), [0, 1])

    print(f"[OK] SearchSpace — n_cols={ss.n_cols}, "
          f"log10_size={ss.log10_size:.2f}, "
          f"x_min[:5]={ss.x_min[:5].tolist()}, x_max[:5]={ss.x_max[:5].tolist()}")
