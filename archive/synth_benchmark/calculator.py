"""calculator.py — 벤치마크(문제) 정의 모듈.

이 모듈은 optimizer 비교 실험에 사용할 "계산기(Calculator)"들을 정의한다.
Calculator는 실제 환경에서 '한 번 평가에 비용이 드는 블랙박스 측정기'에 해당하며,
여기서는 난이도가 다른 합성(synthetic) 벤치마크 3종으로 구현한다.

──────────────────────────────────────────────────────────────────────────────
문제 구조 (모든 벤치마크 공통)
──────────────────────────────────────────────────────────────────────────────
- 입력 X : 길이 30의 정수 벡터. i번째 원소는 signed 구간
  [x_min_i, x_max_i] 의 **값** (space.SearchSpace 가 표준 명세).
  내부적으로는 to_unit 으로 [0,1] 에 매핑해 쓰므로 latent 지형은 값 표현과
  무관하다. 전체 조합 공간 크기 ≈ 10^15.
- 블록 구조 (도메인 지식 — optimizer가 활용해도 됨):
    * common (col 0–9)  : 6개 목적 전부에 영향. max/min 간 trade-off가
                          이 블록에 인코딩되어 있다.
    * set1   (col 10–14): y11, y12, y13 에만 영향 (유효차원 15 = common+set1)
    * set2   (col 15–29): y21, y22, y23 에만 영향 (유효차원 25 = common+set2)
    * set1 ⫫ set2 | common (공통 블록을 통해서만 결합)
- 출력 y_raw : **구조화 관측** — boolean 타원 마스크 2장 + 스칼라 2개.
    * mask1 (b,G,G): y11 = max height, y12 = max width  (가우시안 타원 G≥0.5)
    * mask2 (b,G,G): y21 = max height, y22 = max width
    * y13, y23 (b,): 스칼라 (은닉 스케일 적용)
    * 최대화: y11, y12, y21, y22 / 최소화: y13, y23. 마스크→수치 측정은
      optimizer 의 convert_y_raw 이음새 소관 (calculator 는 원형만 낸다).
    * 값의 범위는 사전에 알 수 없다고 가정한다 (스케일러가 온라인으로 추정).
    * 관측 노이즈: 타원 = 경계 밴드 픽셀 random flip + 격자 양자화,
      스칼라 = 주효과(latent) 표준편차의 5% 크기 가우시안.

──────────────────────────────────────────────────────────────────────────────
난이도 설계
──────────────────────────────────────────────────────────────────────────────
- BenchmarkEasy   : 거의 분리가능(separable), 매끄러운 단봉 지형. 완만한 trade-off.
- BenchmarkMedium : 블록 내 pairwise 상호작용 + 완만한 다봉(multimodal) 성분.
- BenchmarkHard   : common↔set 간 강한 교차 상호작용 + 기만적(deceptive) 성분
                    + 고주파 ruggedness. set2(15컬럼)가 병목.

사용 예:
    calc = BenchmarkHard(noise_seed=0)
    y0 = calc.evaluate(x)              # 노이즈 포함 관측 (실전과 동일)
    y0 = calc.evaluate(x, noisy=False) # true optimum 계산용 (무노이즈)
"""

from __future__ import annotations

import numpy as np

from space import SearchSpace

# ──────────────────────────────────────────────────────────────────────────────
# 탐색 공간 — space.SearchSpace 가 유일한 표준 명세다.
# (어떤 기하를 쓰든 SearchSpace 를 통과해 표준화된 속성만 소비한다)
# ──────────────────────────────────────────────────────────────────────────────

_SPACE = SearchSpace()  # 기본 30컬럼 문제 기하 (signed 범위)

# 목적 이름 / 최적화 방향 (+1 = 최대화, -1 = 최소화)
OBJECTIVE_NAMES: tuple[str, ...] = ("y11", "y12", "y13", "y21", "y22", "y23")
OBJECTIVE_SENSES: tuple[int, ...] = (+1, +1, -1, +1, +1, -1)

# 블록별 컬럼 인덱스 (표준 명세에서 유도 — 하드코딩 금지)
_COMMON_COLS = _SPACE.block_cols("common")
_SET1_COLS = _SPACE.block_cols("set1")
_SET2_COLS = _SPACE.block_cols("set2")

# 각 목적이 의존하는 컬럼 (common + 자기 set 블록)
_GROUP1_COLS = np.concatenate([_COMMON_COLS, _SET1_COLS])  # y11, y12, y13
_GROUP2_COLS = np.concatenate([_COMMON_COLS, _SET2_COLS])  # y21, y22, y23


# ──────────────────────────────────────────────────────────────────────────────
# 벤치마크 베이스 클래스
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkBase:
    """난이도별 벤치마크의 공통 골격.

    서브클래스는 `_latent(u)` 하나만 구현하면 된다:
      입력  u : (n, 30) — 레벨을 [0,1]로 매핑한 좌표
      출력    : (n, 6)  — 스케일 적용 전의 잠재(latent) 목적값.
                방향 무관하게 "값 자체"만 계산하고, 방향(max/min)은
                OBJECTIVE_SENSES 로만 해석한다.

    베이스 클래스가 담당하는 것:
      - 은닉 스케일/오프셋 적용 (j가 같은 목적끼리는 비슷한 스케일)
      - 가우시안 관측 노이즈 (latent 주효과 std의 5%)
      - 평가 횟수 카운트 (예산 관리는 runner 몫이지만 진단용으로 유지)
    """

    #: 서브클래스에서 지정 — 벤치마크 식별용 이름
    name: str = "base"
    #: latent 함수의 내부 파라미터를 생성할 고정 시드 (문제 자체의 재현성 보장)
    _structure_seed: int = 0

    # j(목적의 둘째 자리)별 은닉 스케일 규격.
    # "j가 같으면 스케일 비슷, j가 다르면 크게 다름" 요구를 구현한다.
    # (scale_center, offset_center) — 실제 값은 벤치마크마다 ±20% 지터.
    _SCALE_SPEC: dict[int, tuple[float, float]] = {
        1: (900.0, 5000.0),   # y11, y21 : 수천 단위
        2: (1.1, -3.0),       # y12, y22 : 1 안팎, 음수 오프셋
        3: (0.005, 0.02),     # y13, y23 : 0.0x 단위
    }

    # ── 래스터(타원 관측) 규격 ──
    #: 마스크 한 변 픽셀 수 — y11/y12/y21/y22 는 이 격자 위 타원의 측정치가 된다
    RASTER_GRID: int = 128
    #: 경계 밴드 폭 (|G − 0.5| < 이 값인 픽셀이 flip 후보)
    _FLIP_BAND: float = 0.08
    #: 밴드 내 픽셀의 flip 확률 (관측 노이즈의 원천)
    _FLIP_P: float = 0.35

    def __init__(self, noise_seed: int = 0, noise_level: float = 0.05) -> None:
        self.space = SearchSpace()
        self.noise_level = noise_level
        self._noise_rng = np.random.default_rng(noise_seed)
        self.n_evaluations = 0  # 진단용 누적 평가 횟수

        # 문제 구조(스케일, latent 파라미터)는 _structure_seed 로만 결정된다.
        # noise_seed 와 분리되어 있으므로, seed 를 바꿔도 문제 자체는 동일하다.
        rng = np.random.default_rng(self._structure_seed)
        self._init_scales(rng)
        self._init_latent_params(rng)

        # latent 주효과 통계를 몬테카를로로 추정해 둔다:
        #  - std : 스칼라 목적(y13/y23) 노이즈 크기 (noise_level × std)
        #  - mean/std : 타원 목적의 latent → 반축 픽셀 매핑 캘리브레이션
        probe = self.space.sample(np.random.default_rng(12345), n=4096)
        latent = self._latent(self.space.to_unit(probe))
        self._latent_std = latent.std(axis=0)  # (6,)
        self._latent_mu = latent.mean(axis=0)  # (6,)

    # ─── 문제 구조 초기화 ───────────────────────────────────────────────────

    def _init_scales(self, rng: np.random.Generator) -> None:
        """목적별 은닉 스케일/오프셋 생성. j가 같으면 유사, 다르면 상이."""
        scales, offsets = [], []
        for name in OBJECTIVE_NAMES:
            j = int(name[-1])  # y11 → 1, y23 → 3
            s_c, o_c = self._SCALE_SPEC[j]
            # 같은 j 안에서도 완전히 동일하지 않도록 ±20% 지터를 준다.
            scales.append(s_c * rng.uniform(0.8, 1.2))
            offsets.append(o_c * rng.uniform(0.8, 1.2))
        self._scales = np.asarray(scales)
        self._offsets = np.asarray(offsets)

    def _init_latent_params(self, rng: np.random.Generator) -> None:
        """서브클래스가 latent 함수 파라미터(가중치 등)를 생성하는 훅."""
        raise NotImplementedError

    def _latent(self, u: np.ndarray) -> np.ndarray:
        """(n, 30) → (n, 6) latent 목적값. 서브클래스에서 구현."""
        raise NotImplementedError

    # ─── 래스터 렌더링 (latent → boolean 타원 마스크) ──────────────────────

    def _semi_px(self, lat: np.ndarray, k: int) -> np.ndarray:
        """latent 값 → 타원 반축 픽셀 수 (단조 매핑, probe 통계로 캘리브레이션).

        latent 가 클수록 타원이 크다 — 측정치(max height/width) 최대화가
        latent 최대화와 동치가 되도록 하는 유일한 요구는 이 단조성이다.
        """
        z = (lat - self._latent_mu[k]) / (3.0 * self._latent_std[k] + 1e-12)
        half = self.RASTER_GRID // 2
        return np.clip(np.rint(28 + 26 * z), 4, half - 4).astype(np.int64)

    def _render_masks(self, semi_h: np.ndarray, semi_w: np.ndarray,
                      noisy: bool) -> np.ndarray:
        """가우시안 타원 마스크 렌더 (b, GRID, GRID).

        G(r,c) = exp(−½((r/σr)² + (c/σc)²)), σ = 반축/√(2 ln 2) 로 잡아
        G ≥ 0.5 등고선이 정확히 반축 길이의 타원이 된다. noisy=True 면
        경계 밴드(|G−0.5| < _FLIP_BAND)의 픽셀을 _FLIP_P 확률로 flip —
        이것이 타원 목적의 관측 노이즈다 (가산 가우시안이 아님).
        """
        g = self.RASTER_GRID
        coord = np.arange(g, dtype=np.float64) - (g - 1) / 2.0
        s = 1.0 / np.sqrt(2.0 * np.log(2.0))  # 반축 → σ 변환 계수
        masks = np.empty((len(semi_h), g, g), dtype=bool)
        for i in range(len(semi_h)):
            gr = np.exp(-0.5 * (coord / (semi_h[i] * s)) ** 2)  # (g,) 세로
            gc = np.exp(-0.5 * (coord / (semi_w[i] * s)) ** 2)  # (g,) 가로
            G = gr[:, None] * gc[None, :]
            m = G >= 0.5
            if noisy:
                band = np.abs(G - 0.5) < self._FLIP_BAND
                flips = band & (self._noise_rng.random((g, g)) < self._FLIP_P)
                m = m ^ flips
            masks[i] = m
        return masks

    # ─── 공개 API ──────────────────────────────────────────────────────────

    def evaluate(self, x: np.ndarray, noisy: bool = True) -> dict:
        """X 하나 또는 배치를 평가해 **구조화 raw 관측**을 반환한다.

        y_raw 는 6개 스칼라가 아니라 관측 장치의 원형이다:
          - mask1 (b, G, G) bool — 타원 1. y11 = max height, y12 = max width
          - mask2 (b, G, G) bool — 타원 2. y21 = max height, y22 = max width
          - y13, y23 (b,) float — 스칼라 목적 (은닉 스케일 적용, 기존과 동일)
        수치 목적으로의 변환(마스크 측정)은 optimizer 의 convert_y_raw 소관.

        노이즈: 타원 목적 = 경계 픽셀 random flip (+ 격자 양자화),
                스칼라 목적 = latent 주효과 std 의 noise_level 배 가우시안.

        Args:
            x     : (30,) 또는 (n, 30) 정수 값 벡터 (signed, [x_min, x_max])
            noisy : False면 관측 노이즈를 끈다 (true optimum 계산 전용)
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.int64))
        assert x.shape[1] == self.space.n_cols, f"X는 30컬럼이어야 함: {x.shape}"
        b = x.shape[0]

        latent = self._latent(self.space.to_unit(x))  # (b, 6)

        # 스칼라 목적 (y13, y23): 가우시안 노이즈 + 은닉 스케일/오프셋
        lat_s = latent[:, [2, 5]]
        if noisy:
            sigma = self.noise_level * self._latent_std[[2, 5]]
            lat_s = lat_s + self._noise_rng.normal(0.0, 1.0, (b, 2)) * sigma
        y13 = lat_s[:, 0] * self._scales[2] + self._offsets[2]
        y23 = lat_s[:, 1] * self._scales[5] + self._offsets[5]

        # 타원 목적 (y11/y12 → mask1, y21/y22 → mask2): latent → 반축 → 래스터
        mask1 = self._render_masks(self._semi_px(latent[:, 0], 0),
                                   self._semi_px(latent[:, 1], 1), noisy)
        mask2 = self._render_masks(self._semi_px(latent[:, 3], 3),
                                   self._semi_px(latent[:, 4], 4), noisy)

        self.n_evaluations += b
        return {"mask1": mask1, "mask2": mask2, "y13": y13, "y23": y23}

    # ─── 서브클래스 공용 헬퍼 ──────────────────────────────────────────────

    @staticmethod
    def _tradeoff_axis(u: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """common 블록에서 trade-off 축 c ∈ [0,1] 을 만든다.

        c 가 커지면 최대화 목적(y·1, y·2)에 유리하지만 최소화 목적(y·3)도
        같이 커지도록 latent 를 설계해, '전부 다 좋은 해'가 존재하지 않게 한다.
        """
        c_cols = u[:, _COMMON_COLS]  # (n, 10)
        return c_cols @ weights  # weights 는 합=1 로 정규화되어 c ∈ [0,1]


def _normalized_weights(rng: np.random.Generator, n: int) -> np.ndarray:
    """합이 1인 양수 가중치 벡터를 생성한다."""
    w = rng.uniform(0.5, 1.5, n)
    return w / w.sum()


# ──────────────────────────────────────────────────────────────────────────────
# 난이도 1 — Easy
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkEasy(BenchmarkBase):
    """[Easy] 거의 분리가능한 매끄러운 단봉 지형.

    - 각 목적은 자기 블록 컬럼들의 오목(concave) 단봉 항들의 가중합.
    - trade-off: common 의 c 축이 max 목적에는 +, min 목적에는 + (즉 악화)로
      작용하되 기울기가 완만해 타협점 찾기가 쉽다.
    - 좌표별로 독립이므로 coordinate/greedy 계열이 잘 통해야 정상.
    """

    name = "bm1_easy"
    _structure_seed = 101

    def _init_latent_params(self, rng: np.random.Generator) -> None:
        # 목적별: 의존 컬럼마다 단봉의 꼭짓점 위치(peak)와 가중치
        self._peaks = rng.uniform(0.0, 1.0, (6, self.space.n_cols))
        self._weights = rng.uniform(0.5, 1.5, (6, self.space.n_cols))
        # 목적별 의존 컬럼 마스크 (group1: common+set1 / group2: common+set2)
        mask = np.zeros((6, self.space.n_cols))
        mask[0:3, list(_GROUP1_COLS)] = 1.0
        mask[3:6, list(_GROUP2_COLS)] = 1.0
        self._weights *= mask
        self._weights /= self._weights.sum(axis=1, keepdims=True)
        # trade-off 축 가중치와 강도
        self._c_w = _normalized_weights(rng, len(_COMMON_COLS))
        self._c_gain = rng.uniform(0.25, 0.35, 6)  # 완만한 결합

    def _latent(self, u: np.ndarray) -> np.ndarray:
        # 단봉 항: 1 - (u - peak)^2 (peak 에서 최대인 오목 함수)
        unimodal = 1.0 - (u[:, None, :] - self._peaks[None, :, :]) ** 2  # (n,6,30)
        base = np.einsum("nkc,kc->nk", unimodal, self._weights)  # (n, 6)

        # trade-off: c 가 크면 모든 latent 가 커진다.
        # 최소화 목적(y13, y23)은 latent 가 커지는 것 = 악화이므로,
        # max 목적과 min 목적이 c 를 두고 충돌한다.
        c = self._tradeoff_axis(u, self._c_w)  # (n,)
        return base + c[:, None] * self._c_gain[None, :]


# ──────────────────────────────────────────────────────────────────────────────
# 난이도 2 — Medium
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkMedium(BenchmarkBase):
    """[Medium] 블록 내 상호작용 + 완만한 다봉 성분.

    - Easy 의 단봉 골격 위에
      (1) 블록 내 pairwise 곱 상호작용 (u_a × u_b)
      (2) 진폭이 중간 정도인 저주파 sin 다봉 성분
      을 얹는다. 지역최적이 여러 개 생기지만 basin 이 넓어
      population 계열(GA/PSO/ACO)이 헤쳐나갈 수 있는 수준.
    """

    name = "bm2_medium"
    _structure_seed = 202
    _N_PAIRS = 8  # 목적당 pairwise 상호작용 개수

    def _init_latent_params(self, rng: np.random.Generator) -> None:
        # ── Easy 와 같은 단봉 골격 ──
        self._peaks = rng.uniform(0.0, 1.0, (6, self.space.n_cols))
        self._weights = rng.uniform(0.5, 1.5, (6, self.space.n_cols))
        mask = np.zeros((6, self.space.n_cols))
        mask[0:3, list(_GROUP1_COLS)] = 1.0
        mask[3:6, list(_GROUP2_COLS)] = 1.0
        self._weights *= mask
        self._weights /= self._weights.sum(axis=1, keepdims=True)
        self._c_w = _normalized_weights(rng, len(_COMMON_COLS))
        self._c_gain = rng.uniform(0.3, 0.45, 6)

        # ── (1) 블록 내 pairwise 상호작용: 목적마다 의존 컬럼에서 쌍 추출 ──
        self._pair_idx = np.zeros((6, self._N_PAIRS, 2), dtype=np.int64)
        self._pair_sign = rng.choice([-1.0, 1.0], (6, self._N_PAIRS))
        for k in range(6):
            cols = np.asarray(_GROUP1_COLS if k < 3 else _GROUP2_COLS)
            for p in range(self._N_PAIRS):
                self._pair_idx[k, p] = rng.choice(cols, 2, replace=False)

        # ── (2) 다봉 성분: sin(freq·u + phase) 저주파 (진폭 0.15) ──
        self._sin_freq = rng.uniform(2.0, 4.0, (6, self.space.n_cols)) * np.pi
        self._sin_phase = rng.uniform(0.0, 2 * np.pi, (6, self.space.n_cols))
        self._sin_w = self._weights * 0.15 / np.maximum(
            self._weights.max(axis=1, keepdims=True), 1e-12
        )

    def _latent(self, u: np.ndarray) -> np.ndarray:
        unimodal = 1.0 - (u[:, None, :] - self._peaks[None, :, :]) ** 2
        base = np.einsum("nkc,kc->nk", unimodal, self._weights)

        # 블록 내 pairwise 상호작용 (진폭 0.05/쌍, 총 ±0.4 수준)
        ua = u[:, self._pair_idx[:, :, 0]]  # (n, 6, P)
        ub = u[:, self._pair_idx[:, :, 1]]
        inter = 0.05 * (self._pair_sign[None] * ua * ub).sum(axis=2)  # (n, 6)

        # 저주파 다봉 성분
        waves = np.sin(u[:, None, :] * self._sin_freq[None] + self._sin_phase[None])
        multi = np.einsum("nkc,kc->nk", waves, self._sin_w)

        c = self._tradeoff_axis(u, self._c_w)
        return base + inter + multi + c[:, None] * self._c_gain[None, :]


# ──────────────────────────────────────────────────────────────────────────────
# 난이도 3 — Hard
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkHard(BenchmarkBase):
    """[Hard] 교차-블록 상호작용 + 기만적(deceptive) 성분 + ruggedness.

    Medium 대비 추가되는 것:
      (1) common × set 교차 상호작용 — common 을 바꾸면 set 블록의 좋은 값이
          함께 바뀌므로 블록을 따로 최적화하면 함정에 빠진다.
      (2) deceptive 항 — 넓은 영역에서는 u 를 한쪽으로 밀수록 좋아 보이지만,
          진짜 최적은 반대쪽 좁은 basin 에 있다 (greedy/coordinate 계열 저격).
      (3) 고주파 rugged 성분 — 노이즈와 구분하기 어려운 잔물결.
    set2 가 15컬럼이라 유효차원 25(common+set2)가 병목이 된다.
    """

    name = "bm3_hard"
    _structure_seed = 303
    _N_PAIRS = 8        # 블록 내 상호작용 수 (Medium 과 동일)
    _N_CROSS = 10       # common × set 교차 상호작용 수

    def _init_latent_params(self, rng: np.random.Generator) -> None:
        # ── 단봉 골격 (비중을 낮춰 다른 성분의 영향력을 키운다) ──
        self._peaks = rng.uniform(0.0, 1.0, (6, self.space.n_cols))
        self._weights = rng.uniform(0.5, 1.5, (6, self.space.n_cols))
        mask = np.zeros((6, self.space.n_cols))
        mask[0:3, list(_GROUP1_COLS)] = 1.0
        mask[3:6, list(_GROUP2_COLS)] = 1.0
        self._weights *= mask
        self._weights /= self._weights.sum(axis=1, keepdims=True)
        self._c_w = _normalized_weights(rng, len(_COMMON_COLS))
        self._c_gain = rng.uniform(0.35, 0.5, 6)

        # ── 블록 내 pairwise 상호작용 ──
        self._pair_idx = np.zeros((6, self._N_PAIRS, 2), dtype=np.int64)
        self._pair_sign = rng.choice([-1.0, 1.0], (6, self._N_PAIRS))
        for k in range(6):
            cols = np.asarray(_GROUP1_COLS if k < 3 else _GROUP2_COLS)
            for p in range(self._N_PAIRS):
                self._pair_idx[k, p] = rng.choice(cols, 2, replace=False)

        # ── (1) 교차 상호작용: (common 컬럼, set 컬럼) 쌍의 cos 결합 ──
        # cos(π(u_c - u_s)) 형태 — common 값에 따라 set 쪽 최적 위치가 이동한다.
        self._cross_idx = np.zeros((6, self._N_CROSS, 2), dtype=np.int64)
        self._cross_w = rng.uniform(0.03, 0.07, (6, self._N_CROSS))
        for k in range(6):
            set_cols = _SET1_COLS if k < 3 else _SET2_COLS
            for p in range(self._N_CROSS):
                self._cross_idx[k, p, 0] = rng.choice(_COMMON_COLS)
                self._cross_idx[k, p, 1] = rng.choice(set_cols)

        # ── (2) deceptive 항: 목적마다 '기만 컬럼' 몇 개 선택 ──
        # g(u) = 0.35·u  (u<0.8 구간: 클수록 좋아 보이는 완만한 오르막)
        #      + 1.2·max(0, u-0.85)/0.15 (진짜 최적은 u≈1 근처 좁은 급경사)
        # → 낮은 해상도 탐색은 u≈0.8 언저리 가짜 정상에 안착하기 쉽다.
        self._dec_cols = np.zeros((6, 3), dtype=np.int64)
        for k in range(6):
            cols = np.asarray(_GROUP1_COLS if k < 3 else _GROUP2_COLS)
            self._dec_cols[k] = rng.choice(cols, 3, replace=False)
        self._dec_w = rng.uniform(0.10, 0.16, (6, 3))

        # ── (3) 고주파 rugged 성분 (진폭 작음: 노이즈 5%와 비슷한 크기) ──
        self._rug_freq = rng.uniform(15.0, 25.0, (6, self.space.n_cols))
        self._rug_phase = rng.uniform(0.0, 2 * np.pi, (6, self.space.n_cols))
        rug_w = rng.uniform(0.5, 1.5, (6, self.space.n_cols)) * mask
        self._rug_w = 0.02 * rug_w / rug_w.sum(axis=1, keepdims=True) * 30

    @staticmethod
    def _deceptive(u_cols: np.ndarray) -> np.ndarray:
        """기만 함수 g(u): 완만한 가짜 오르막 + 좁고 높은 진짜 정상."""
        gentle = 0.35 * u_cols
        spike = 1.2 * np.maximum(0.0, u_cols - 0.85) / 0.15
        return gentle + spike

    def _latent(self, u: np.ndarray) -> np.ndarray:
        unimodal = 1.0 - (u[:, None, :] - self._peaks[None, :, :]) ** 2
        base = 0.7 * np.einsum("nkc,kc->nk", unimodal, self._weights)

        # 블록 내 pairwise
        ua = u[:, self._pair_idx[:, :, 0]]
        ub = u[:, self._pair_idx[:, :, 1]]
        inter = 0.05 * (self._pair_sign[None] * ua * ub).sum(axis=2)

        # (1) common × set 교차 상호작용
        uc = u[:, self._cross_idx[:, :, 0]]  # (n, 6, C)
        us = u[:, self._cross_idx[:, :, 1]]
        cross = (self._cross_w[None] * np.cos(np.pi * (uc - us))).sum(axis=2)

        # (2) deceptive
        ud = u[:, self._dec_cols]  # (n, 6, 3)
        dec = (self._dec_w[None] * self._deceptive(ud)).sum(axis=2)

        # (3) rugged
        waves = np.sin(u[:, None, :] * self._rug_freq[None] + self._rug_phase[None])
        rug = np.einsum("nkc,kc->nk", waves, self._rug_w) / 30.0

        c = self._tradeoff_axis(u, self._c_w)
        return base + inter + cross + dec + rug + c[:, None] * self._c_gain[None, :]


# ──────────────────────────────────────────────────────────────────────────────
# 난이도 요인 분리 벤치마크 — BM4 / BM5
#
# 배경: "어떤 optimizer 가 이기는가"는 벤치마크의 지배적 난이도 요인에 따라
# 달라진다는 가설을 검증하기 위해, BM3 과 '전체 난이도는 비슷하되' 요인
# 배합이 다른 변형을 만든다.
#   BM3 = 교차-블록 결합 + deceptive + rugged 의 복합 (기존 유지)
#   BM4 = deceptive 지배   (상호작용 최소 — 사실상 분리가능하지만 기만적)
#   BM5 = epistasis 지배   (강한 상호작용 — 기만 없음, 매끄럽지만 비분리)
# 난이도 균등화: 단봉 골격 대비 '난이도 성분'의 분산 기여를 BM3 과 비슷한
# 수준으로 맞추고(가중치 배합), 최종적으로는 random search 대비 각 method 의
# 성능 격차로 실측 검증한다. trade-off/노이즈/스케일 규격은 BM3 과 동일.
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkDeceptive(BenchmarkBase):
    """[BM4] deceptive 지배형 — **진짜 trap 함수** 사용.

    주의: BM3 의 g(u) = 0.35u + spike 는 단조증가라 1-hop 언덕오르기로도
    오를 수 있는 '약한 기만'이다 (경계 선호 지형에 가깝다). BM4 는 고전적
    trap 모양을 쓴다:
        trap(u) = (v − u)/v            (u ≤ v: 가짜 정상 u=0 을 향한 오르막)
                = 1.4·(u − v)/(1 − v)  (u > v: 좁은 골짜기 너머 진짜 정상 u=1)
        (v = 0.85 — 국소 정보는 전부 u=0 쪽을 가리키고, 진짜 최적 u=1 은
         골짜기(u≈v)를 '점프'해야만 도달할 수 있다)

    - 목적당 6개 컬럼에 강한 trap (cardinality ≥ 4 컬럼만 선택 — card 2 는
      양 끝을 다 보므로 trap 이 성립하지 않는다).
    - 교차-블록 상호작용 없음, 블록 내 상호작용 약함, rugged 없음.
      trap 은 컬럼별 독립(가법적)이므로 '상호작용 모델링 능력'은 도움이 안
      되고, 골짜기를 건너뛰는 **탈출 능력**(점프 mutation/restart/전역 탐험)
      이 승부를 가른다.
    """

    name = "bm4_deceptive"
    _structure_seed = 404
    _N_PAIRS = 4       # 블록 내 상호작용 (약하게)
    _N_DEC = 6         # 목적당 trap 컬럼 수
    _TRAP_V = 0.85     # 골짜기 위치

    @classmethod
    def _trap(cls, u_cols: np.ndarray) -> np.ndarray:
        """고전적 trap: 가짜 정상 u=0 (높이 1.0), 진짜 정상 u=1 (높이 1.4)."""
        v = cls._TRAP_V
        fake = (v - u_cols) / v            # u ≤ v 구간: u=0 에서 최대 1.0
        true = 1.4 * (u_cols - v) / (1 - v)  # u > v 구간: u=1 에서 최대 1.4
        return np.where(u_cols <= v, fake, true)

    def _init_latent_params(self, rng: np.random.Generator) -> None:
        # 단봉 골격 (BM3 과 같은 0.7 계수 — 기본 지형의 비중 동일)
        self._peaks = rng.uniform(0.0, 1.0, (6, self.space.n_cols))
        self._weights = rng.uniform(0.5, 1.5, (6, self.space.n_cols))
        mask = np.zeros((6, self.space.n_cols))
        mask[0:3, list(_GROUP1_COLS)] = 1.0
        mask[3:6, list(_GROUP2_COLS)] = 1.0
        self._weights *= mask
        self._weights /= self._weights.sum(axis=1, keepdims=True)
        self._c_w = _normalized_weights(rng, len(_COMMON_COLS))
        self._c_gain = rng.uniform(0.35, 0.5, 6)  # trade-off 강도는 BM3 동일

        # 약한 블록 내 상호작용 (BM3 의 절반 규모)
        self._pair_idx = np.zeros((6, self._N_PAIRS, 2), dtype=np.int64)
        self._pair_sign = rng.choice([-1.0, 1.0], (6, self._N_PAIRS))
        for k in range(6):
            cols = np.asarray(_GROUP1_COLS if k < 3 else _GROUP2_COLS)
            for p in range(self._N_PAIRS):
                self._pair_idx[k, p] = rng.choice(cols, 2, replace=False)

        # 지배 성분: trap — cardinality ≥ 4 인 컬럼만 후보로 (card 2 는 trap 불성립)
        self._dec_cols = np.zeros((6, self._N_DEC), dtype=np.int64)
        for k in range(6):
            cols = np.asarray(_GROUP1_COLS if k < 3 else _GROUP2_COLS)
            cols = cols[_SPACE.cardinalities[cols] >= 4]
            self._dec_cols[k] = rng.choice(cols, self._N_DEC, replace=False)
        self._dec_w = rng.uniform(0.10, 0.16, (6, self._N_DEC))

    def _latent(self, u: np.ndarray) -> np.ndarray:
        unimodal = 1.0 - (u[:, None, :] - self._peaks[None, :, :]) ** 2
        base = 0.7 * np.einsum("nkc,kc->nk", unimodal, self._weights)

        ua = u[:, self._pair_idx[:, :, 0]]
        ub = u[:, self._pair_idx[:, :, 1]]
        inter = 0.05 * (self._pair_sign[None] * ua * ub).sum(axis=2)

        ud = u[:, self._dec_cols]
        dec = (self._dec_w[None] * self._trap(ud)).sum(axis=2)

        c = self._tradeoff_axis(u, self._c_w)
        return base + inter + dec + c[:, None] * self._c_gain[None, :]


class BenchmarkEpistasis(BenchmarkBase):
    """[BM5] epistasis(상호작용) 지배형.

    - 교차-블록 결합을 목적당 20쌍 × 강한 가중치로 (BM3: 10쌍 × 약한 가중치),
      블록 내 pairwise 도 12쌍으로 강화. deceptive/rugged 없음.
    - 지형은 매끄럽지만 심하게 비분리(non-separable) — 어떤 컬럼의 최적값도
      다른 컬럼들에 조건부다.
      → 좌표/marginal 계열이 무너지고, 상호작용 학습(xgb) 또는 블록 단위
        재조합(ga)이 유리할 것으로 예상되는 지형.
    """

    name = "bm5_epistasis"
    _structure_seed = 505
    _N_PAIRS = 12      # 블록 내 상호작용 (BM3 의 1.5배)
    _N_CROSS = 20      # 교차-블록 상호작용 (BM3 의 2배)

    def _init_latent_params(self, rng: np.random.Generator) -> None:
        # 단봉 골격의 비중을 낮춰 상호작용이 지형을 지배하게 한다
        self._peaks = rng.uniform(0.0, 1.0, (6, self.space.n_cols))
        self._weights = rng.uniform(0.5, 1.5, (6, self.space.n_cols))
        mask = np.zeros((6, self.space.n_cols))
        mask[0:3, list(_GROUP1_COLS)] = 1.0
        mask[3:6, list(_GROUP2_COLS)] = 1.0
        self._weights *= mask
        self._weights /= self._weights.sum(axis=1, keepdims=True)
        self._c_w = _normalized_weights(rng, len(_COMMON_COLS))
        self._c_gain = rng.uniform(0.35, 0.5, 6)  # trade-off 강도는 BM3 동일

        # 블록 내 pairwise (강화: 진폭 0.08)
        self._pair_idx = np.zeros((6, self._N_PAIRS, 2), dtype=np.int64)
        self._pair_sign = rng.choice([-1.0, 1.0], (6, self._N_PAIRS))
        for k in range(6):
            cols = np.asarray(_GROUP1_COLS if k < 3 else _GROUP2_COLS)
            for p in range(self._N_PAIRS):
                self._pair_idx[k, p] = rng.choice(cols, 2, replace=False)

        # 지배 성분: 교차-블록 결합 (쌍 수 2배, 가중치 ≈ 2배)
        self._cross_idx = np.zeros((6, self._N_CROSS, 2), dtype=np.int64)
        self._cross_w = rng.uniform(0.06, 0.12, (6, self._N_CROSS))
        for k in range(6):
            set_cols = _SET1_COLS if k < 3 else _SET2_COLS
            for p in range(self._N_CROSS):
                self._cross_idx[k, p, 0] = rng.choice(_COMMON_COLS)
                self._cross_idx[k, p, 1] = rng.choice(set_cols)

    def _latent(self, u: np.ndarray) -> np.ndarray:
        unimodal = 1.0 - (u[:, None, :] - self._peaks[None, :, :]) ** 2
        base = 0.5 * np.einsum("nkc,kc->nk", unimodal, self._weights)

        ua = u[:, self._pair_idx[:, :, 0]]
        ub = u[:, self._pair_idx[:, :, 1]]
        inter = 0.08 * (self._pair_sign[None] * ua * ub).sum(axis=2)

        uc = u[:, self._cross_idx[:, :, 0]]
        us = u[:, self._cross_idx[:, :, 1]]
        cross = (self._cross_w[None] * np.cos(np.pi * (uc - us))).sum(axis=2)

        c = self._tradeoff_axis(u, self._c_w)
        return base + inter + cross + c[:, None] * self._c_gain[None, :]


# ──────────────────────────────────────────────────────────────────────────────
# 벤치마크 레지스트리 — runner 가 이름으로 조회한다
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARKS: dict[str, type[BenchmarkBase]] = {
    BenchmarkEasy.name: BenchmarkEasy,
    BenchmarkMedium.name: BenchmarkMedium,
    BenchmarkHard.name: BenchmarkHard,
    BenchmarkDeceptive.name: BenchmarkDeceptive,
    BenchmarkEpistasis.name: BenchmarkEpistasis,
}

#: 토너먼트 진행 순서 (쉬움 → 어려움)
TOURNAMENT_ORDER: tuple[str, ...] = (
    BenchmarkEasy.name,
    BenchmarkMedium.name,
    BenchmarkHard.name,
)


def serve_eval(benchmark_name: str, exchange_dir, seed: int) -> int:
    """프로세스 분리 실행의 calculator 한 스텝: x.txt 읽기 → 평가 → y_raw.bin.

    노이즈는 (seed, eval_index) 로 재시딩해 프로세스 경계와 무관하게 결정적이다
    (배치 단위 — 같은 eval_index 는 같은 노이즈). 교환 셸 함수(read_x/write_y_raw)는
    optimizer 소유라 여기서 지연 import 한다 (모듈 순환 회피).
    """
    from pathlib import Path

    from optimizer import read_x, write_y_raw

    d = Path(exchange_dir)
    space = SearchSpace()
    X, eval_index = read_x(d / "x.txt", space=space)
    calc = BENCHMARKS[benchmark_name](noise_seed=seed)
    calc._noise_rng = np.random.default_rng([seed, eval_index])  # 배치 단위 결정적 노이즈
    raw = calc.evaluate(X)  # noisy=True — 구조화 관측 원형
    write_y_raw(d / "y_raw.bin", raw, eval_index=eval_index)
    return len(X)


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="calculator — 자가 점검 또는 프로세스 분리 평가")
    _ap.add_argument("--serve-eval", action="store_true",
                     help="파일 기반 프로세스 분리 실행의 calculator 한 스텝")
    _ap.add_argument("--benchmark", choices=list(BENCHMARKS), default="bm1_easy")
    _ap.add_argument("--dir", type=str, default=None, help="교환 디렉토리")
    _ap.add_argument("--seed", type=int, default=0)
    _args = _ap.parse_args()

    if _args.serve_eval:  # runner 가 서브프로세스로 호출하는 경로
        assert _args.dir, "--serve-eval 에는 --dir 필요"
        b = serve_eval(_args.benchmark, _args.dir, _args.seed)
        print(f"[calc-eval] {_args.benchmark} → {b} evals → y_raw.bin")
        raise SystemExit(0)

    # 인자 없음 → 자가 점검: 공간 크기, 구조화 y_raw 형상, 측정치 스케일, 노이즈 수준.
    # (측정 변환은 optimizer 소유 — 여기서는 표시용으로만 빌려 쓴다.
    #  __main__ 가드 안 import 라 모듈 순환 없음)
    from optimizer import convert_y_raw

    space = SearchSpace()
    print(f"search space log10 size = {space.log10_size:.2f} (목표 ≈ 15)")
    rng = np.random.default_rng(0)
    for name, cls in BENCHMARKS.items():
        calc = cls(noise_seed=0)
        xs = space.sample(rng, 5)
        raw = calc.evaluate(xs)
        g = calc.RASTER_GRID
        assert raw["mask1"].shape == raw["mask2"].shape == (5, g, g)
        assert raw["mask1"].dtype == bool and raw["y13"].shape == (5,)
        y = convert_y_raw(raw)
        y_clean = convert_y_raw(calc.evaluate(xs, noisy=False))
        print(f"\n[{name}] y_raw = mask1/mask2 (5, {g}, {g}) bool + y13/y23 (5,)")
        print(f"  측정 y (행=샘플):")
        header = "  " + "  ".join(f"{n:>12s}" for n in OBJECTIVE_NAMES)
        print(header)
        for row in y:
            print("  " + "  ".join(f"{v:12.4g}" for v in row))
        # 노이즈 수준: 같은 xs 를 반복 관측해 측정치 표준편차 / 신호 범위
        reps = np.stack([convert_y_raw(calc.evaluate(xs)) for _ in range(20)])
        noise_std = reps.std(axis=0).mean(axis=0)          # (6,)
        signal_span = y_clean.max(axis=0) - y_clean.min(axis=0) + 1e-12
        print("  noise std / signal span ≈", np.round(noise_std / signal_span, 4))
