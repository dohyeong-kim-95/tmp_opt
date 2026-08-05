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
    * 관측 노이즈: 타원 = 격자 양자화(반축 픽셀 반올림)뿐,
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

    def _render_masks(self, semi_h: np.ndarray, semi_w: np.ndarray) -> np.ndarray:
        """가우시안 타원 마스크 렌더 (b, GRID, GRID).

        G(r,c) = exp(−½((r/σr)² + (c/σc)²)), σ = 반축/√(2 ln 2) 로 잡아
        G ≥ 0.5 등고선이 정확히 반축 길이의 타원이 된다.

        타원 목적의 관측 노이즈는 **격자 양자화뿐**이다 (_semi_px 의 반올림).
        이전 판에는 경계 밴드 픽셀을 확률적으로 뒤집는 flip 노이즈가 있었으나,
        가장자리에 붙는 이상치 픽셀이 blob 형상 자체를 왜곡해 (측정치는 개수
        기반이라 견뎌도 형상 보간은 못 견딘다) 제거했다.
        """
        g = self.RASTER_GRID
        coord = np.arange(g, dtype=np.float64) - (g - 1) / 2.0
        s = 1.0 / np.sqrt(2.0 * np.log(2.0))  # 반축 → σ 변환 계수
        masks = np.empty((len(semi_h), g, g), dtype=bool)
        for i in range(len(semi_h)):
            gr = np.exp(-0.5 * (coord / (semi_h[i] * s)) ** 2)  # (g,) 세로
            gc = np.exp(-0.5 * (coord / (semi_w[i] * s)) ** 2)  # (g,) 가로
            masks[i] = (gr[:, None] * gc[None, :]) >= 0.5
        return masks

    # ─── 공개 API ──────────────────────────────────────────────────────────

    def evaluate(self, x: np.ndarray, noisy: bool = True) -> dict:
        """X 하나 또는 배치를 평가해 **구조화 raw 관측**을 반환한다.

        y_raw 는 6개 스칼라가 아니라 관측 장치의 원형이다:
          - mask1 (b, G, G) bool — 타원 1. y11 = max height, y12 = max width
          - mask2 (b, G, G) bool — 타원 2. y21 = max height, y22 = max width
          - y13, y23 (b,) float — 스칼라 목적 (은닉 스케일 적용, 기존과 동일)
        수치 목적으로의 변환(마스크 측정)은 optimizer 의 convert_y_raw 소관.

        노이즈: 타원 목적 = 격자 양자화뿐 (noisy 와 무관하게 결정적),
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
        # (noisy 는 스칼라 목적에만 작용한다 — 타원 쪽 노이즈는 격자 양자화뿐)
        mask1 = self._render_masks(self._semi_px(latent[:, 0], 0),
                                   self._semi_px(latent[:, 1], 1))
        mask2 = self._render_masks(self._semi_px(latent[:, 3], 3),
                                   self._semi_px(latent[:, 4], 4))

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
# 데이터 기반 반응표면 — 실측 관측을 재생하는 모킹(mock) calculator
#
# 용도: 실제 TEST 가 비싸서 알고리즘을 실측으로 돌릴 수 없을 때, 이미 확보한
# (X, y) 관측만으로 calculator 와 **동일한 계약**의 대역을 세운다. 목표는 예측
# 정확도가 아니라 두 가지 검증이다:
#   (1) 알고리즘이 프로그램적으로 완주하는가 (ask→평가→tell 전 경로 smoke test)
#   (2) 알고리즘이 터무니없는 X 를 추천하지 않는가 (관측에서 벗어나는 정도 계측)
#
# 설계 원칙 — 평활 회귀가 아니라 **재생기(replayer) + 국소 보간기**다:
#   · 관측점에서는 실측값을 그대로 낸다 (평활화 금지)
#   · 관측에서 멀면 값을 지어내지 않고 "데이터 없음" 으로 표시한다
#   · 어떤 관측이 근거였는지 항상 되짚을 수 있다 (근거 관측 + 해밍거리 보고)
# 릿지/GBM/nugget 있는 GP 같은 예측 모델은 첫 번째 성질을 원리적으로 못 지킨다
# (관측점을 평활해 버린다). 그래서 모킹에는 그 계열을 쓰지 않는다.
#
# 거리: **블록 제한 정규화 해밍**. y11/y12/y13 은 common+set1(15컬럼)에만,
# y21/y22/y23 은 common+set2(25컬럼)에만 의존하므로 (_GROUP*_COLS), 무관한
# 컬럼의 차이를 거리에 넣으면 근거 관측을 놓친다. 그 결과 커버리지 판정도
# 목적 그룹별로 따로 난다 — y1x 는 데이터가 있고 y2x 는 없는 X 가 존재한다.
# ──────────────────────────────────────────────────────────────────────────────


class NoDataError(RuntimeError):
    """관측 커버리지 밖의 X 가 요청됨 (policy="strict" 일 때 raise)."""


#: 목적 인덱스 → 의존 컬럼 그룹 (0,1,2 = y11,y12,y13 / 3,4,5 = y21,y22,y23)
_OBJ_GROUP: tuple[int, ...] = (1, 1, 1, 2, 2, 2)
_GROUP_COLS: dict[int, np.ndarray] = {1: _GROUP1_COLS, 2: _GROUP2_COLS}


class SurfaceCalculator:
    """실측 (X, y) 관측으로 만든 반응표면 — calculator 계약의 모킹 구현.

    공개 계약은 BenchmarkBase 와 동일하다: `evaluate(x, noisy=...)` 가
    {"mask1", "mask2", "y13", "y23"} 를 돌려주므로 runner / optimizer /
    convert_y_raw 가 아무것도 모른 채 그대로 돈다.

    한 X 에 대한 판정 사다리 (목적 그룹별로 독립 수행):
      1. `exact`       — 30컬럼 전부 일치하는 관측이 있다 → 그 실측값 그대로
      2. `exact_block` — 해당 그룹의 의존 컬럼이 전부 일치 → 그 관측들을 합성
                         (그룹 밖 컬럼은 이 목적에 영향이 없으므로 정당하다)
      3. `interp`      — 최근접 거리 ≤ d_gate 이고 이웃 불일치 ≤ spread_gate
                         → k최근접 역거리가중(Shepard) 보간
      4. `no_data`     — 그 외. 값을 지어내지 않는다 (아래 policy 참조)

    근거 관측이 **여러 개**일 때 (같은 X 를 반복 측정했는데 노이즈로 y 가
    조금씩 다른 경우, 또는 이웃 여러 개를 섞는 경우) 자료형마다 합성이 다르다:
      · 스칼라 y13/y23 → 가중평균
      · boolean 마스크 → **blob 형상 보간**. 마스크는 픽셀 집합이 아니라 덩어리
        (blob) 이므로 픽셀 단위 평균/다수결은 가장자리를 갈라놓는다. 무게중심과
        각도별 반경 r(θ) 로 분해해 각각 가중평균한 뒤 다시 래스터화한다
        (`_blend_blobs`) — 반복측정이면 경계 노이즈가 지워지고, 서로 다른
        이웃이면 형상이 부드럽게 모핑된다.
    근거가 정확히 하나면 실측 마스크를 **바이트 그대로** 돌려준다 (요구 3).

    "데이터 없음" 처리 (policy):
      - "flag"        (기본) 관측 y 의 **중앙값**으로 채우고 no_data 로 기록.
                      외삽값을 내면 관측에서 먼 곳이 더 좋아 보이는 가짜 최적이
                      생겨 검증 (2) 가 무의미해진다. 중앙값은 "아는 게 없다" 는
                      정직한 사전값이다. 조용한 대체가 아니다 — 매 평가가
                      `coverage_log` 에 남고 `report()` 에 비율로 드러난다.
      - "pessimistic" 목적별 **최악 관측치**(sense 반영)로 채운다. 커버리지
                      밖이 실제로 나쁘게 보이므로 알고리즘이 관측 영역으로
                      끌려온다. 관측이 성겨 "flag" 로는 지형이 평평해져 검증
                      (2) 를 못 할 때 쓴다 — 알고리즘이 근거 있는 영역으로
                      수렴하는지를 실제로 볼 수 있다.
      - "strict"      NoDataError 를 raise. 커버리지를 벗어나는 순간 실행을
                      멈추고 싶을 때 (검증 (2) 를 하드 게이트로 쓰는 경우).

    Args:
        X_obs      : (n, 30) 정수 — 실측한 X (space 범위 안이어야 함)
        Y_obs      : (n, 6)  float — 대응 실측 y, 열 순서 = OBJECTIVE_NAMES
        masks_obs  : (선택) {"mask1": (n,G,G) bool, "mask2": ...} — 실측 마스크
                     원형이 있으면 exact 재생 시 이걸 바이트 그대로 돌려준다.
                     없으면 y 의 픽셀 측정치로부터 마스크를 합성한다.
        k          : 보간에 쓸 최근접 이웃 수
        power      : 역거리가중 지수 (클수록 최근접에 쏠림)
        d_gate     : 보간 허용 최대 정규화 해밍거리. None 이면 _DEFAULT_D_GATE.
                     데이터에 맞춘 값은 `suggest_d_gate()` 로 뽑아 명시 전달.
        spread_gate: 이웃 y 불일치 허용치 — 목적별 robust 스케일(IQR) 대비 비율
        noise_level: >0 이면 관측 재생에도 노이즈를 얹는다. 기본 0 (정확 재생).
    """

    name = "surface"
    RASTER_GRID: int = BenchmarkBase.RASTER_GRID
    #: 기본 보간 반경 — 의존 컬럼의 10% 까지만 달라도 보간을 허용한다
    #: (30컬럼 기준 3개, set1 기준 1.5개, set2 기준 2.5개). 해밍거리는 값의
    #: 크기 차이를 무시하므로 (card 30 컬럼이 1칸 달라도, 반대편 끝이어도
    #: 똑같이 1) 반경을 넓게 잡으면 근거 없는 값을 자신 있게 내놓게 된다.
    _DEFAULT_D_GATE: float = 0.10

    def __init__(
        self,
        X_obs: np.ndarray,
        Y_obs: np.ndarray,
        masks_obs: dict | None = None,
        *,
        k: int = 4,
        power: float = 2.0,
        d_gate: float | None = None,
        spread_gate: float = 0.5,
        policy: str = "flag",
        noise_level: float = 0.0,
        noise_seed: int = 0,
    ) -> None:
        if policy not in ("flag", "pessimistic", "strict"):
            raise ValueError(
                f"policy 는 'flag' | 'pessimistic' | 'strict' — {policy!r}")
        self.space = SearchSpace()
        X = np.atleast_2d(np.asarray(X_obs, dtype=np.int64))
        Y = np.atleast_2d(np.asarray(Y_obs, dtype=np.float64))
        if X.shape[1] != self.space.n_cols:
            raise ValueError(f"X_obs 는 {self.space.n_cols}컬럼이어야 함: {X.shape}")
        if Y.shape != (len(X), len(OBJECTIVE_NAMES)):
            raise ValueError(
                f"Y_obs 형상 불일치 — {Y.shape} ≠ ({len(X)}, {len(OBJECTIVE_NAMES)})")
        if len(X) == 0:
            raise ValueError("관측이 0개 — 반응표면을 만들 수 없다")
        if not np.isfinite(Y).all():
            raise ValueError("Y_obs 에 비유한값 — 실측 결측은 호출자가 먼저 처리할 것")
        if (X < self.space.x_min).any() or (X > self.space.x_max).any():
            raise ValueError("X_obs 가 탐색 공간 범위를 벗어남 — space.py 명세 확인")

        self._X, self._Y = X, Y
        self.k = int(k)
        self.power = float(power)
        self.spread_gate = float(spread_gate)
        self.policy = policy
        self.noise_level = float(noise_level)
        self._noise_rng = np.random.default_rng(noise_seed)
        self.n_evaluations = 0
        #: 평가 이력 — 검증 (2) 의 실제 산출물 (거리/상태 누적)
        self.coverage_log: list[dict] = []

        # 목적별 robust 스케일 (IQR) — 이웃 불일치를 무차원화하는 기준
        q75, q25 = np.percentile(Y, [75, 25], axis=0)
        self._scale = np.maximum(q75 - q25, 1e-12)
        # no_data 채움값 — policy 별로 하나씩. pessimistic 은 sense 를 반영해
        # "최대화 목적은 최소 관측치, 최소화 목적은 최대 관측치" 를 쓴다.
        self._median = np.median(Y, axis=0)
        senses = np.asarray(OBJECTIVE_SENSES)
        self._worst = np.where(senses > 0, Y.min(axis=0), Y.max(axis=0))

        # 실측 마스크 원형 (있으면 exact 재생에 사용)
        self._masks: dict | None = None
        if masks_obs is not None:
            m1 = np.asarray(masks_obs["mask1"], dtype=bool)
            m2 = np.asarray(masks_obs["mask2"], dtype=bool)
            g = self.RASTER_GRID
            if m1.shape != (len(X), g, g) or m1.shape != m2.shape:
                raise ValueError(
                    f"masks_obs 형상 불일치 — {m1.shape}, 기대 ({len(X)}, {g}, {g})")
            self._masks = {"mask1": m1, "mask2": m2}

        self.d_gate = self._DEFAULT_D_GATE if d_gate is None else float(d_gate)

    # ─── 생성/저장 ─────────────────────────────────────────────────────────

    @classmethod
    def from_npz(cls, path, **kwargs) -> "SurfaceCalculator":
        """npz 에서 로드. 키: X, Y (+ 선택 mask1, mask2 — uint8 (n,G,G))."""
        from pathlib import Path

        z = np.load(Path(path))
        masks = None
        if "mask1" in z and "mask2" in z:
            masks = {"mask1": z["mask1"].astype(bool), "mask2": z["mask2"].astype(bool)}
        return cls(z["X"], z["Y"], masks, **kwargs)

    def save_npz(self, path) -> None:
        """관측을 npz 로 저장 (마스크 원형이 있으면 uint8 로 함께)."""
        from pathlib import Path

        payload = {"X": self._X, "Y": self._Y}
        if self._masks is not None:
            payload["mask1"] = self._masks["mask1"].astype(np.uint8)
            payload["mask2"] = self._masks["mask2"].astype(np.uint8)
        np.savez_compressed(Path(path), **payload)

    # ─── 거리 ──────────────────────────────────────────────────────────────

    def _dist(self, x: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """(b, n) 블록 제한 정규화 해밍거리 — 다른 컬럼 수 = 다른 컬럼 비율."""
        return (x[:, None, cols] != self._X[None, :, cols]).mean(axis=2)

    def suggest_d_gate(self, q: float = 25.0) -> float:
        """관측끼리의 leave-one-out 최근접거리 q퍼센타일 — 데이터 맞춤 게이트.

        "관측이 실제로 이웃을 갖는 간격" 이다. **낮은** 퍼센타일을 쓰는 게
        핵심 — 높은 퍼센타일은 데이터가 성길수록 게이트를 넓혀서, 근거가
        없을수록 자신 있게 값을 내놓는 정반대 동작이 된다. 관측이 1개뿐이면
        게이트를 열 근거가 없어 0 (exact 만 허용).

        반환값을 그대로 쓰지 말고 `_DEFAULT_D_GATE` 와 비교해 더 작은 쪽을
        택하는 식으로 쓰는 것을 권한다.
        """
        if len(self._X) < 2:
            return 0.0
        d = self._dist(self._X, np.arange(self.space.n_cols, dtype=np.int64))
        np.fill_diagonal(d, np.inf)
        return float(np.percentile(d.min(axis=1), q))

    def observed_space(self) -> SearchSpace:
        """관측이 실제로 덮은 범위로 좁힌 SearchSpace (컬럼별 [min, max]).

        실측이 컬럼당 소수의 수준만 시도한 경우(실험 데이터에서 흔하다) 공간이
        크게 줄어 optimizer 의 제안이 데이터 근방에 떨어진다.

        한계 — 이건 **범위** 만 좁힌다. 관측이 각 컬럼의 양 끝을 이미 찍고
        있으면 (예: 무작위 샘플링으로 모은 데이터) 전혀 줄지 않는다. 커버리지가
        비는 진짜 이유는 범위가 아니라 조합 희소성(10^15 에 관측 수백)이라,
        이 함수로 no_data 를 없앨 수 있다고 기대하면 안 된다.

        주의: 좁힌 공간은 원 문제와 다른 문제다. 알고리즘의 **동작 점검**
        용도로만 쓰고, 성능 비교의 근거로 삼지 말 것.
        """
        return SearchSpace(x_min=self._X.min(axis=0), x_max=self._X.max(axis=0),
                           blocks=self.space.blocks)

    def neighbors(self, x: np.ndarray, k: int = 3, block: str = "all") -> list[dict]:
        """**요구 1** — x 에 가장 가까운 관측들과 그때의 y 를 돌려준다.

        Args:
            x     : (30,) 하나의 X
            k     : 반환할 이웃 수 (가까운 순)
            block : "all"(30컬럼) | "set1"(common+set1) | "set2"(common+set2).
                    목적 그룹별로 근거 관측이 다르므로 블록별 조회를 지원한다.
        Returns:
            [{rank, index, distance, n_diff, cols_diff, X, y}, ...]
            distance = 정규화 해밍(=cols_diff/len(cols)), y = 실측 6목적 dict
        """
        cols = {"all": np.arange(self.space.n_cols, dtype=np.int64),
                "set1": _GROUP1_COLS, "set2": _GROUP2_COLS}[block]
        x = np.asarray(x, dtype=np.int64).reshape(1, -1)
        d = self._dist(x, cols)[0]
        order = np.argsort(d, kind="stable")[:k]
        out = []
        for rank, i in enumerate(order):
            diff = np.flatnonzero(x[0] != self._X[i])
            out.append({
                "rank": rank,
                "index": int(i),
                "distance": float(d[i]),
                "n_diff": int(len(np.intersect1d(diff, cols))),
                "cols_diff": diff.tolist(),
                "X": self._X[i].tolist(),
                "y": {n: float(v) for n, v in zip(OBJECTIVE_NAMES, self._Y[i])},
            })
        return out

    def explain(self, x: np.ndarray, k: int = 3) -> str:
        """neighbors + 커버리지 판정을 사람이 읽는 형태로 요약한다."""
        _, status, info = self.predict(x)
        lines = [f"X = {np.asarray(x, dtype=np.int64).ravel().tolist()}"]
        for gi, g in enumerate((1, 2)):
            names = OBJECTIVE_NAMES[gi * 3:(gi + 1) * 3]
            lines.append(
                f"  group{g} ({'/'.join(names)}): status={status[0, gi]} "
                f"d_min={info['d_min'][0, gi]:.4f} (게이트 {self.d_gate:.4f}) "
                f"spread={info['spread'][0, gi]:.3f} (게이트 {self.spread_gate:.3f})")
        for nb in self.neighbors(x, k=k):
            ys = "  ".join(f"{n}={v:.4g}" for n, v in nb["y"].items())
            lines.append(f"  #{nb['rank']} d={nb['distance']:.4f} "
                         f"({nb['n_diff']}컬럼 상이, idx={nb['index']})  {ys}")
        return "\n".join(lines)

    # ─── 예측 ──────────────────────────────────────────────────────────────

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """X → (Y (b,6), status (b,2), info). **요구 2·3 의 본체**.

        status[:, 0] 은 y11/y12/y13, status[:, 1] 은 y21/y22/y23 의 판정
        ("exact" | "exact_block" | "interp" | "no_data").
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.int64))
        b = len(x)
        Y = np.empty((b, len(OBJECTIVE_NAMES)), dtype=np.float64)
        status = np.empty((b, 2), dtype=object)
        d_min = np.zeros((b, 2))
        spread = np.zeros((b, 2))
        src = [[None, None] for _ in range(b)]  # 근거 관측 인덱스
        wts = [[None, None] for _ in range(b)]  # 그 근거들의 가중치 (blob 보간용)

        all_cols = np.arange(self.space.n_cols, dtype=np.int64)
        d_full = self._dist(x, all_cols)  # (b, n) — exact 판정용

        for gi, g in enumerate((1, 2)):
            cols = _GROUP_COLS[g]
            objs = np.arange(gi * 3, gi * 3 + 3)
            d = self._dist(x, cols)  # (b, n)
            kk = min(self.k, len(self._X))
            order = np.argsort(d, axis=1, kind="stable")[:, :kk]
            dd = np.take_along_axis(d, order, axis=1)  # (b, kk)

            # 역거리가중 (Shepard). d=0 은 아래에서 exact 로 덮어쓰므로
            # 여기서는 0 분모만 피하면 된다.
            w = 1.0 / np.maximum(dd, 1e-9) ** self.power
            w /= w.sum(axis=1, keepdims=True)
            neigh = self._Y[order][:, :, objs]                 # (b, kk, 3)
            yhat = (w[:, :, None] * neigh).sum(axis=1)         # (b, 3)
            var = (w[:, :, None] * (neigh - yhat[:, None, :]) ** 2).sum(axis=1)
            rel_spread = (np.sqrt(var) / self._scale[objs]).max(axis=1)  # (b,)

            d_min[:, gi] = dd[:, 0]
            spread[:, gi] = rel_spread
            ok = (dd[:, 0] <= self.d_gate) & (rel_spread <= self.spread_gate)

            for i in range(b):
                exact = np.flatnonzero(d_full[i] == 0.0)
                if len(exact):                       # 1. 30컬럼 완전 일치
                    Y[i, objs] = self._Y[np.ix_(exact, objs)].mean(axis=0)
                    status[i, gi] = "exact"
                    src[i][gi] = exact.tolist()
                    wts[i][gi] = np.full(len(exact), 1.0 / len(exact))
                    d_min[i, gi] = 0.0
                    spread[i, gi] = 0.0
                    continue
                blk = np.flatnonzero(d[i] == 0.0)
                if len(blk):                         # 2. 의존 블록 일치
                    Y[i, objs] = self._Y[np.ix_(blk, objs)].mean(axis=0)
                    status[i, gi] = "exact_block"
                    src[i][gi] = blk.tolist()
                    wts[i][gi] = np.full(len(blk), 1.0 / len(blk))
                    spread[i, gi] = 0.0
                    continue
                if ok[i]:                            # 3. 국소 보간
                    Y[i, objs] = yhat[i]
                    status[i, gi] = "interp"
                    src[i][gi] = order[i].tolist()
                    wts[i][gi] = w[i]
                    continue
                # 4. 커버리지 밖 — 값을 지어내지 않는다
                if self.policy == "strict":
                    raise NoDataError(
                        f"데이터 없음: group{g} d_min={dd[i, 0]:.4f} "
                        f"(게이트 {self.d_gate:.4f}), spread={rel_spread[i]:.3f} "
                        f"(게이트 {self.spread_gate:.3f}), X={x[i].tolist()}")
                fill = self._worst if self.policy == "pessimistic" else self._median
                Y[i, objs] = fill[objs]
                status[i, gi] = "no_data"
                src[i][gi] = []
                wts[i][gi] = np.zeros(0)

        return Y, status, {"d_min": d_min, "spread": spread, "sources": src,
                           "weights": wts, "d_full_min": d_full.min(axis=1)}

    # ─── blob 보간 (마스크 원형끼리의 형상 보간) ────────────────────────────

    #: blob 반경 프로파일에서 "덩어리가 끊겼다" 로 볼 반경 간격(픽셀).
    #: 이보다 멀리 떨어져 있는 바깥 픽셀은 본체가 아니라 이상치로 본다.
    _BLOB_GAP: float = 2.0

    @classmethod
    def _blob_profile(cls, mask: np.ndarray,
                      n_theta: int) -> tuple[np.ndarray, np.ndarray]:
        """boolean blob → (무게중심, 각도별 반경 프로파일 r(θ)).

        마스크를 픽셀 집합이 아니라 **형상**으로 다룬다. 무게중심에서 본
        각도 bin 별 경계 반경을 재면 크기·위치·찌그러짐이 (c, r(θ)) 로 분리돼
        서로 다른 blob 사이를 자연스럽게 섞을 수 있다.

        경계 반경은 각 bin 의 **최대** 반경이 아니라 **본체가 연결된 데까지**
        의 반경이다: bin 안의 반경들을 정렬해 _BLOB_GAP 보다 큰 틈이 처음
        나타나는 지점에서 끊는다. 실측 마스크에서 가장자리에 붙는 낱개
        이상치 픽셀이 프로파일을 통째로 부풀리는 것을 막으면서, 깨끗한
        blob 에는 아무 영향도 주지 않는다 (틈이 없으므로 전부 본체).
        비어 있는 마스크는 (중심, 전부 0) 으로 돌려준다.
        """
        rr, cc = np.nonzero(mask)
        g = mask.shape[0]
        if len(rr) == 0:
            return np.array([(g - 1) / 2.0] * 2), np.zeros(n_theta)
        cen = np.array([rr.mean(), cc.mean()])
        dy, dx = rr - cen[0], cc - cen[1]
        rad = np.hypot(dy, dx)
        ang = np.arctan2(dy, dx) % (2 * np.pi)
        b = np.minimum((ang / (2 * np.pi) * n_theta).astype(np.int64), n_theta - 1)

        # bin 별로 반경 오름차순 정렬 → 첫 틈 이전까지만 본체로 채택
        o = np.lexsort((rad, b))
        bs, rs = b[o], rad[o]
        new_bin = np.empty(len(bs), dtype=bool)
        new_bin[0] = True
        new_bin[1:] = bs[1:] != bs[:-1]
        brk = np.zeros(len(bs), dtype=bool)
        brk[1:] = (np.diff(rs) > cls._BLOB_GAP) & ~new_bin[1:]
        seg = np.cumsum(new_bin | brk)          # 연속 구간 id
        first = np.zeros(n_theta, dtype=np.int64)
        first[bs[new_bin]] = seg[new_bin]        # 각 bin 의 첫 구간 = 본체
        body = seg == first[bs]

        prof = np.zeros(n_theta)
        # 픽셀 중심이 아니라 바깥 모서리까지를 반경으로 본다 (반올림 손실 방지)
        np.maximum.at(prof, bs[body], rs[body] + 0.5)
        # 빈 각도 bin 은 이웃에서 메운다 (blob 은 연결된 덩어리라는 가정)
        idx = np.flatnonzero(prof > 0)
        if len(idx) and len(idx) < n_theta:
            allb = np.arange(n_theta)
            d = np.abs(allb[:, None] - idx[None, :])
            d = np.minimum(d, n_theta - d)       # 원형 거리 기준 최근접 채워진 bin
            prof = prof[idx[np.argmin(d, axis=1)]]
        return cen, prof

    @classmethod
    def _blend_blobs(cls, masks: np.ndarray, weights: np.ndarray,
                     n_theta: int = 180) -> np.ndarray:
        """여러 blob 을 가중 보간해 하나의 blob 으로 (형상 보간).

        픽셀 단위 다수결/평균은 blob 이 조금만 어긋나도 가장자리가 갈라지거나
        렌즈 모양으로 찌그러진다. 여기서는 (무게중심, r(θ)) 를 각각 가중평균한
        뒤 다시 래스터화하므로, 같은 X 의 반복측정이면 경계 노이즈가 평균으로
        지워지고, 서로 다른 X 의 이웃이면 형상이 부드럽게 모핑된다.

        단일 blob 을 그대로 통과시키면 측정치(max height/width)가 보존된다
        — 프로파일 분해→재래스터화가 항등이라는 뜻이라, 보간이 형상을
        체계적으로 부풀리거나 줄이지 않는다는 근거가 된다.
        """
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        g = masks.shape[1]
        cen = np.zeros(2)
        prof = np.zeros(n_theta)
        for m, wi in zip(masks, w):
            c, p = cls._blob_profile(m, n_theta)
            cen += wi * c
            prof += wi * p
        if not prof.any():
            return np.zeros((g, g), dtype=bool)
        rr, cc = np.mgrid[0:g, 0:g]
        dy, dx = rr - cen[0], cc - cen[1]
        ang = np.arctan2(dy, dx) % (2 * np.pi)
        b = np.minimum((ang / (2 * np.pi) * n_theta).astype(np.int64), n_theta - 1)
        return np.hypot(dy, dx) <= prof[b]

    # ─── 마스크 합성 (측정치 → 원형) ────────────────────────────────────────

    @staticmethod
    def _mask_from_extents(h: int, w: int, g: int) -> np.ndarray:
        """max height=h, max width=w 를 **정확히** 만족하는 타원형 마스크.

        convert_y_raw 의 측정 정의(열별 True 개수의 최대 / 행별 True 개수의
        최대)를 그대로 역산한다. 구성:
          · 중심에서 가까운 h 개 행만 채운다 → 중앙 열의 True 개수 = h
          · 각 행의 폭은 중심에서 바깥으로 단조 감소(중첩) → 다른 열은 ≤ h
          · 중심 행의 폭을 정확히 w, 나머지는 그 이하 → 행 최대 = w
        측정 정의 자체는 optimizer 소유라 여기서 바꾸지 않는다 — 역산만 한다.
        """
        h, w = int(h), int(w)
        # 표현 불가능한 측정치는 조용히 클램프하지 않고 즉시 알린다 — 모킹이
        # 실측을 못 재현하는 상황을 숨기면 검증 자체가 무의미해진다.
        if not 1 <= h <= g:
            raise ValueError(f"max height {h} 가 격자 {g} 밖 — 재현 불가")
        if not 2 <= w <= g:  # 폭 1 은 중앙 두 열을 동시에 못 덮어 역산 불가
            raise ValueError(f"max width {w} 가 [2, {g}] 밖 — 재현 불가")
        c = (g - 1) / 2.0
        dr = np.abs(np.arange(g) - c)
        rows = np.argsort(dr, kind="stable")[:h]      # 중심에 가까운 h 행
        a = max(h / 2.0, 0.5)
        frac = np.sqrt(np.maximum(0.0, 1.0 - (dr[rows] / a) ** 2))
        widths = np.maximum(2, np.rint(w * frac).astype(np.int64))
        o = np.argsort(dr[rows], kind="stable")       # 중심에서 먼 순서로 정렬
        widths[o] = np.minimum.accumulate(widths[o])  # 단조 감소 강제 (중첩 보장)
        widths[o[0]] = w                              # 중심 행은 정확히 w
        col_order = np.argsort(np.abs(np.arange(g) - c), kind="stable")
        mask = np.zeros((g, g), dtype=bool)
        for r, wd in zip(rows, widths):
            mask[r, col_order[:wd]] = True
        return mask

    # ─── 공개 API — BenchmarkBase 와 동일 계약 ─────────────────────────────

    def evaluate(self, x: np.ndarray, noisy: bool = True) -> dict:
        """X → y_raw dict. calculator 계약(mask1/mask2/y13/y23)을 그대로 지킨다.

        noisy 는 self.noise_level > 0 일 때만 의미가 있다 (기본 0 = 정확 재생).
        마스크는 근거 관측이 하나면 원형 그대로, 여럿이면 blob 보간으로 만든다.

        주의: 마스크 원형이 있는 경우 **마스크가 원본이고 수치는 파생**이다.
        따라서 근거가 여럿인 점에서는 하류 convert_y_raw 가 재는 값이
        predict() 가 돌려준 가중평균과 ±수 픽셀 다를 수 있다 (형상을 섞은 뒤
        다시 잰 값이기 때문). 근거가 하나면 둘은 정확히 일치한다.
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.int64))
        if x.shape[1] != self.space.n_cols:
            raise ValueError(f"X 는 {self.space.n_cols}컬럼이어야 함: {x.shape}")
        b = len(x)
        Y, status, info = self.predict(x)

        if noisy and self.noise_level > 0:
            Y = Y + self._noise_rng.normal(0.0, 1.0, Y.shape) * (
                self.noise_level * self._scale)

        g = self.RASTER_GRID
        m1 = np.empty((b, g, g), dtype=bool)
        m2 = np.empty((b, g, g), dtype=bool)
        for i in range(b):
            for gi, (out, key, px) in enumerate(
                    ((m1, "mask1", (0, 1)), (m2, "mask2", (3, 4)))):
                srcs, wt = info["sources"][i][gi], info["weights"][i][gi]
                if self._masks is None or status[i, gi] == "no_data":
                    # 마스크 원형이 없다 → 아는 건 측정치 두 개뿐이다. 형상
                    # 디테일을 지어내지 않고 그 측정치를 만족하는 타원을 낸다.
                    out[i] = self._mask_from_extents(
                        round(Y[i, px[0]]), round(Y[i, px[1]]), g)
                elif len(srcs) == 1:
                    out[i] = self._masks[key][srcs[0]]      # 그대로 재생 (요구 3)
                else:
                    # 반복측정(같은 X, 노이즈로 조금씩 다름)이든 이웃 보간이든
                    # blob 형상 자체를 섞는다 — 픽셀 평균이 아니라 (중심, r(θ)).
                    out[i] = self._blend_blobs(self._masks[key][srcs], wt)

        for i in range(b):  # 검증 (2) 의 산출물 — 매 평가의 커버리지를 남긴다
            self.coverage_log.append({
                "eval": self.n_evaluations + i,
                "X": x[i].tolist(),
                "d_hamming": float(info["d_full_min"][i]),
                "status1": status[i, 0], "status2": status[i, 1],
                "d_min1": float(info["d_min"][i, 0]),
                "d_min2": float(info["d_min"][i, 1]),
            })
        self.n_evaluations += b
        return {"mask1": m1, "mask2": m2, "y13": Y[:, 2], "y23": Y[:, 5]}

    # ─── 검증 리포트 ───────────────────────────────────────────────────────

    def report(self) -> dict:
        """**검증 (2)** — 알고리즘이 관측 커버리지를 얼마나 벗어났는지 요약.

        no_data 비율이 높거나 d_hamming 분포가 게이트 위로 치우쳐 있으면
        "알고리즘이 근거 없는 영역만 추천하고 있다" 는 신호다.
        """
        if not self.coverage_log:
            return {"n_evals": 0}
        d = np.array([r["d_hamming"] for r in self.coverage_log])
        s1 = [r["status1"] for r in self.coverage_log]
        s2 = [r["status2"] for r in self.coverage_log]
        counts: dict[str, int] = {}
        for s in s1 + s2:
            counts[s] = counts.get(s, 0) + 1
        n = len(self.coverage_log)
        return {
            "n_evals": n,
            "status_counts": counts,
            "no_data_rate": (s1.count("no_data") + s2.count("no_data")) / (2 * n),
            "exact_rate": (s1.count("exact") + s2.count("exact")) / (2 * n),
            "d_hamming": {"min": float(d.min()), "median": float(np.median(d)),
                          "p90": float(np.percentile(d, 90)), "max": float(d.max())},
            "d_gate": self.d_gate,
        }

    def loo_report(self) -> dict:
        """관측을 하나씩 빼고 보간해 본 leave-one-out 오차 — 표면의 정직도.

        이 오차가 실측 반복오차보다 크면, 보간 구간의 값은 알고리즘을 흔들
        만큼 부정확하다는 뜻이다 (d_gate 를 좁히거나 관측을 더 모을 근거).
        """
        n = len(self._X)
        if n < 3:
            return {"n": n, "note": "관측 3개 미만 — LOO 불가"}
        errs, used = [], 0
        for i in range(n):
            keep = np.arange(n) != i
            sub = SurfaceCalculator(
                self._X[keep], self._Y[keep], k=self.k, power=self.power,
                d_gate=self.d_gate, spread_gate=self.spread_gate, policy="flag")
            yh, st, _ = sub.predict(self._X[i])
            if "no_data" in (st[0, 0], st[0, 1]):
                continue
            errs.append(np.abs(yh[0] - self._Y[i]) / self._scale)
            used += 1
        if not errs:
            return {"n": n, "covered": 0, "note": "전부 커버리지 밖 — 게이트 재검토"}
        e = np.array(errs)
        return {"n": n, "covered": used,
                "mae_rel": dict(zip(OBJECTIVE_NAMES, e.mean(axis=0).round(4))),
                "p90_rel": dict(zip(OBJECTIVE_NAMES,
                                    np.percentile(e, 90, axis=0).round(4)))}


def register_surface(surface: SurfaceCalculator, name: str = "surface") -> str:
    """SurfaceCalculator 를 BENCHMARKS 에 등록해 runner 가 이름으로 쓰게 한다.

    runner 는 `BENCHMARKS[name](noise_seed=seed)` 로 생성하므로, 이미 만들어진
    표면을 그 시그니처로 감싼다 (관측 데이터는 표면에 이미 들어 있다).
    argparse choices 가 import 시점에 고정되므로, CLI 로 쓰려면 파서 생성
    전에 등록해야 한다.
    """
    def _factory(noise_seed: int = 0, **_kw) -> SurfaceCalculator:
        surface._noise_rng = np.random.default_rng(noise_seed)
        return surface

    BENCHMARKS[name] = _factory  # type: ignore[assignment]
    return name


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


def serve_eval(benchmark_name: str, exchange_dir, seed: int,
               surface_data=None) -> int:
    """프로세스 분리 실행의 calculator 한 스텝: x.txt 읽기 → 평가 → y_raw.bin.

    노이즈는 (seed, eval_index) 로 재시딩해 프로세스 경계와 무관하게 결정적이다
    (배치 단위 — 같은 eval_index 는 같은 노이즈). 교환 셸 함수(read_x/write_y_raw)는
    optimizer 소유라 여기서 지연 import 한다 (모듈 순환 회피).

    surface_data 를 주면 합성 벤치마크 대신 **실측 관측으로 만든 반응표면**
    (SurfaceCalculator) 으로 평가한다. 표면은 프로세스마다 새로 로드되므로
    커버리지 판정은 교환 디렉토리의 `coverage.jsonl` 에 append 로 남긴다 —
    스텝 간 기억이 파일뿐인 것은 optimizer 쪽과 같은 제약이다.
    """
    import json
    from pathlib import Path

    from optimizer import read_x, write_y_raw

    d = Path(exchange_dir)
    space = SearchSpace()
    X, eval_index = read_x(d / "x.txt", space=space)
    if surface_data is not None:
        calc = SurfaceCalculator.from_npz(surface_data, noise_seed=seed)
    else:
        calc = BENCHMARKS[benchmark_name](noise_seed=seed)
    calc._noise_rng = np.random.default_rng([seed, eval_index])  # 배치 단위 결정적 노이즈
    raw = calc.evaluate(X)  # noisy=True — 구조화 관측 원형
    write_y_raw(d / "y_raw.bin", raw, eval_index=eval_index)
    if surface_data is not None:  # 커버리지 이력은 파일에만 남는다
        with (d / "coverage.jsonl").open("a") as fh:
            for rec in calc.coverage_log:
                rec["eval"] += eval_index  # 프로세스 로컬 카운터 → 전역 인덱스
                fh.write(json.dumps(rec) + "\n")
    return len(X)


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="calculator — 자가 점검 또는 프로세스 분리 평가")
    _ap.add_argument("--serve-eval", action="store_true",
                     help="파일 기반 프로세스 분리 실행의 calculator 한 스텝")
    _ap.add_argument("--benchmark", choices=list(BENCHMARKS), default="bm1_easy")
    _ap.add_argument("--dir", type=str, default=None, help="교환 디렉토리")
    _ap.add_argument("--seed", type=int, default=0)
    _ap.add_argument("--surface-selfcheck", action="store_true",
                     help="SurfaceCalculator(모킹 반응표면) 자가 점검")
    _ap.add_argument("--surface-data", type=str, default=None, metavar="NPZ",
                     help="--serve-eval 시 합성 벤치마크 대신 실측 관측 npz "
                          "(키: X, Y [, mask1, mask2]) 로 만든 반응표면을 쓴다")
    _args = _ap.parse_args()

    if _args.surface_selfcheck:
        # 실측 대역으로 BenchmarkHard 를 '비싼 TEST' 삼아 관측을 소량 뽑고,
        # 그 관측만으로 세운 반응표면이 요구 1·2·3 을 지키는지 점검한다.
        from optimizer import _mask_extents, convert_y_raw

        space = SearchSpace()
        rng = np.random.default_rng(7)
        truth = BenchmarkHard(noise_seed=0)

        # 관측 60개: 랜덤 40 + 그 중 한 점 주변의 근접 변이 20 (실측은 보통
        # 이렇게 뭉쳐서 쌓인다 — 커버리지가 고르지 않은 상황을 재현)
        X_obs = space.sample(rng, 40)
        anchor = X_obs[0]
        near = np.repeat(anchor[None, :], 20, axis=0)
        for i in range(20):
            for c in rng.choice(space.n_cols, 2, replace=False):
                near[i, c] = rng.integers(space.x_min[c], space.x_max[c] + 1)
        X_obs = np.vstack([X_obs, near])
        raw_obs = truth.evaluate(X_obs)  # 노이즈 포함 = 실측
        Y_obs = convert_y_raw(raw_obs)
        masks_obs = {"mask1": raw_obs["mask1"], "mask2": raw_obs["mask2"]}

        surf = SurfaceCalculator(X_obs, Y_obs, masks_obs)
        print(f"[surface] 관측 {len(X_obs)}개, d_gate={surf.d_gate:.4f} "
              f"(데이터 맞춤 제안값 {surf.suggest_d_gate():.4f})")

        # ── 요구 3: 관측점은 실측값이 그대로 나와야 한다 (측정 경로 전체 통과)
        Y_back = convert_y_raw(surf.evaluate(X_obs, noisy=False))
        # 같은 X 가 두 번 이상 측정된 행은 '반복측정의 평균' 이 정답이다
        # (개별 측정치와는 노이즈만큼 다르다). 그 외 행은 완전 일치해야 한다.
        _, inv, cnt = np.unique(X_obs, axis=0, return_inverse=True,
                                return_counts=True)
        uniq = cnt[inv] == 1
        assert np.array_equal(Y_back[uniq], Y_obs[uniq]), \
            f"관측 재현 실패 — 최대 오차 {np.abs(Y_back[uniq] - Y_obs[uniq]).max()}"
        # 같은 X 의 반복측정: 스칼라는 평균, 마스크는 blob 보간이므로 측정치가
        # 반복측정들 사이(±1px)에 놓여야 한다 — 평균의 반올림이 아니다.
        _PX, _SC = [0, 1, 3, 4], [2, 5]
        n_rep = 0
        for gid in np.flatnonzero(cnt > 1):
            rows = np.flatnonzero(inv == gid)
            obs = Y_obs[rows][:, _PX]
            got = Y_back[rows][:, _PX]
            assert (got >= obs.min(axis=0) - 1).all() and \
                   (got <= obs.max(axis=0) + 1).all(), \
                f"blob 보간 결과가 반복측정 범위 밖\n{got}\n{obs}"
            assert np.allclose(Y_back[rows][:, _SC],
                               Y_obs[rows].mean(axis=0)[_SC]), "반복측정 스칼라 평균 불일치"
            n_rep += len(rows)
        print(f"[요구3] 유일 X {int(uniq.sum())}행 실측과 완전 일치, "
              f"반복측정 {n_rep}행은 스칼라=평균 / 마스크=blob 보간 "
              f"— convert_y_raw 통과 후 검증")

        # 마스크 원형 없이(측정치만) 세운 표면도 유일 X 는 그대로 재현해야 한다
        surf_nm = SurfaceCalculator(X_obs, Y_obs)
        Y_nm = convert_y_raw(surf_nm.evaluate(X_obs, noisy=False))
        assert np.array_equal(Y_nm[uniq], Y_obs[uniq]), "마스크 합성 경로에서 관측 재현 실패"
        print("[요구3] 마스크 원형 없이 합성만으로도 측정치 6/6 열 일치")

        # ── blob 보간 점검 ①: 단일 blob 통과가 항등이어야 한다 (부풀림/줄어듦 없음)
        one = raw_obs["mask1"][:12]
        rt = np.array([SurfaceCalculator._blend_blobs(m[None], [1.0]) for m in one])
        assert np.array_equal(_mask_extents(rt), _mask_extents(one)), \
            "blob 프로파일 라운드트립이 측정치를 바꿈"
        print("[blob] 단일 blob 프로파일 라운드트립 12/12 측정치 보존")

        # ── blob 보간 점검 ②: 실측 마스크에 경계 노이즈가 있어도 섞으면 복원되나
        # (벤치마크 자체에는 flip 노이즈가 없으므로 여기서 직접 만들어 넣는다 —
        #  실데이터의 가장자리 흔들림을 모사)
        anchor, nrep = X_obs[0], 8
        clean_m = raw_obs["mask1"][0]
        nrng = np.random.default_rng(5)
        reps = np.repeat(clean_m[None], nrep, axis=0).copy()
        for i in range(nrep):  # 경계에 인접한 픽셀을 무작위로 뒤집는다
            edge = clean_m ^ np.roll(clean_m, 1, axis=0) | clean_m ^ np.roll(clean_m, 1, axis=1)
            reps[i] ^= edge & (nrng.random(clean_m.shape) < 0.35)
        rep_X = np.repeat(anchor[None, :], nrep, axis=0)
        rep_Y = np.repeat(Y_obs[0][None, :], nrep, axis=0).copy()
        rep_Y[:, 0], rep_Y[:, 1] = _mask_extents(reps)
        s_rep = SurfaceCalculator(rep_X, rep_Y,
                                  {"mask1": reps, "mask2": reps})
        got = convert_y_raw(s_rep.evaluate(anchor[None, :], noisy=False))[0]
        h0, w0 = (int(v[0]) for v in _mask_extents(clean_m[None]))
        e_rep = np.abs(rep_Y[:, [0, 1]] - [h0, w0]).mean(axis=0)
        e_bl = np.abs(got[[0, 1]] - [h0, w0])
        print(f"[blob] 경계노이즈 반복 {nrep}회 — 개별 평균오차 "
              f"{np.round(e_rep, 2).tolist()} → blob 보간 {e_bl.tolist()} 픽셀")
        assert (e_bl <= np.maximum(e_rep, 1.0)).all(), "blob 보간이 개별 관측보다 나쁨"

        # ── 요구 1: 최근접 관측과 그때의 y
        probe = X_obs[0].copy()
        _pc = int(_SET2_COLS[0])  # set2 컬럼 하나만, 반드시 다른 값으로 흔든다
        probe[_pc] = (space.x_max[_pc] if probe[_pc] != space.x_max[_pc]
                      else space.x_min[_pc])
        nb = surf.neighbors(probe, k=2)
        assert nb[0]["distance"] == 1.0 / space.n_cols, nb[0]["distance"]
        print(f"[요구1] 최근접 d={nb[0]['distance']:.4f} "
              f"(idx={nb[0]['index']}), y11={nb[0]['y']['y11']:.1f}")
        # set1 블록 기준으로는 거리 0 — y1x 는 set2 변화에 영향받지 않는다
        assert surf.neighbors(probe, k=1, block="set1")[0]["distance"] == 0.0
        _, st, _ = surf.predict(probe)
        assert st[0, 0] == "exact_block", st[0, 0]
        print(f"[요구1] 블록 인지: group1={st[0,0]}, group2={st[0,1]}")

        # ── 요구 2: 관측에서 먼 X 는 '데이터 없음'
        far = space.sample(np.random.default_rng(999), 200)
        _, st_far, info = surf.predict(far)
        n_nodata = int((st_far == "no_data").sum())
        assert n_nodata > 0, "먼 X 를 전부 커버한다고 주장 — 게이트가 열려 있음"
        print(f"[요구2] 무작위 200점 × 2그룹 중 no_data {n_nodata} "
              f"(d_min 중앙값 {np.median(info['d_min']):.3f})")
        strict = SurfaceCalculator(X_obs, Y_obs, policy="strict")
        try:
            strict.predict(far)
            raise AssertionError("policy='strict' 인데 raise 하지 않음")
        except NoDataError as e:
            print(f"[요구2] strict 모드 차단 OK — {str(e)[:60]}…")

        # ── 검증 (1)(2) 용 리포트
        surf.evaluate(far[:50])
        print(f"[report] {surf.report()}")
        print(f"[LOO]    {surf.loo_report()}")
        raise SystemExit(0)

    if _args.serve_eval:  # runner 가 서브프로세스로 호출하는 경로
        assert _args.dir, "--serve-eval 에는 --dir 필요"
        b = serve_eval(_args.benchmark, _args.dir, _args.seed, _args.surface_data)
        src = _args.surface_data or _args.benchmark
        print(f"[calc-eval] {src} → {b} evals → y_raw.bin")
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
