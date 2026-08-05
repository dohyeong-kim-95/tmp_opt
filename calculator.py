"""calculator.py — 문제 정의 모듈 (X → y_raw).

Calculator 는 '한 번 평가에 비용이 드는 블랙박스 측정기'다. 실제 TEST 가 비싸
알고리즘을 실측으로 돌릴 수 없으므로, 이 모듈은 **이미 확보한 실측 관측으로
세운 반응표면**(`SurfaceCalculator`)을 그 대역으로 제공한다.

한때 여기에 합성 벤치마크 5종(bm1~bm5)이 있었다. 실측 기반으로 방향을 틀면서
전부 걷어냈고, 그 작업에서 배운 것은 `lesson_learned.md` 에 있다.
(코드가 필요하면 `legacy` 브랜치)

──────────────────────────────────────────────────────────────────────────────
문제 구조
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
- 출력 y_raw : **구조화 관측** — boolean blob 마스크 2장 + 스칼라 2개.
    * mask1 (b,G,G): y11 = max height, y12 = max width
    * mask2 (b,G,G): y21 = max height, y22 = max width
    * y13, y23 (b,): 스칼라 (은닉 스케일 적용)
    * 마스크는 덩어리(blob)다 — 형상은 관측 장치가 정하고, 반응표면은 그
      형상을 픽셀 집합이 아니라 덩어리로 다룬다 (`_blend_blobs`).
    * 최대화: y11, y12, y21, y22 / 최소화: y13, y23. 마스크→수치 측정은
      optimizer 의 convert_y_raw 이음새 소관 (calculator 는 원형만 낸다).
    * 값의 범위는 사전에 알 수 없다고 가정한다 (스케일러가 온라인으로 추정).

사용 예:
    calc = SurfaceCalculator.from_jsonl("obs.jsonl")   # 실측 관측
    y_raw = calc.evaluate(x)                        # 계약은 기존 calculator 와 동일
    print(calc.explain(x))                          # 근거 관측 + 커버리지 판정
    print(calc.report())                            # 알고리즘이 얼마나 벗어났나

자가 점검:
    python calculator.py --surface-selfcheck
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


# ──────────────────────────────────────────────────────────────────────────────
# 관측 데이터셋 파일 형식 — 반응표면의 입력 (obs.jsonl)
#
# **한 파일, append-only 텍스트.** 한 줄이 관측 하나이고, 마스크 원형까지
# 그 줄 안에 들어간다. 관측은 이 프로젝트의 진실이므로 사람이 읽고 git 으로
# diff 할 수 있어야 한다는 요구가 형식을 정한다 (history.jsonl 과 같은 원리).
#
#   {"i":0, "block":"one_hot", "X":[...30개...], "Y":[...6개...],
#    "mask1":{"shape":[128,128],"runs":[[row,col0,len],...]}, "mask2":{...}}
#
# 마스크는 **행 단위 run-length** 로 넣는다. 무손실이고, blob 은 행마다 연속
# 구간이 한두 개뿐이라 조밀하다 — 실측 111점 기준 base64(비트팩) 634KB 대비
# RLE 235KB. 무엇보다 `[row, col0, len]` 은 눈으로 읽어도 의미가 보인다
# (base64 는 한 줄 5.7KB 의 불투명한 덩어리다).
#
# float 은 json 의 shortest-round-trip repr 이라 float64 무손실.
# 마스크 키가 없는 줄도 유효하다 — 그 경우 반응표면이 측정치로 형상을 합성한다.
# ──────────────────────────────────────────────────────────────────────────────


def _mask_to_runs(mask: np.ndarray) -> dict:
    """boolean 마스크 → {"shape", "runs"} (행 단위 run-length, 무손실)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"마스크는 2차원이어야 함: {mask.shape}")
    h, w = mask.shape
    # 행마다 양옆을 0 으로 패딩 → run 이 행 경계를 넘지 않고, 시작/끝이 1:1
    pad = np.zeros((h, w + 2), dtype=np.int16)
    pad[:, 1:-1] = mask
    d = np.diff(pad.ravel())
    s = np.flatnonzero(d == 1) + 1
    e = np.flatnonzero(d == -1) + 1
    rows = (s // (w + 2)).tolist()
    cols = (s % (w + 2) - 1).tolist()
    lens = (e - s).tolist()
    return {"shape": [h, w],
            "runs": [[r, c, ln] for r, c, ln in zip(rows, cols, lens)]}


def _runs_to_mask(enc: dict) -> np.ndarray:
    """{"shape", "runs"} → boolean 마스크."""
    h, w = (int(v) for v in enc["shape"])
    m = np.zeros((h, w), dtype=bool)
    for r, c0, ln in enc["runs"]:
        if not (0 <= r < h and 0 <= c0 and c0 + ln <= w and ln > 0):
            raise ValueError(f"run 이 형상 {h}x{w} 을 벗어남: {[r, c0, ln]}")
        m[r, c0:c0 + ln] = True
    return m


def save_observations(path, X, Y, block=None, masks=None,
                      append: bool = False) -> None:
    """관측을 obs.jsonl 로 쓴다 (마스크 포함, 한 파일).

    Args:
        X      : (n, 30) 정수
        Y      : (n, 6) float — 열 순서 = OBJECTIVE_NAMES
        block  : (n,) 설계 블록 라벨 (선택)
        masks  : {"mask1": (n,G,G) bool, "mask2": ...} (선택)
        append : True 면 이어쓴다. 관측이 늘어나도 전체 재작성이 없다.
    """
    import json
    from pathlib import Path

    path = Path(path)
    X = np.atleast_2d(np.asarray(X, dtype=np.int64))
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    if len(X) != len(Y):
        raise ValueError(f"X {len(X)} 행 ≠ Y {len(Y)} 행")
    if Y.shape[1] != len(OBJECTIVE_NAMES):
        raise ValueError(f"Y 는 {len(OBJECTIVE_NAMES)}열이어야 함: {Y.shape}")
    if not np.isfinite(Y).all():
        raise ValueError("Y 에 비유한값 — 결측은 호출자가 먼저 처리할 것")
    blk = ["" for _ in X] if block is None else [str(b) for b in block]
    if len(blk) != len(X):
        raise ValueError(f"block 길이 {len(blk)} ≠ 관측 수 {len(X)}")
    if masks is not None:
        for key in ("mask1", "mask2"):
            if len(masks[key]) != len(X):
                raise ValueError(f"{key} {len(masks[key])}개 ≠ 관측 {len(X)}개")

    base = 0
    if append and path.exists():
        with path.open() as fh:
            base = sum(1 for line in fh if line.strip())
    with path.open("a" if append else "w") as fh:
        for i in range(len(X)):
            rec = {"i": base + i, "block": blk[i],
                   "X": X[i].tolist(), "Y": Y[i].tolist()}
            if masks is not None:
                rec["mask1"] = _mask_to_runs(masks["mask1"][i])
                rec["mask2"] = _mask_to_runs(masks["mask2"][i])
            fh.write(json.dumps(rec) + "\n")


def load_observations(path) -> dict:
    """obs.jsonl → {"X", "Y", "block", "mask1", "mask2"}.

    마스크가 한 줄이라도 빠져 있으면 mask1/mask2 는 None 이다 (일부만 있는
    상태는 어느 관측의 형상인지 밀릴 수 있어 받지 않는다 — 즉시 raise).
    """
    import json
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"관측 파일 없음: {path}")
    X, Y, blk, m1, m2 = [], [], [], [], []
    with path.open() as fh:
        for ln, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln + 1} JSON 파싱 실패 — {e}") from e
            X.append(rec["X"])
            Y.append(rec["Y"])
            blk.append(rec.get("block", ""))
            if ("mask1" in rec) != ("mask2" in rec):
                raise ValueError(f"{path}:{ln + 1} 마스크가 한 장만 있음")
            if "mask1" in rec:
                m1.append(_runs_to_mask(rec["mask1"]))
                m2.append(_runs_to_mask(rec["mask2"]))
    if not X:
        raise ValueError(f"{path}: 관측이 0개")
    if m1 and len(m1) != len(X):
        raise ValueError(
            f"{path}: 마스크가 {len(m1)}/{len(X)} 줄에만 있음 — 전부 있거나 전부 없어야 함")
    return {"X": np.asarray(X, dtype=np.int64),
            "Y": np.asarray(Y, dtype=np.float64),
            "block": np.asarray(blk),
            "mask1": np.array(m1) if m1 else None,
            "mask2": np.array(m2) if m2 else None}


class NoDataError(RuntimeError):
    """관측 커버리지 밖의 X 가 요청됨 (policy="strict" 일 때 raise)."""


#: 목적 인덱스 → 의존 컬럼 그룹 (0,1,2 = y11,y12,y13 / 3,4,5 = y21,y22,y23)
_OBJ_GROUP: tuple[int, ...] = (1, 1, 1, 2, 2, 2)
_GROUP_COLS: dict[int, np.ndarray] = {1: _GROUP1_COLS, 2: _GROUP2_COLS}


class SurfaceCalculator:
    """실측 (X, y) 관측으로 만든 반응표면 — calculator 계약의 모킹 구현.

    공개 계약: `evaluate(x, noisy=...)` 가
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
    #: 마스크 한 변 픽셀 수 — 실측 관측 장치의 격자 크기와 같아야 한다
    RASTER_GRID: int = 128
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
    def from_jsonl(cls, path, **kwargs) -> "SurfaceCalculator":
        """obs.jsonl 에서 로드 (`load_observations` 형식)."""
        d = load_observations(path)
        masks = None if d["mask1"] is None else {"mask1": d["mask1"],
                                                 "mask2": d["mask2"]}
        return cls(d["X"], d["Y"], masks, **kwargs)

    def save_jsonl(self, path, block=None, append: bool = False) -> None:
        """관측을 obs.jsonl 로 저장 (마스크 원형이 있으면 함께)."""
        save_observations(path, self._X, self._Y, block=block,
                          masks=self._masks, append=append)

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

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        """4-이웃 최대 연결 성분만 남긴다 (본체에서 떨어져 나온 낱개 픽셀 제거).

        실측 마스크에서 가장자리에 흩뿌려지는 이상치를 거르는 **올바른**
        연산이다 — blob 은 하나의 덩어리라는 것이 이 자료형의 정의이므로.
        (반경 방향으로 자르는 방식은 각도 bin 이 중심 근처에서 서브픽셀 폭이라
         본체 자체가 잘려나간다 — 형상이 축 방향만 남은 별이 된다)
        """
        if not mask.any():
            return mask
        rr, cc = np.nonzero(mask)
        cen = np.array([rr.mean(), cc.mean()])
        seed = np.argmin((rr - cen[0]) ** 2 + (cc - cen[1]) ** 2)
        comp = np.zeros_like(mask)
        comp[rr[seed], cc[seed]] = True
        while True:  # 팽창 ∩ 마스크 를 고정점까지 (전부 벡터 연산)
            grown = comp.copy()
            grown[1:, :] |= comp[:-1, :]
            grown[:-1, :] |= comp[1:, :]
            grown[:, 1:] |= comp[:, :-1]
            grown[:, :-1] |= comp[:, 1:]
            grown &= mask
            if grown.sum() == comp.sum():
                return comp
            comp = grown

    @classmethod
    def _blob_profile(cls, mask: np.ndarray,
                      n_theta: int) -> tuple[np.ndarray, np.ndarray]:
        """boolean blob → (무게중심, 각도별 반경 프로파일 r(θ)).

        마스크를 픽셀 집합이 아니라 **형상**으로 다룬다. 무게중심에서 본
        각도 bin 별 최대 반경을 재면 크기·위치·찌그러짐이 (c, r(θ)) 로 분리돼
        서로 다른 blob 사이를 자연스럽게 섞을 수 있다.

        이상치 제거는 반경이 아니라 **연결성**으로 한다 (`_largest_component`).
        본체에 붙어 있는 픽셀은 전부 형상의 일부이고, 떨어져 나온 것만 버린다.
        비어 있는 마스크는 (중심, 전부 0) 으로 돌려준다.
        """
        g = mask.shape[0]
        mask = cls._largest_component(np.asarray(mask, dtype=bool))
        rr, cc = np.nonzero(mask)
        if len(rr) == 0:
            return np.array([(g - 1) / 2.0] * 2), np.zeros(n_theta)
        cen = np.array([rr.mean(), cc.mean()])
        dy, dx = rr - cen[0], cc - cen[1]
        rad = np.hypot(dy, dx)
        ang = np.arctan2(dy, dx) % (2 * np.pi)
        b = np.minimum((ang / (2 * np.pi) * n_theta).astype(np.int64), n_theta - 1)

        prof = np.zeros(n_theta)
        # 픽셀 중심이 아니라 바깥 모서리까지를 반경으로 본다 (반올림 손실 방지)
        np.maximum.at(prof, b, rad + 0.5)
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
        """max height=h, max width=w 를 **정확히** 만족하는 다이아몬드형 마스크.

        convert_y_raw 의 측정 정의(열별 True 개수의 최대 / 행별 True 개수의
        최대)를 그대로 역산한다. 구성:
          · 중심에서 가까운 h 개 행만 채운다 → 중앙 열의 True 개수 = h
          · 각 행의 폭은 중심에서 바깥으로 단조 감소(중첩) → 다른 열은 ≤ h
          · 중심 행의 폭을 정확히 w, 나머지는 그 이하 → 행 최대 = w
        측정 정의 자체는 optimizer 소유라 여기서 바꾸지 않는다 — 역산만 한다.

        폭 프로파일은 |dr| 에 **선형**(다이아몬드 = L1 공)이다. 관측 장치의
        blob 이 다이아몬드에 가까우므로 같은 계열로 맞춘다. 이 경로는 실측
        마스크 원형이 없어 **측정치 두 개밖에 모를 때만** 쓰이므로, 형상
        디테일을 지어내지 않고 그 계열의 가장 단순한 형태를 낸다.
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
        frac = np.maximum(0.0, 1.0 - dr[rows] / a)  # L1 공 → 다이아몬드
        widths = np.maximum(2, np.rint(w * frac).astype(np.int64))
        o = np.argsort(dr[rows], kind="stable")       # 중심에서 먼 순서로 정렬
        widths[o] = np.minimum.accumulate(widths[o])  # 단조 감소 강제 (중첩 보장)
        widths[o[0]] = w                              # 중심 행은 정확히 w
        col_order = np.argsort(np.abs(np.arange(g) - c), kind="stable")
        mask = np.zeros((g, g), dtype=bool)
        for r, wd in zip(rows, widths):
            mask[r, col_order[:wd]] = True
        return mask

    # ─── 공개 API — calculator 계약 ────────────────────────────────────────

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


def serve_eval(surface_data, exchange_dir, seed: int) -> int:
    """프로세스 분리 실행의 calculator 한 스텝: x.txt 읽기 → 평가 → y_raw.bin.

    surface_data(obs.jsonl 경로)로 실측 반응표면을 만들어 평가한다. 교환 셸 함수
    (read_x/write_y_raw)는 optimizer 소유라 여기서 지연 import 한다 (모듈 순환
    회피).

    표면은 프로세스마다 새로 로드되므로 커버리지 판정은 교환 디렉토리의
    `coverage.jsonl` 에 append 로 남긴다 — 스텝 간 기억이 파일뿐인 것은
    optimizer 쪽과 같은 제약이다.

    노이즈는 (seed, eval_index) 로 재시딩해 프로세스 경계와 무관하게 결정적이다
    (표면 기본값 noise_level=0 이면 애초에 무관하다).
    """
    import json
    from pathlib import Path

    from optimizer import read_x, write_y_raw

    d = Path(exchange_dir)
    X, eval_index = read_x(d / "x.txt", space=SearchSpace())
    calc = SurfaceCalculator.from_jsonl(surface_data, noise_seed=seed)
    calc._noise_rng = np.random.default_rng([seed, eval_index])
    raw = calc.evaluate(X)  # 구조화 관측 원형
    write_y_raw(d / "y_raw.bin", raw, eval_index=eval_index)
    with (d / "coverage.jsonl").open("a") as fh:  # 커버리지 이력은 파일에만 남는다
        for rec in calc.coverage_log:
            rec["eval"] += eval_index  # 프로세스 로컬 카운터 → 전역 인덱스
            fh.write(json.dumps(rec) + "\n")
    return len(X)


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="calculator — 자가 점검 또는 프로세스 분리 평가")
    _ap.add_argument("--serve-eval", action="store_true",
                     help="파일 기반 프로세스 분리 실행의 calculator 한 스텝")
    _ap.add_argument("--dir", type=str, default=None, help="교환 디렉토리")
    _ap.add_argument("--seed", type=int, default=0)
    _ap.add_argument("--surface-selfcheck", action="store_true",
                     help="SurfaceCalculator(모킹 반응표면) 자가 점검")
    _ap.add_argument("--surface-data", type=str, default=None, metavar="JSONL",
                     help="실측 관측 obs.jsonl — --serve-eval 에 필수")
    _args = _ap.parse_args()

    if _args.surface_selfcheck:
        from optimizer import _mask_extents, convert_y_raw

        class _FakeInstrument:
            """자가 점검 전용 가짜 측정 장치 — '비싼 TEST' 의 대역.

            여기서 뽑은 관측만으로 SurfaceCalculator 를 세우고 요구 1·2·3 을
            점검한다. 지형이 무엇이든 검증에는 무관하므로 매끄러운 단봉
            가중합이면 충분하다. 중요한 건 계약뿐이다:
            X → {mask1, mask2, y13, y23}, 그리고 같은 X 는 같은 마스크.
            (스칼라에만 가우시안 노이즈 — 반복측정 경로를 만들기 위해)
            """

            def __init__(self, seed: int = 0, g: int = 128):
                self.g = g
                r = np.random.default_rng(seed)
                self.space = SearchSpace()
                n = self.space.n_cols
                self._peaks = r.uniform(0.0, 1.0, (6, n))
                w = r.uniform(0.5, 1.5, (6, n))
                m = np.zeros((6, n))
                m[0:3, list(_GROUP1_COLS)] = 1.0
                m[3:6, list(_GROUP2_COLS)] = 1.0
                w *= m
                self._w = w / w.sum(axis=1, keepdims=True)
                self._rng = np.random.default_rng(seed + 1)
                probe = self._latent(self.space.to_unit(
                    self.space.sample(np.random.default_rng(999), 2048)))
                self._mu, self._sd = probe.mean(axis=0), probe.std(axis=0)

            def _latent(self, u):
                return np.einsum("nkc,kc->nk",
                                 1.0 - (u[:, None, :] - self._peaks[None]) ** 2,
                                 self._w)

            def _render(self, sh, sw):
                g = self.g
                coord = np.arange(g, dtype=np.float64) - (g - 1) / 2.0
                s = 1.0 / np.sqrt(2.0 * np.log(2.0))
                out = np.empty((len(sh), g, g), dtype=bool)
                for i in range(len(sh)):
                    gr = np.exp(-0.5 * (coord / (sh[i] * s)) ** 2)
                    gc = np.exp(-0.5 * (coord / (sw[i] * s)) ** 2)
                    out[i] = (gr[:, None] * gc[None, :]) >= 0.5
                return out

            def evaluate(self, X, noisy: bool = True) -> dict:
                X = np.atleast_2d(np.asarray(X, dtype=np.int64))
                lat = self._latent(self.space.to_unit(X))
                z = (lat - self._mu) / (3.0 * self._sd + 1e-12)
                semi = np.clip(np.rint(28 + 26 * z), 6, self.g // 2 - 4).astype(int)
                sc = lat[:, [2, 5]]
                if noisy:
                    sc = sc + self._rng.normal(0, 1, sc.shape) * 0.05 * self._sd[[2, 5]]
                return {"mask1": self._render(semi[:, 0], semi[:, 1]),
                        "mask2": self._render(semi[:, 3], semi[:, 4]),
                        "y13": sc[:, 0] * 0.005 + 0.02,
                        "y23": sc[:, 1] * 0.005 + 0.02}

        space = SearchSpace()
        rng = np.random.default_rng(7)
        truth = _FakeInstrument(seed=303)

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

        # ── blob 보간 점검 ①: 단일 blob 통과가 항등이어야 한다.
        # 측정치(extent)만 보면 안 된다 — 형상이 축 방향만 남은 별로 무너져도
        # max height/width 는 그대로다 (실제로 그 버그가 있었다). IoU 로 본다.
        one = raw_obs["mask1"][:12]
        rt = np.array([SurfaceCalculator._blend_blobs(m[None], [1.0]) for m in one])
        assert np.array_equal(_mask_extents(rt), _mask_extents(one)), \
            "blob 프로파일 라운드트립이 측정치를 바꿈"
        ious = np.array([(a & b).sum() / max((a | b).sum(), 1)
                         for a, b in zip(one, rt)])
        assert ious.min() > 0.97, f"blob 형상이 보존되지 않음 — IoU 최소 {ious.min():.3f}"
        print(f"[blob] 단일 blob 라운드트립 12/12 — 측정치 보존 + "
              f"형상 IoU 평균 {ious.mean():.3f} 최소 {ious.min():.3f}")

        # 떨어져 나온 이상치 픽셀은 연결성으로 걸러야 한다
        spk = one[0].copy()
        _srng = np.random.default_rng(3)
        for _ in range(40):
            spk[_srng.integers(0, spk.shape[0]), _srng.integers(0, spk.shape[1])] = True
        kept = SurfaceCalculator._blend_blobs(spk[None], [1.0])
        iou_spk = (one[0] & kept).sum() / max((one[0] | kept).sum(), 1)
        assert iou_spk > 0.95, f"speckle 40개에 형상이 흔들림 — IoU {iou_spk:.3f}"
        print(f"[blob] speckle 40개 주입에도 형상 유지 — IoU {iou_spk:.3f}")

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

        # ── 파일 형식: obs.jsonl 왕복 + append 가 무손실인가
        import tempfile
        from pathlib import Path as _P

        with tempfile.TemporaryDirectory() as _td:
            f = _P(_td) / "obs.jsonl"
            save_observations(f, X_obs, Y_obs, block=["t"] * len(X_obs),
                              masks=masks_obs)
            back = load_observations(f)
            assert np.array_equal(back["X"], X_obs), "X 왕복 불일치"
            assert np.array_equal(back["Y"], Y_obs), "Y 왕복 불일치 (float64 무손실 실패)"
            assert np.array_equal(back["mask1"], masks_obs["mask1"]), "mask1 왕복 불일치"
            assert np.array_equal(back["mask2"], masks_obs["mask2"]), "mask2 왕복 불일치"
            n0 = len(X_obs)
            save_observations(f, X_obs[:5], Y_obs[:5], block=["u"] * 5,
                              masks={k: v[:5] for k, v in masks_obs.items()},
                              append=True)
            g2 = load_observations(f)
            assert len(g2["X"]) == n0 + 5 and np.array_equal(g2["X"][:n0], X_obs), \
                "append 가 기존 관측을 훼손"
            assert np.array_equal(g2["mask1"][n0:], masks_obs["mask1"][:5]), \
                "append 된 마스크 불일치"
            kb = f.stat().st_size / 1e3
            print(f"[형식] obs.jsonl 왕복 무손실 (X·Y·마스크 전부), "
                  f"append {n0}→{n0 + 5}줄, {kb:.0f} KB")
            # 마스크 없는 줄도 유효해야 한다
            f2 = _P(_td) / "nomask.jsonl"
            save_observations(f2, X_obs[:3], Y_obs[:3])
            assert load_observations(f2)["mask1"] is None, "마스크 없는 파일 처리 실패"
            print("[형식] 마스크 없는 obs.jsonl 도 유효 (표면이 측정치로 합성)")

        # ── 검증 (1)(2) 용 리포트
        surf.evaluate(far[:50])
        print(f"[report] {surf.report()}")
        print(f"[LOO]    {surf.loo_report()}")
        raise SystemExit(0)

    if _args.serve_eval:  # runner 가 서브프로세스로 호출하는 경로
        assert _args.dir, "--serve-eval 에는 --dir 필요"
        assert _args.surface_data, "--serve-eval 에는 --surface-data 필요"
        b = serve_eval(_args.surface_data, _args.dir, _args.seed)
        print(f"[calc-eval] {_args.surface_data} → {b} evals → y_raw.bin")
        raise SystemExit(0)

    # 인자 없음 → 공간 명세만 확인하고 사용법 안내
    space = SearchSpace()
    print(f"search space log10 size = {space.log10_size:.2f}  "
          f"(n_cols={space.n_cols}, blocks={list(space.blocks)})")
    print(f"목적 {OBJECTIVE_NAMES} / sense {OBJECTIVE_SENSES}")
    print("이 모듈의 calculator 는 SurfaceCalculator (실측 반응표면) 하나다.")
    print("  자가 점검 : python calculator.py --surface-selfcheck")
    print("  분리 평가 : python calculator.py --serve-eval "
          "--surface-data obs.jsonl --dir xchg/")
