"""surface.py — pseudo-반응표면. **BO 의 surrogate 다.**

실측이 5분/점이라 optimizer 를 실측으로 굴릴 수 없다. 그래서 지금까지 쌓은
관측으로 표면을 세우고, 다음에 잴 X 를 그 위에서 고른다.

┌──────────────────────────────────────────────────────────────────────────┐
│ 관측이 수십 점, 조합공간이 10^15                                          │
│   후보 X 는 **거의 전부** 관측에서 멀다. 그래서 이 표면의 일은 "값을 잘   │
│   맞히는 것" 이 아니라 **어디를 아는지 정직하게 말하는 것**이다:          │
│     · 데이터가 있는 곳 → 실측 수준으로 정확히                             │
│     · 그 외            → σ 를 관측 전체 산포까지 키워 "모름" 이라고        │
│   값만 내고 σ 를 안 내면 acquisition 이 성립하지 않는다.                  │
└──────────────────────────────────────────────────────────────────────────┘

μ(x) — 예측 평균 (판정 사다리, 목적별로 독립)
  1. `exact`       30컬럼 전부 일치 → **실측값 그대로**. 반복측정이면 평균
  2. `exact_block` 그 목적의 의존 컬럼이 전부 일치 → 그 관측들 평균
                   (그룹 밖 컬럼은 이 목적에 영향이 없으므로 정당하다)
  3. `near`/`far`  XGB 배깅 앙상블 예측

  1·2 가 모델을 덮어쓰는 게 핵심이다. 트리 앙상블은 관측점도 평활하므로,
  모델에만 맡기면 "데이터가 있는 곳은 정확히" 를 원리적으로 못 지킨다.

σ(x) — 불확실성 (세 항의 제곱합)
  σ² = σ_ens(x)²          앙상블 불일치 (배깅 멤버들의 예측 표준편차)
     + σ_noise²           관측 노이즈 바닥 (**반복측정에서 직접 추정**)
     + (α·σ_obs·f(d))²    거리 팽창

  거리 팽창이 없으면 안 된다. **트리 앙상블은 외삽에서 오히려 자신만만해진다**
  — 데이터 밖의 점도 결국 학습된 leaf 로 떨어지고, 멤버들이 같은 leaf 를 고르면
  불일치가 0 이 된다. GP 와 정반대이고, XGB-BO 의 대표적 함정이다. 그래서
  "관측에서 얼마나 먼가" 를 σ 에 명시적으로 넣는다:
      f(d) = 1 − exp(−d / d_ref)     d=0 → 0,  d ≫ d_ref → 1
  d_ref 는 관측끼리의 최근접거리 하위 퍼센타일이다("관측이 실제로 이웃을 갖는
  간격"). **낮은** 퍼센타일을 쓰는 게 핵심 — 높은 걸 쓰면 데이터가 성길수록
  게이트가 넓어져서, 근거가 없을수록 자신 있게 답하는 정반대 동작이 된다.

α — 캘리브레이션 계수
  σ 는 크기가 맞아야 쓸모가 있다. 과소평가하면 BO 가 근거 없는 곳을 안전하다
  믿고, 과대평가하면 탐험만 하다 끝난다. LOO 로 z=(y−μ)/σ 를 구해 **std(z)=1**
  이 되도록 α 를 맞춘다. 맞춘 뒤의 std(z) 가 `report()` 에 나오며, 이게 1 에서
  멀면 이 표면의 σ 는 믿을 게 못 된다.

거리 — unit 공간 정규화 L1, **목적의 의존 컬럼으로 제한**
  해밍(몇 컬럼이나 다른가)은 card=30 컬럼에서 1칸 차이와 29칸 차이를 같게 본다.
  순서형 컬럼에는 unit 공간 L1 이 맞다. 해밍은 보조 지표로 함께 보고한다.
  그룹 제한 때문에 커버리지 판정이 목적별로 갈린다 — group1 은 데이터가 있고
  group2 는 없는 X 가 실제로 존재한다.

사용:
    surf = Surface.fit(records)                 # scorer 기본 = score.DEFAULT
    p = surf.predict(X)                         # p.mu, p.sigma, p.status, p.d_min
    print(surf.report())                        # 노이즈 바닥 · 캘리브레이션 · 커버리지
    print(surf.explain(x))                      # 근거 관측

자가 점검:
    python surface.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import score
from record import Record
from space import SearchSpace

#: 후보를 한 번에 이만큼씩 끊어 거리 계산 (acquisition 이 수만 점을 넣는다)
_CHUNK = 4096


def _dof_for(hi_mult: float, conf: float = 0.95) -> int:
    """노이즈 바닥 CI 의 상한 배수를 `hi_mult` 이하로 만드는 최소 dof."""
    from scipy.stats import chi2
    a = (1.0 - conf) / 2.0
    for k in range(2, 4001):
        if np.sqrt(k / chi2.ppf(a, k)) <= hi_mult:
            return k
    return 4000


# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Prediction:
    """표면의 답. 모든 배열이 (n_query, n_obj) 로 목적별이다."""

    mu: np.ndarray        # 예측 평균
    sigma: np.ndarray     # 불확실성 (>0)
    status: np.ndarray    # "exact" | "exact_block" | "near" | "far"
    d_min: np.ndarray     # 그 목적의 의존 컬럼 기준 최근접 관측까지의 거리
    hamming: np.ndarray   # 같은 기준의 해밍거리 (보조 지표)

    @property
    def known(self) -> np.ndarray:
        """(n_query, n_obj) bool — 근거가 있는 칸."""
        return np.isin(self.status, ("exact", "exact_block", "near"))


# ──────────────────────────────────────────────────────────────────────────────


class Surface:
    """관측으로 세운 surrogate. `fit` 으로 만든다.

    Args:
        records     : 관측 (record.Record). 같은 x 가 여러 개면 반복측정이다
        scorer      : 점수 정의. 바꾸면 표면이 통째로 달라진다 (그게 정상)
        n_ensemble  : 배깅 멤버 수. 늘리면 σ_ens 가 안정되고 느려진다
        d_ref_q     : d_ref 를 뽑을 최근접거리 퍼센타일 (낮게 쓸 것)
        loo_ensemble: 캘리브레이션 LOO 때 쓸 멤버 수. LOO 는 관측 수만큼 재학습
                      하므로 여기가 비용을 지배한다. σ_ens 의 대략적 크기만
                      필요하므로 본 앙상블보다 작게 쓴다
        alpha       : 거리 팽창 계수. None 이면 LOO 캘리브레이션으로 정한다
        calibrate   : False 면 alpha=1 로 두고 LOO 를 돌리지 않는다 (빠름)
    """

    def __init__(self, records, *, scorer: score.Scorer | None = None,
                 space: SearchSpace | None = None, n_ensemble: int = 8,
                 seed: int = 0, d_ref_q: float = 20.0, loo_ensemble: int = 4,
                 alpha: float | None = None, calibrate: bool = True) -> None:
        records = list(records)
        if not records:
            raise ValueError("관측이 0개 — 표면을 만들 수 없다")
        self.space = space or SearchSpace()
        self.scorer = scorer or score.get()
        self.n_ensemble = int(n_ensemble)
        self.loo_ensemble = max(2, min(int(loo_ensemble), int(n_ensemble)))
        self.seed = int(seed)

        self.X = np.array([r.x for r in records], dtype=np.int64)
        if self.X.shape[1] != self.space.n_cols:
            raise ValueError(f"x 는 {self.space.n_cols}컬럼이어야 함: {self.X.shape}")
        self.Y = self.scorer(records)
        self.U = self.space.to_unit(self.X)
        self.n, self.n_obj = self.Y.shape

        # 목적별 의존 컬럼 (그룹 1 → common+set1, 2 → common+set2)
        common = self.space.block_cols("common")
        self._group_cols = {
            1: np.concatenate([common, self.space.block_cols("set1")]),
            2: np.concatenate([common, self.space.block_cols("set2")]),
        }
        self.obj_cols = [self._group_cols[g] for g in self.scorer.groups]

        # 관측 산포 — "모름" 일 때 σ 가 도달할 크기
        self.y_scale = np.maximum(self.Y.std(axis=0, ddof=0), 1e-12)

        self.noise = self._estimate_noise()
        self.d_ref = self._estimate_d_ref(d_ref_q)
        self._models = self._fit_models(np.arange(self.n))
        self.alpha = 1.0 if alpha is None else float(alpha)
        self._loo_cache: dict | None = None
        self._replay_floor = np.zeros(self.n_obj)
        if alpha is None and calibrate:
            self.alpha = self._calibrate()
        self._replay_floor = self._compute_replay_floor()

    # ─── 생성 ──────────────────────────────────────────────────────────────

    @classmethod
    def fit(cls, records, **kw) -> "Surface":
        return cls(records, **kw)

    @classmethod
    def from_jsonl(cls, path, **kw) -> "Surface":
        from record import read
        return cls(read(path), **kw)

    # ─── 노이즈 바닥 — 반복측정에서만 나온다 ────────────────────────────────

    def _estimate_noise(self) -> np.ndarray:
        """같은 x 를 여러 번 잰 그룹들의 pooled 표준편차 (목적별).

        반복측정이 없으면 0 이다. 그 경우 "표면이 틀린 건지 측정이 시끄러운
        건지" 구분할 근거가 없다 — `report()` 가 경고한다.
        """
        idx: dict[bytes, list[int]] = {}
        for i in range(self.n):
            idx.setdefault(self.X[i].tobytes(), []).append(i)
        ss = np.zeros(self.n_obj)
        dof = 0
        for rows in idx.values():
            if len(rows) < 2:
                continue
            ss += self.Y[rows].var(axis=0, ddof=1) * (len(rows) - 1)
            dof += len(rows) - 1
        self.n_repeat_groups = sum(1 for v in idx.values() if len(v) > 1)
        self.n_unique_x = len(idx)
        self.noise_dof = dof
        return np.sqrt(ss / dof) if dof else np.zeros(self.n_obj)

    def noise_ci(self, conf: float = 0.95) -> tuple[float, float]:
        """노이즈 바닥 추정의 배수 신뢰구간 (χ² 기반, 목적 공통).

        반복측정이 적으면 이 구간이 매우 넓다. 반복 8회(dof 7)면 참값이
        추정치의 0.66~2.0배 어디에나 있을 수 있다 — 노이즈 바닥은 σ 의 하한이자
        LOO 오차의 비교 기준이므로, 이 구간을 모르고 쓰면 표면을 과신하게 된다.
        """
        if not self.noise_dof:
            return (0.0, float("inf"))
        from scipy.stats import chi2
        k = self.noise_dof
        a = (1.0 - conf) / 2.0
        return (float(np.sqrt(k / chi2.ppf(1 - a, k))),
                float(np.sqrt(k / chi2.ppf(a, k))))

    # ─── 거리 ──────────────────────────────────────────────────────────────

    def _dist(self, Uq: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """(nq, n) unit 공간 정규화 L1 — 의존 컬럼으로 제한."""
        return np.abs(Uq[:, None, cols] - self.U[None, :, cols]).mean(axis=2)

    def _hamming(self, Xq: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """(nq, n) 정규화 해밍 — 몇 컬럼이나 다른가."""
        return (Xq[:, None, cols] != self.X[None, :, cols]).mean(axis=2)

    def _estimate_d_ref(self, q: float) -> float:
        """관측끼리의 leave-one-out 최근접거리 q퍼센타일 — "이웃을 갖는 간격".

        전 컬럼 기준 하나로 잡는다(그룹별로 나누면 표본이 더 줄어든다).
        관측이 1개뿐이면 이웃 간격을 정의할 근거가 없어 0 이다 — 그러면
        f(d)=1 이 되어 관측점 외에는 전부 "모름" 이 된다(정직한 퇴화).
        """
        if self.n < 2:
            return 0.0
        all_cols = np.arange(self.space.n_cols)
        d = self._dist(self.U, all_cols)
        np.fill_diagonal(d, np.inf)
        near = d.min(axis=1)
        near = near[np.isfinite(near)]
        return float(np.percentile(near, q)) if len(near) else 0.0

    def _inflate(self, d: np.ndarray) -> np.ndarray:
        """거리 → [0, 1) 팽창 계수. d=0 에서 0, d ≫ d_ref 에서 1."""
        if self.d_ref <= 0:
            return np.ones_like(d)
        return 1.0 - np.exp(-d / self.d_ref)

    # ─── 모델 ──────────────────────────────────────────────────────────────

    def _fit_models(self, rows: np.ndarray, n_members: int | None = None) -> list[list]:
        """목적별 XGB 배깅 앙상블. 의존 컬럼만 넣는다.

        멤버마다 부트스트랩 표본이 달라 예측이 갈린다 — 그 불일치가 σ_ens 다.
        수십 점이므로 얕은 트리(depth 3)에 작은 learning rate 를 쓴다.
        """
        from xgboost import XGBRegressor

        n_members = self.n_ensemble if n_members is None else int(n_members)
        out = []
        for j in range(self.n_obj):
            cols = self.obj_cols[j]
            Xj, yj = self.U[np.ix_(rows, cols)], self.Y[rows, j]
            members = []
            for m in range(n_members):
                rng = np.random.default_rng(self.seed * 1000 + j * 100 + m)
                boot = rng.integers(0, len(rows), len(rows))
                mdl = XGBRegressor(
                    n_estimators=150, max_depth=3, learning_rate=0.08,
                    subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                    min_child_weight=1, n_jobs=1, verbosity=0,
                    random_state=self.seed * 1000 + j * 100 + m)
                mdl.fit(Xj[boot], yj[boot])
                members.append(mdl)
            out.append(members)
        return out

    def _model_predict(self, Uq: np.ndarray, models) -> tuple[np.ndarray, np.ndarray]:
        """(nq, n_obj) 앙상블 평균과 표준편차."""
        nq = len(Uq)
        mu = np.empty((nq, self.n_obj))
        sd = np.empty((nq, self.n_obj))
        for j in range(self.n_obj):
            cols = self.obj_cols[j]
            P = np.stack([m.predict(Uq[:, cols]) for m in models[j]], axis=0)
            mu[:, j] = P.mean(axis=0)
            sd[:, j] = P.std(axis=0, ddof=0)
        return mu, sd

    # ─── 예측 ──────────────────────────────────────────────────────────────

    def predict(self, Xq: np.ndarray) -> Prediction:
        """X → Prediction. 배치로 받는다 (acquisition 이 수만 점을 넣는다)."""
        Xq = np.atleast_2d(np.asarray(Xq, dtype=np.int64))
        if Xq.shape[1] != self.space.n_cols:
            raise ValueError(f"X 는 {self.space.n_cols}컬럼이어야 함: {Xq.shape}")
        parts = [self._predict_chunk(Xq[i:i + _CHUNK])
                 for i in range(0, len(Xq), _CHUNK)]
        return Prediction(*(np.concatenate([p[k] for p in parts], axis=0)
                            for k in range(5)))

    def _predict_chunk(self, Xq: np.ndarray) -> tuple:
        nq = len(Xq)
        Uq = self.space.to_unit(Xq)
        mu, sd_ens = self._model_predict(Uq, self._models)

        sigma = np.empty((nq, self.n_obj))
        status = np.empty((nq, self.n_obj), dtype=object)
        d_min = np.empty((nq, self.n_obj))
        ham = np.empty((nq, self.n_obj))

        # 전 컬럼 일치 여부 — exact 판정용 (그룹과 무관하게 한 번만)
        full_eq = (Xq[:, None, :] == self.X[None, :, :]).all(axis=2)   # (nq, n)

        # 같은 그룹을 쓰는 목적끼리 거리 계산을 공유한다
        for g in (1, 2):
            js = [j for j in range(self.n_obj) if self.scorer.groups[j] == g]
            if not js:
                continue
            cols = self._group_cols[g]
            d = self._dist(Uq, cols)                       # (nq, n)
            h = self._hamming(Xq, cols)
            blk_eq = h == 0.0                              # 의존 컬럼 전부 일치
            dm = d.min(axis=1)
            infl = self._inflate(dm)

            for j in js:
                d_min[:, j] = dm
                ham[:, j] = h.min(axis=1)
                sigma[:, j] = np.sqrt(
                    sd_ens[:, j] ** 2 + self.noise[j] ** 2
                    + (self.alpha * self.y_scale[j] * infl) ** 2)
                status[:, j] = np.where(dm <= self.d_ref, "near", "far")

            # 근거가 있는 점은 모델을 덮어쓴다 — 여기가 "정확히 재생" 의 본체
            for i in np.flatnonzero(blk_eq.any(axis=1)):
                rows = np.flatnonzero(blk_eq[i])
                exact = full_eq[i, rows].all()
                for j in js:
                    mu[i, j] = self.Y[rows, j].mean()
                    sigma[i, j] = self._replay_sigma(rows, j)
                    status[i, j] = "exact" if exact else "exact_block"
                    d_min[i, j] = 0.0
                    ham[i, j] = 0.0

        return mu, sigma, status, d_min, ham

    def _replay_sigma(self, rows: np.ndarray, j: int) -> float:
        """관측을 재생할 때의 σ. **0 을 내지 않는다.**

        σ=0 은 "이 점은 완벽히 안다" 는 주장이다. TRUE 표면에 노이즈가 있는 한
        그건 거짓이고, acquisition 은 그 점을 무한히 신뢰해 버린다(EI 가 그
        방향으로 붕괴한다). 그래서 근거 세 가지 중 가장 큰 것을 쓴다:

          · 전역 노이즈 바닥 (반복측정 pooled) — 있으면 이게 정답이다
          · 이 자리에서 겹친 관측들의 산포 — 근거가 2개 이상이면 국소 추정치다
          · `_replay_floor` — 위 둘이 다 없을 때의 마지노선. 노이즈를 못 쟀다면
            모델의 LOO 오차보다 정확하다고 주장할 근거가 없다

        마지막으로 근거 개수로 나눈다 (평균의 표준오차).
        """
        var = self.noise[j] ** 2
        if len(rows) >= 2:
            var = max(var, float(self.Y[rows, j].var(ddof=1)))
        var = max(var, float(self._replay_floor[j]) ** 2)
        return float(np.sqrt(var / len(rows)))

    def _compute_replay_floor(self) -> np.ndarray:
        """반복측정이 없을 때 재생 σ 의 마지노선 (목적별).

        임의의 상수를 박지 않고 데이터에서 뽑는다 — LOO 절대오차의 평균을
        정규분포의 std 로 환산한 값(×1.2533)이다. "모델이 실제로 내는 오차보다
        더 정확하다고 주장하지 않는다" 는 뜻. LOO 조차 없으면 관측 산포를 쓴다
        (아무것도 모른다는 뜻이므로 가장 보수적인 값).
        """
        if self.noise_dof > 0:
            return np.zeros(self.n_obj)      # 진짜 노이즈 추정이 있으면 불필요
        lo = self._loo()
        if not lo.get("n"):
            return self.y_scale.copy()
        floor = self.y_scale.copy()
        for j in range(self.n_obj):
            m = lo["obj"] == j
            if m.any():
                floor[j] = float(np.abs(lo["resid"][m]).mean()) * np.sqrt(np.pi / 2)
        return floor

    # ─── 캘리브레이션 ──────────────────────────────────────────────────────

    def _loo(self) -> dict:
        """관측을 하나씩 빼고 예측한다 — σ 가 맞는 크기인지 재는 유일한 근거.

        빠진 점과 **같은 x 인 반복측정도 함께 뺀다**. 안 그러면 그 점이
        exact 로 되살아나 오차가 0 이 되고, 캘리브레이션이 무의미해진다.
        """
        if self._loo_cache is not None:
            return self._loo_cache
        key = [self.X[i].tobytes() for i in range(self.n)]
        groups: dict[bytes, list[int]] = {}
        for i, k in enumerate(key):
            groups.setdefault(k, []).append(i)

        resid, sd_raw, infl, dist, tgt = [], [], [], [], []
        for k, rows in groups.items():
            keep = np.array([i for i in range(self.n) if key[i] != k])
            if len(keep) < 2:
                continue
            sub = Surface.__new__(Surface)
            sub.__dict__.update(self.__dict__)
            sub.X, sub.U, sub.Y = self.X[keep], self.U[keep], self.Y[keep]
            sub.n = len(keep)
            sub._models = sub._fit_models(np.arange(len(keep)), self.loo_ensemble)
            Uq = self.U[rows]
            mu, sd = sub._model_predict(Uq, sub._models)
            for g in (1, 2):
                js = [j for j in range(self.n_obj) if self.scorer.groups[j] == g]
                if not js:
                    continue
                d = sub._dist(Uq, sub._group_cols[g]).min(axis=1)
                f = sub._inflate(d)
                for j in js:
                    resid.append(self.Y[rows, j] - mu[:, j])
                    sd_raw.append(np.full(len(rows), 0.0) + sd[:, j])
                    infl.append(f)
                    dist.append(d)
                    tgt.append(np.full(len(rows), j))
        if not resid:
            self._loo_cache = {"n": 0}
            return self._loo_cache
        self._loo_cache = {
            "n": int(sum(len(r) for r in resid)),
            "resid": np.concatenate(resid),
            "sd_ens": np.concatenate(sd_raw),
            "infl": np.concatenate(infl),
            "d": np.concatenate(dist),
            "obj": np.concatenate(tgt).astype(int),
        }
        return self._loo_cache

    def _z(self, alpha: float, lo: dict) -> np.ndarray:
        var = (lo["sd_ens"] ** 2 + self.noise[lo["obj"]] ** 2
               + (alpha * self.y_scale[lo["obj"]] * lo["infl"]) ** 2)
        return lo["resid"] / np.sqrt(np.maximum(var, 1e-24))

    def _calibrate(self) -> float:
        """std(z) = 1 이 되는 α 를 찾는다 (단조 감소라 이분법으로 충분)."""
        lo = self._loo()
        if not lo.get("n"):
            return 1.0
        f = lambda a: float(np.std(self._z(a, lo)))   # noqa: E731
        a_lo, a_hi = 1e-3, 1e3
        if f(a_hi) > 1.0:      # 아무리 키워도 σ 가 모자란다 (외삽이 심하게 틀림)
            return a_hi
        if f(a_lo) < 1.0:      # α 없이도 이미 σ 가 충분하다
            return a_lo
        for _ in range(60):
            mid = np.sqrt(a_lo * a_hi)
            if f(mid) > 1.0:
                a_lo = mid
            else:
                a_hi = mid
        return float(np.sqrt(a_lo * a_hi))

    # ─── 보고 ──────────────────────────────────────────────────────────────

    def report(self, n_probe: int = 2000, seed: int = 0) -> dict:
        """표면이 믿을 만한지 한 장으로. 경고가 있으면 `warnings` 에 담긴다."""
        rng = np.random.default_rng(seed)
        p = self.predict(self.space.sample(rng, n_probe))
        counts: dict[str, int] = {}
        for s in p.status.ravel():
            counts[s] = counts.get(s, 0) + 1

        rep = {
            "n_obs": self.n, "n_unique_x": self.n_unique_x,
            "n_repeat_groups": self.n_repeat_groups,
            "objectives": list(self.scorer.names),
            "noise_floor": dict(zip(self.scorer.names, self.noise.round(6))),
            "noise_rel": dict(zip(self.scorer.names,
                                  (self.noise / self.y_scale).round(4))),
            "noise_dof": self.noise_dof,
            "replay_floor": dict(zip(self.scorer.names,
                                     np.round(self._replay_floor, 6))),
            "noise_ci_x": [round(v, 2) for v in self.noise_ci()],
            "d_ref": round(self.d_ref, 5),
            "alpha": round(self.alpha, 4),
            "coverage": {k: round(v / p.status.size, 4) for k, v in counts.items()},
        }

        lo = self._loo()
        warn = []
        if lo.get("n"):
            z = self._z(self.alpha, lo)
            rep["calibration"] = {
                "loo_n": lo["n"], "z_std": round(float(z.std()), 3),
                "z_mean": round(float(z.mean()), 3),
                "within_1sigma": round(float((np.abs(z) <= 1).mean()), 3),
                "within_2sigma": round(float((np.abs(z) <= 2).mean()), 3),
            }
            # 목적별 LOO 오차 — 노이즈 바닥과 견줘야 의미가 있다
            mae = {}
            for j, nm in enumerate(self.scorer.names):
                m = lo["obj"] == j
                if m.any():
                    mae[nm] = round(float(np.abs(lo["resid"][m]).mean()), 5)
            rep["loo_mae"] = mae
            rep["loo_mae_over_noise"] = {
                nm: (round(mae[nm] / self.noise[j], 2) if self.noise[j] > 0 else None)
                for j, nm in enumerate(self.scorer.names) if nm in mae}
            if abs(z.std() - 1.0) > 0.35:
                warn.append(f"캘리브레이션 이탈 — std(z)={z.std():.2f} (1.0 이어야 함). "
                            "σ 를 acquisition 에 그대로 쓰기 전에 확인할 것")

            # 거리대별 캘리브레이션 — α 는 전역 상수 하나라, 가까운 데서 맞춘 게
            # 먼 데서도 맞는다는 보장이 없다. 여기서 그 가정을 실제로 확인한다.
            bands = {"d≤d_ref": lo["d"] <= self.d_ref,
                     "d≤3·d_ref": (lo["d"] > self.d_ref) & (lo["d"] <= 3 * self.d_ref),
                     "d>3·d_ref": lo["d"] > 3 * self.d_ref}
            by_d = {k: {"n": int(m.sum()), "z_std": round(float(z[m].std()), 3)}
                    for k, m in bands.items() if m.sum() >= 5}
            rep["calibration_by_distance"] = by_d
            worst = [k for k, v in by_d.items() if abs(v["z_std"] - 1.0) > 0.5]
            if worst:
                warn.append(
                    f"거리대 {worst} 에서 σ 가 안 맞는다 — α 는 전역 상수 하나라 "
                    "가까운 데서 맞춘 값이 먼 데서도 맞지는 않는다")

            # α 는 전역 상수 하나다. LOO 가 **optimizer 가 실제로 질의할 거리**를
            # 덮었을 때만 그 값을 믿을 수 있다. 두 거리 분포를 직접 견준다 —
            # 겹치면 검증된 것이고, 질의 쪽이 훨씬 멀면 외삽이다.
            d_loo = np.percentile(lo["d"], [50, 90])
            d_qry = np.percentile(p.d_min, [50, 90])
            rep["d_loo_p50_p90"] = [round(float(v), 4) for v in d_loo]
            rep["d_query_p50_p90"] = [round(float(v), 4) for v in d_qry]
            rep["calibration_covers_query"] = bool(d_qry[0] <= d_loo[1])
            if not rep["calibration_covers_query"]:
                warn.append(
                    f"캘리브레이션 외삽 — 질의 거리 중앙값 {d_qry[0]:.3f} 이 LOO "
                    f"표본의 p90 {d_loo[1]:.3f} 보다 멀다. α 를 그 영역에서 검증한 "
                    "적이 없으니 σ 를 믿지 말고 alpha 를 직접 올려 잡을 것")

            # 커버리지 밖 σ 가 관측 산포 대비 얼마인지는 **경고가 아니라 정보**다.
            # 작다고 해서 틀린 게 아니다 — 모델이 주효과를 잡았다면 아무것도
            # 모르는 것보다 나은 게 맞고, 위의 거리대별 z_std 가 그 근거다.
            rep["sigma_far_over_yscale"] = round(float(np.mean(
                np.sqrt(self.noise ** 2 + (self.alpha * self.y_scale) ** 2)
                / self.y_scale)), 3)
        else:
            warn.append("LOO 불가 — 관측이 너무 적어 σ 크기를 검증하지 못했다")

        if self.n_repeat_groups == 0:
            warn.append(
                "반복측정 없음 — 노이즈 바닥을 못 쟀다. 재생 σ 는 LOO 오차에서 뽑은 "
                f"마지노선({np.round(self._replay_floor / self.y_scale, 3).tolist()} "
                "× 관측 산포)으로 대신하고 있다. 같은 X 를 두어 번 재면 이 자리가 "
                "실측으로 바뀌고 표면 전체가 정직해진다")
        elif self.noise_ci()[1] > 1.5:
            lo_x, hi_x = self.noise_ci()
            warn.append(
                f"노이즈 바닥이 성기게 추정됨 (dof {self.noise_dof}) — 참값이 "
                f"추정치의 {lo_x:.2f}~{hi_x:.2f}배 사이다. 상한을 1.25배로 좁히려면 "
                f"dof 가 {_dof_for(1.25)} 는 되어야 한다 (반복측정은 한 X 에 몰지 "
                "않고 서로 다른 X 에 2회씩 나눠 걸어도 dof 가 쌓인다)")
        far = rep["coverage"].get("far", 0.0)
        if far > 0.95:
            warn.append(f"무작위 X 의 {far:.0%} 가 커버리지 밖 — 정상이다(10^15 에 "
                        f"{self.n}점). optimizer 는 값이 아니라 σ 로 움직여야 한다")
        rep["warnings"] = warn
        return rep

    def explain(self, x: np.ndarray, k: int = 3) -> str:
        """이 X 에 대한 답의 근거를 사람이 읽는 형태로."""
        x = np.asarray(x, dtype=np.int64).reshape(1, -1)
        p = self.predict(x)
        lines = [f"X = {x[0].tolist()}"]
        for j, nm in enumerate(self.scorer.names):
            lines.append(
                f"  {nm:<8} μ={p.mu[0, j]:>10.4g}  σ={p.sigma[0, j]:>9.4g}  "
                f"{p.status[0, j]:<11} d={p.d_min[0, j]:.4f} "
                f"(d_ref {self.d_ref:.4f}) 해밍={p.hamming[0, j]:.3f}")
        for g in (1, 2):
            cols = self._group_cols[g]
            d = self._dist(self.space.to_unit(x), cols)[0]
            order = np.argsort(d, kind="stable")[:k]
            lines.append(f"  ── group{g} 근거 관측 (의존 {len(cols)}컬럼 기준)")
            for rank, i in enumerate(order):
                ys = "  ".join(
                    f"{n}={self.Y[i, jj]:.4g}"
                    for jj, n in enumerate(self.scorer.names)
                    if self.scorer.groups[jj] == g)
                lines.append(f"     #{rank} d={d[i]:.4f} idx={i}  {ys}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    from sim import Simulator, design

    rng = np.random.default_rng(0)
    sim = Simulator(seed=303, noise_level=0.03)
    ss = sim.space

    recs = design(sim, rng, n_random=40, n_repeat=8)
    t0 = time.perf_counter()
    surf = Surface.fit(recs)
    t_fit = time.perf_counter() - t0
    print(f"[fit] 관측 {surf.n}점 · 목적 {surf.n_obj}개 · 앙상블 {surf.n_ensemble} "
          f"— {t_fit:.1f}s (캘리브레이션 LOO 포함)")

    # ─ 요구 1: 데이터가 있는 곳은 정확히 재생하는가 ─
    Xobs = np.array([r.x for r in recs])
    p = surf.predict(Xobs)
    assert (np.isin(p.status, ("exact", "exact_block"))).all(), "관측점이 exact 가 아님"
    err = np.abs(p.mu - surf.Y)
    # 근거가 **하나뿐인** 칸은 실측과 바이트 그대로 같아야 한다.
    # 근거가 여럿인 칸(같은 x 반복측정 / 그 목적의 의존 컬럼이 같은 다른 x)은
    # 평균이라 개별 실측과 노이즈만큼 다르다 — 그게 옳은 동작이다.
    n_src = np.empty((surf.n, surf.n_obj), dtype=int)
    for j in range(surf.n_obj):
        cols = surf.obj_cols[j]
        n_src[:, j] = (surf.X[:, None, cols] == surf.X[None, :, cols]).all(2).sum(1)
    solo = n_src == 1
    assert np.allclose(p.mu[solo], surf.Y[solo]), "근거가 하나인 칸이 정확 재생 안 됨"
    multi = ~solo
    tol = np.maximum(np.tile(surf.noise * 4, (surf.n, 1)), 1e-9)
    assert (err[multi] <= tol[multi]).all(), \
        f"평균 재생 오차가 노이즈 4σ 초과: {err[multi].max():.4g}"
    print(f"[OK] 요구1 정확 재생 — 근거 1개인 {int(solo.sum())}칸은 실측과 완전 일치, "
          f"근거 여럿인 {int(multi.sum())}칸은 평균 (오차 ≤ 노이즈 4σ)")

    # 설계가 준 공짜 반복: 기준점 + set2 one-hot 점들은 group1 의존 컬럼이 같다
    # → group1 입장에서 16회 반복측정이고, 평균이 개별 실측보다 정확해야 한다
    x_blk = Xobs[0]
    rows = np.flatnonzero((surf.X[:, surf.obj_cols[0]] == x_blk[surf.obj_cols[0]]).all(1))
    if len(rows) > 2:
        truth0 = score.get()([sim.record(x_blk)])[0, 0]      # 노이즈 없는 참값
        e_avg = abs(surf.predict(x_blk[None, :]).mu[0, 0] - truth0)
        e_one = float(np.abs(surf.Y[rows, 0] - truth0).mean())
        assert e_avg < e_one, "평균이 개별 실측보다 나쁘다"
        print(f"[OK] 블록 평균 — 의존 컬럼이 같은 관측 {len(rows)}개를 평균해 "
              f"참값 오차 {e_one:.3g} → {e_avg:.3g} ({e_one / max(e_avg, 1e-12):.1f}배 개선)")

    # ─ 요구 2: 관측에서 멀면 "모름" 이라고 하는가 ─
    Xfar = ss.sample(rng, 3000)
    pf = surf.predict(Xfar)
    far_rate = float((pf.status == "far").mean())
    ratio = float((pf.sigma[pf.status == "far"]
                   / np.tile(surf.y_scale, (len(Xfar), 1))[pf.status == "far"]).mean())
    assert far_rate > 0.8, f"무작위 X 의 far 비율이 낮음 {far_rate:.2f}"
    assert ratio > 0.4, f"far 인데 σ 가 관측 산포 대비 작음 ({ratio:.2f})"
    print(f"[OK] 요구2 정직한 거부 — 무작위 X 의 {far_rate:.0%} 가 far, "
          f"그때 σ 가 관측 산포의 {ratio:.0%}")

    # σ 가 거리에 따라 실제로 커지는가.
    # 무작위 X 로는 못 잰다 — 전부 far 라 팽창항이 포화돼 σ 가 평평하다.
    # 관측점에서 k개 컬럼씩 떼어놓으며 봐야 한다.
    x_seed = Xobs[-1]                              # 반복측정된 점 (근거가 확실)
    g1_cols = surf.obj_cols[0]
    prof, ds = [], []
    for k in (0, 1, 2, 4, 8, len(g1_cols)):
        Xm = np.tile(x_seed, (300, 1))
        for t in range(300):
            if k:
                cc = rng.choice(g1_cols, size=k, replace=False)
                Xm[t, cc] = rng.integers(ss.x_min[cc], ss.x_max[cc] + 1)
        pm = surf.predict(Xm)
        prof.append(float(pm.sigma[:, 0].mean()))
        ds.append(float(pm.d_min[:, 0].mean()))
    assert prof == sorted(prof), f"의존 컬럼을 뗄수록 σ 가 커지지 않음: {prof}"
    print("[OK] σ 단조성 — group1 의존 컬럼을 0→1→2→4→8→15개 흔들 때 평균 σ "
          + " < ".join(f"{v:.3g}" for v in prof))
    print("     (그때 평균 거리 " + " → ".join(f"{v:.3f}" for v in ds)
          + f", d_ref={surf.d_ref:.3f})")

    # ─ 노이즈 바닥이 시뮬레이터의 실제 노이즈와 맞는가 ─
    x_rep = recs[-1].x
    truth = score.get()([sim.record(x_rep, rng) for _ in range(200)])
    print("[검증] 노이즈 바닥 (반복 8회 추정 vs 실제 200회):")
    for j, nm in enumerate(surf.scorer.names):
        print(f"    {nm:<8} 추정 {surf.noise[j]:>9.4g}   실제 {truth[:, j].std():>9.4g}")

    # ─ 캘리브레이션 — σ 가 맞는 크기인가 (제대로 된 surrogate 의 핵심) ─
    rep = surf.report()
    cal = rep["calibration"]
    print(f"[OK] 캘리브레이션 — α={rep['alpha']}, std(z)={cal['z_std']} (1.0 목표), "
          f"1σ 안 {cal['within_1sigma']:.0%} (68% 목표), "
          f"2σ 안 {cal['within_2sigma']:.0%} (95% 목표)")
    assert abs(cal["z_std"] - 1.0) < 0.35, f"σ 크기가 안 맞음: std(z)={cal['z_std']}"

    # 캘리브레이션을 끄면 실제로 나빠지는가 (α 가 일을 하고 있다는 증거)
    raw = Surface.fit(recs, alpha=1.0)
    z_raw = raw._z(1.0, raw._loo())
    print(f"[대조] α 를 1.0 으로 고정하면 std(z)={z_raw.std():.2f} "
          f"→ 캘리브레이션이 {abs(z_raw.std() - 1):.2f} 만큼의 오차를 교정했다")

    # ─ 블록 구조를 실제로 쓰는가: group2 컬럼만 바꾸면 group1 예측이 그대로여야 ─
    xa = ss.sample(rng, 1)[0]
    xb = xa.copy()
    s2 = ss.block_cols("set2")
    xb[s2] = ss.x_max[s2]
    pa, pb = surf.predict(xa), surf.predict(xb)
    g1 = [j for j in range(surf.n_obj) if surf.scorer.groups[j] == 1]
    assert np.allclose(pa.mu[0, g1], pb.mu[0, g1]), "set2 를 바꿨는데 group1 예측이 변함"
    print("[OK] 블록 제한 — set2 를 전부 바꿔도 group1 목적의 μ·σ 가 불변")

    # ─ 그룹별 커버리지가 갈리는가 (목적별 판정의 존재 이유) ─
    # 무작위 X 로는 안 갈린다 — 둘 다 far 다. set2 만 흔들면 group1 은 근거가
    # 그대로고 group2 만 근거를 잃는다. 이 비대칭이 목적별 판정을 두는 이유다.
    x_base = Xobs[3]
    Xs2 = np.tile(x_base, (200, 1))
    s2c = ss.block_cols("set2")
    for t in range(200):
        Xs2[t, s2c] = rng.integers(ss.x_min[s2c], ss.x_max[s2c] + 1)
    ps = surf.predict(Xs2)
    j1 = [j for j in range(surf.n_obj) if surf.scorer.groups[j] == 1][0]
    j2 = [j for j in range(surf.n_obj) if surf.scorer.groups[j] == 2][0]
    known1 = float(ps.known[:, j1].mean())
    known2 = float(ps.known[:, j2].mean())
    assert known1 > 0.9 and known2 < 0.1, f"판정이 그룹별로 안 갈림: {known1}, {known2}"
    print(f"[OK] 목적별 판정 — set2 만 흔든 200점에서 group1 은 {known1:.0%} 근거 있음, "
          f"group2 는 {known2:.0%} (같은 X 인데 목적마다 아는 정도가 다르다)")
    print(f"     그때 σ 비 — {surf.scorer.names[j1]} {ps.sigma[:, j1].mean():.3g} vs "
          f"{surf.scorer.names[j2]} {ps.sigma[:, j2].mean():.3g}")

    # ─ 관측이 늘면 그 근방이 실제로 알려지는가 ─
    x_new = Xfar[int(np.argmax(pf.d_min[:, 0]))]      # 가장 먼 점
    before = surf.predict(x_new)
    surf2 = Surface.fit(recs + sim.records(x_new[None, :], rng), calibrate=False)
    after = surf2.predict(x_new)
    assert before.status[0, 0] == "far" and after.status[0, 0] == "exact"
    assert after.sigma[0, 0] < before.sigma[0, 0]
    print(f"[OK] 갱신 — 가장 먼 점을 측정하니 far→exact, "
          f"σ {before.sigma[0, 0]:.3g} → {after.sigma[0, 0]:.3g}")

    # ─ 성능: acquisition 이 후보 수만 개를 넣는다 ─
    Xbig = ss.sample(rng, 20000)
    t0 = time.perf_counter()
    surf.predict(Xbig)
    dt = time.perf_counter() - t0
    print(f"[성능] 후보 20,000점 예측 {dt:.2f}s ({20000 / dt:,.0f} pts/s)")

    print("\n" + surf.explain(Xobs[3]))
    if rep["warnings"]:
        print("\n[경고]")
        for w in rep["warnings"]:
            print("  ·", w)
