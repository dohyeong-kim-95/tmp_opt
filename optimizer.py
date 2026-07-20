"""optimizer.py — ask-tell 구조의 stateless optimizer 모음 + 공유 score 파이프라인.

──────────────────────────────────────────────────────────────────────────────
설계 원칙
──────────────────────────────────────────────────────────────────────────────
1. **Stateless**: optimizer 인스턴스는 설정(탐색 공간, 하이퍼파라미터)만 갖고
   탐색 상태는 전혀 갖지 않는다. 상태(히스토리 포함)는 순수 dict 이며,
   pickle 로 파일 직렬화가 가능하다 (체크포인트/재개 지원).

2. **ask-tell 사이클** (runner 는 calculator 와 optimizer 를 반복 호출하는
   기계일 뿐이다):
       state = opt.init_state(seed)
       loop:
           X_batch, state = opt.ask(state)            # 후보 1 batch 제안
           ... runner 가 X_batch 를 순차 평가 ...
           state = opt.tell(state, X_batch, Y_raw)    # 증분 raw 관측 통보

3. **히스토리·점수는 optimizer 소유**: tell 은 이번 batch 의 (X, raw y0) 만
   받고, 베이스 클래스가 state 안에 전체 히스토리를 누적한다. 값 범위를
   사전에 모르므로 매 tell 전체 raw 히스토리로 robust quantile 스케일러를
   다시 적합하고 전 관측을 재점수한다 — 과거 관측의 점수도 매번 바뀐다.
   알고리즘 갱신 훅 `_update` 는 최신 (X_hist, scores_hist) 를 받으며,
   자기 구성원(개체/현재해 등)을 히스토리 인덱스로 기억해 두고 매번
   최신 점수를 다시 조회한다.

4. **점수 방향**: scores 는 클수록 좋다. 방향(sense) 통일은 이 파일의
   RobustScaler 한 곳에서만 일어난다 (calculator 의 OBJECTIVE_SENSES 소비).

구현된 알고리즘:
    - RandomSearchOptimizer      : 균등 랜덤 (baseline)
    - BlockwiseCoordinateOptimizer: 블록 순환 coordinate descent
    - GAOptimizer                : 유전 알고리즘 (블록 단위 crossover)
    - SAOptimizer                : Simulated Annealing
    - PSOOptimizer               : 이산화된 Particle Swarm
    - ACOOptimizer               : Ant Colony (컬럼×레벨 페로몬)
    - TPEOptimizer               : Tree-structured Parzen Estimator (직접 구현)
    - XGBSurrogateOptimizer      : XGBoost surrogate + novelty acquisition
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np

from calculator import OBJECTIVE_SENSES
from space import SearchSpace

# ──────────────────────────────────────────────────────────────────────────────
# 공통 유틸: RNG 상태의 저장/복원 (stateless 를 유지하면서 재현성 보장)
# ──────────────────────────────────────────────────────────────────────────────

def _rng_load(state: dict) -> np.random.Generator:
    """state dict 에 저장된 비트제너레이터 상태로 RNG 를 복원한다."""
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng"]
    return rng


def _rng_save(state: dict, rng: np.random.Generator) -> None:
    """RNG 의 현재 상태를 state dict 에 기록한다 (pickle 가능한 순수 dict)."""
    state["rng"] = rng.bit_generator.state


# ──────────────────────────────────────────────────────────────────────────────
# 파일 교환 셸 — optimizer 가 소유하는 프로세스 간 교환 형식.
# 주의: 이 계층은 진입부(셸) 전용이다. OptimizerBase 와 알고리즘 클래스의
# ask/tell 은 파일의 존재를 모르는 순수 함수로 유지한다.
#
# [x.txt — 나가는 쪽, 우리가 형식을 소유]
#     # eval_index=123
#     [15,0,0,0,-1,-3,5,3]
#     [-15,0,3,0,-1,3,2,9]
# - 1행 헤더 = 이 배치 첫 평가의 전역 카운터 (노이즈 시딩·대응 검증용)
# - 2행부터 한 줄 = 해 하나 (signed 정수, 텍스트 왕복 무손실)
#
# [y_raw.bin — 들어오는 쪽, 내부 구조를 우리가 통제하지 못하는 불투명 바이너리]
# 실제 문제의 bin 레이아웃은 생산자(calculator) 소관이라 달라질 수 있다.
# 대응 지점을 두 함수로 고정한다 — 레이아웃이 바뀌면 **이 둘만 교체**하고,
# 하류(tell 이후 파이프라인)는 불변:
#     read_y_raw(path)   : bin 디코딩 → (원형 배열, eval_index)
#     convert_y_raw(Yf)  : 원형 → 표준 (b, K) float64  ← y_raw→y 변환 이음새
# 레퍼런스 기본 레이아웃: int64 eval_index, int64 b, int64 K, float64×(b·K) (LE).
#
# 규율: 원자적 쓰기(tmp + os.replace), fail-loud(형식/범위/NaN 위반 즉시 raise).
# ──────────────────────────────────────────────────────────────────────────────

def write_x(path: str | Path, X: np.ndarray, eval_index: int) -> None:
    """후보 배치 X 를 x.txt 형식으로 원자적으로 쓴다.

    Args:
        X          : (b, n_cols) 정수 배열 (signed 값)
        eval_index : 이 배치의 첫 평가가 갖는 전역 평가 카운터 (0-base)
    """
    X = np.asarray(X)
    assert X.ndim == 2 and len(X) >= 1, f"X 는 (b, n_cols) 2차원이어야 함: {X.shape}"
    assert np.issubdtype(X.dtype, np.integer), f"X 는 정수여야 함: {X.dtype}"
    path = Path(path)

    lines = [f"# eval_index={int(eval_index)}"]
    lines += ["[" + ",".join(str(int(v)) for v in row) + "]" for row in X]

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)  # 원자적 교체 — 독자는 완전한 파일만 본다


def read_x(path: str | Path, space: SearchSpace | None = None) -> tuple[np.ndarray, int]:
    """x.txt 를 읽어 (X, eval_index) 를 돌려준다. 형식 위반은 즉시 raise.

    Args:
        space: 주어지면 각 값이 [x_min, x_max] 안인지까지 검증한다.
    """
    text = Path(path).read_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("# eval_index="):
        raise ValueError(f"{path}: 1행은 '# eval_index=<int>' 헤더여야 함")
    eval_index = int(lines[0].removeprefix("# eval_index="))

    rows = []
    for i, ln in enumerate(lines[1:], start=2):
        if not (ln.startswith("[") and ln.endswith("]")):
            raise ValueError(f"{path}:{i}: 행은 [v,v,...] 형식이어야 함: {ln!r}")
        rows.append([int(tok) for tok in ln[1:-1].split(",")])  # 정수 아님 → 즉시 raise
    if not rows:
        raise ValueError(f"{path}: 해가 한 줄도 없음")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{path}: 행 길이 불일치: {sorted(widths)}")

    X = np.asarray(rows, dtype=np.int64)
    if space is not None:
        if X.shape[1] != space.n_cols:
            raise ValueError(f"{path}: n_cols {X.shape[1]} ≠ 명세 {space.n_cols}")
        bad = (X < space.x_min) | (X > space.x_max)
        if bad.any():
            r, c = map(int, np.argwhere(bad)[0])
            raise ValueError(
                f"{path}: 값 범위 위반 — 행 {r} 컬럼 {c}: {X[r, c]} ∉ "
                f"[{space.x_min[c]}, {space.x_max[c]}]"
            )
    return X, eval_index


def write_y_raw(path: str | Path, y_raw: dict, eval_index: int) -> None:
    """레퍼런스 y_raw.bin 작성기 (calculator 측 구조화 레이아웃).

    레이아웃 (전부 little-endian):
        int64 × 4 : eval_index, b, G(마스크 한 변), n_scalar(=2)
        uint8 × (b·G·G) : mask1        uint8 × (b·G·G) : mask2
        float64 × b : y13              float64 × b : y23
    실제 시스템의 calculator 가 다른 레이아웃을 쓰면 이 함수가 아니라
    read_y_raw/convert_y_raw 쪽을 교체한다.
    """
    m1 = np.ascontiguousarray(y_raw["mask1"], dtype=np.uint8)
    m2 = np.ascontiguousarray(y_raw["mask2"], dtype=np.uint8)
    y13 = np.ascontiguousarray(y_raw["y13"], dtype="<f8")
    y23 = np.ascontiguousarray(y_raw["y23"], dtype="<f8")
    b, g, _ = m1.shape
    assert m1.shape == m2.shape == (b, g, g) and y13.shape == y23.shape == (b,)
    path = Path(path)
    header = np.array([int(eval_index), b, g, 2], dtype="<i8")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(header.tobytes() + m1.tobytes() + m2.tobytes()
                    + y13.tobytes() + y23.tobytes())
    os.replace(tmp, path)  # 원자적 교체


def read_y_raw(path: str | Path) -> tuple[dict, int]:
    """y_raw.bin 디코딩 → (구조화 y_raw dict, eval_index). 위반 즉시 raise.

    ⚠️ 교체 지점 ①: 실제 문제의 bin 내부 구조가 다르면 이 함수를 그 레이아웃에
    맞게 갈아끼운다. 반환 계약(convert_y_raw 가 받는 형태)만 지키면 하류 불변.
    """
    buf = Path(path).read_bytes()
    if len(buf) < 32:
        raise ValueError(f"{path}: 헤더(32바이트)보다 짧음 — {len(buf)}바이트")
    eval_index, b, g, n_scalar = (int(v) for v in np.frombuffer(buf[:32], dtype="<i8"))
    if b < 1 or g < 1 or n_scalar != 2:
        raise ValueError(f"{path}: 헤더 손상 — b={b}, G={g}, n_scalar={n_scalar}")
    mask_n = b * g * g
    expected = 32 + 2 * mask_n + 8 * 2 * b
    if len(buf) != expected:
        raise ValueError(f"{path}: 크기 불일치 — {len(buf)} ≠ {expected} (b={b}, G={g})")
    off = 32
    m1 = np.frombuffer(buf[off:off + mask_n], dtype=np.uint8).reshape(b, g, g)
    off += mask_n
    m2 = np.frombuffer(buf[off:off + mask_n], dtype=np.uint8).reshape(b, g, g)
    off += mask_n
    y13 = np.frombuffer(buf[off:off + 8 * b], dtype="<f8")
    y23 = np.frombuffer(buf[off + 8 * b:], dtype="<f8")
    return {"mask1": m1.astype(bool), "mask2": m2.astype(bool),
            "y13": y13, "y23": y23}, eval_index


# ─── 체크포인트: history.jsonl (관측의 진실) + state.pkl (내부 상태) ──────────
#
# history.jsonl — append-only, 한 줄 = tell 한 번:
#     {"eval_index":0,"X":[[15,0,...]],"y_raw":[[5686.2,...]]}
#   X 는 정수, y_raw 는 Python json 의 shortest-round-trip repr 로 float64
#   무손실. 사람이 읽고 diff 할 수 있으며, pkl 없이도 post-hoc 분석이 가능하다.
# state.pkl — 히스토리를 제외한 나머지: 알고리즘 상태 + RNG + 스케일러
#   파라미터 + 점수 캐시(_s_buf). 점수는 스케일러 *이력* 에 의존하는 파생
#   상태라(rescore_interval>1 이면 재계산 불가) 관측이 아니라 상태로 취급한다.
# 정합성: load_state 가 pkl 의 n_evals 와 jsonl 의 누적 평가 수를 대조해
#   어긋나면 즉시 raise. jsonl 이 진실이므로, pkl 이 깨지면 히스토리를
#   처음부터 tell 로 재생(replay)해 상태를 재구성할 수도 있다.
# ──────────────────────────────────────────────────────────────────────────────

#: state.pkl 에서 제외되는 키 — 히스토리 버퍼(jsonl 이 원천)와 파생 뷰
_HISTORY_STATE_KEYS = ("_X_buf", "_Y_buf", "_s_buf",
                       "X_hist", "Y_raw_hist", "scores_hist")


def append_history(path: str | Path, X_new: np.ndarray, Y_new: np.ndarray,
                   eval_index: int) -> None:
    """tell 한 번 분량의 (X, y_raw) 관측을 history.jsonl 에 한 줄 append 한다.

    eval_index 는 이 batch 첫 평가의 전역 카운터 — load 시 연속성 검증에 쓴다.
    """
    X_new = np.atleast_2d(np.asarray(X_new))
    Y_new = np.atleast_2d(np.asarray(Y_new))
    assert len(X_new) == len(Y_new) >= 1
    line = json.dumps(
        {"eval_index": int(eval_index),
         "X": X_new.astype(np.int64).tolist(),
         "y_raw": Y_new.astype(np.float64).tolist()},
        separators=(",", ":"),
    )
    with open(path, "a") as f:
        f.write(line + "\n")


def load_history(path: str | Path,
                 space: SearchSpace | None = None) -> tuple[np.ndarray, np.ndarray]:
    """history.jsonl 전체를 읽어 (X (n, n_cols), Y_raw (n, K)) 로 잇는다.

    fail-loud: JSON 손상, eval_index 불연속(빠졌거나 중복된 batch), 행 폭
    불일치, (space 를 주면) 값 범위 위반 — 전부 즉시 raise. 크래시로 마지막
    줄이 잘렸다면 그 줄을 수동으로 지운 뒤 다시 로드한다(자동 절단은 안 한다).
    """
    Xs, Ys = [], []
    n = 0
    with open(path) as f:
        for lineno, ln in enumerate(f, start=1):
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: JSON 손상 — {e}") from e
            if rec["eval_index"] != n:
                raise ValueError(
                    f"{path}:{lineno}: eval_index 불연속 — {rec['eval_index']} ≠ 누적 {n}")
            X = np.asarray(rec["X"], dtype=np.int64)
            Y = np.asarray(rec["y_raw"], dtype=np.float64)
            if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
                raise ValueError(f"{path}:{lineno}: X/y_raw 형상 불일치 — {X.shape}/{Y.shape}")
            Xs.append(X)
            Ys.append(Y)
            n += len(X)
    if not Xs:
        raise ValueError(f"{path}: 관측이 한 줄도 없음")
    X_all = np.vstack(Xs)
    Y_all = np.vstack(Ys)
    if space is not None:
        if X_all.shape[1] != space.n_cols:
            raise ValueError(f"{path}: n_cols {X_all.shape[1]} ≠ 명세 {space.n_cols}")
        if ((X_all < space.x_min) | (X_all > space.x_max)).any():
            raise ValueError(f"{path}: 값 범위 위반 행 존재")
    return X_all, Y_all


def save_state(path: str | Path, state: dict) -> None:
    """히스토리를 제외한 optimizer 상태를 state.pkl 로 원자적으로 저장한다.

    점수 캐시(_s_buf)는 채워진 구간만 잘라 포함한다(파생 상태 — §체크포인트 노트).
    """
    path = Path(path)
    slim = {k: v for k, v in state.items() if k not in _HISTORY_STATE_KEYS}
    slim["_scores"] = np.array(state["_s_buf"][: state["n_evals"]])
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(slim, f)
    os.replace(tmp, path)  # 원자적 교체


def load_state(state_path: str | Path, history_path: str | Path,
               space: SearchSpace | None = None) -> dict:
    """state.pkl + history.jsonl 에서 완전한 state dict 를 재구성한다.

    정합성 fail-loud: pkl 의 n_evals ≠ jsonl 누적 평가 수 → 즉시 raise.
    반환된 state 는 무중단 실행의 그 시점 state 와 동일 거동(동일 궤적 재개).
    """
    with open(state_path, "rb") as f:
        slim = pickle.load(f)
    X, Y = load_history(history_path, space=space)
    n = slim["n_evals"]
    if len(X) != n:
        raise ValueError(
            f"체크포인트 정합성 위반 — state.pkl n_evals={n} ≠ history.jsonl {len(X)}")
    scores = slim.pop("_scores")
    if len(scores) != n:
        raise ValueError(f"점수 캐시 길이 {len(scores)} ≠ n_evals {n}")

    state = dict(slim)
    state["_X_buf"] = np.ascontiguousarray(X, dtype=np.int64)
    state["_Y_buf"] = np.ascontiguousarray(Y, dtype=np.float64)
    state["_s_buf"] = np.ascontiguousarray(scores, dtype=np.float64)
    OptimizerBase._sync_views(state)
    return state


def _mask_extents(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """boolean 마스크 (b, G, G) → (max height, max width) 픽셀 측정.

    max height = 컬럼별 True 개수의 최대(가장 긴 세로 현),
    max width  = 행별 True 개수의 최대(가장 긴 가로 현).
    개수 기반이라 경계 flip 노이즈에 ±수 픽셀 수준으로만 흔들린다.
    """
    height = mask.sum(axis=1).max(axis=1)  # (b, G) 컬럼 카운트 → 최대
    width = mask.sum(axis=2).max(axis=1)   # (b, G) 행 카운트 → 최대
    return height.astype(np.float64), width.astype(np.float64)


def convert_y_raw(Y_raw, n_obj: int | None = None) -> np.ndarray:
    """y_raw(관측 원형) → 표준 (b, K) float64 — **y_raw→y 변환 이음새**.

    ⚠️ 교체 지점 ②: 실제 문제에서 관측 형태가 다르면 이 함수만 수정한다.
    현재 관측 형태 = 구조화 dict:
        mask1 (b,G,G) bool → y11 = max height, y12 = max width
        mask2 (b,G,G) bool → y21 = max height, y22 = max width
        y13, y23 (b,)      → 그대로 통과
    출력 계약: (b, K) float64, 열 순서 = OBJECTIVE_NAMES, 전 원소 유한.
    NaN/inf 는 즉시 raise — 조용한 대체 금지.
    (하위호환: 이미 (b, K) 수치 배열이면 검증만 하고 통과 — 합성 테스트용)
    """
    if isinstance(Y_raw, dict):
        for key in ("mask1", "mask2", "y13", "y23"):
            if key not in Y_raw:
                raise ValueError(f"y_raw dict 에 {key!r} 없음 — keys={list(Y_raw)}")
        m1 = np.asarray(Y_raw["mask1"], dtype=bool)
        m2 = np.asarray(Y_raw["mask2"], dtype=bool)
        if m1.ndim != 3 or m1.shape != m2.shape:
            raise ValueError(f"마스크 형상 불일치 — {m1.shape} vs {m2.shape}")
        h1, w1 = _mask_extents(m1)
        h2, w2 = _mask_extents(m2)
        y13 = np.asarray(Y_raw["y13"], dtype=np.float64).reshape(-1)
        y23 = np.asarray(Y_raw["y23"], dtype=np.float64).reshape(-1)
        if not (len(h1) == len(y13) == len(y23)):
            raise ValueError(
                f"batch 크기 불일치 — mask {len(h1)}, y13 {len(y13)}, y23 {len(y23)}")
        Y = np.column_stack([h1, w1, y13, h2, w2, y23])  # OBJECTIVE_NAMES 순서
    else:
        Y = np.atleast_2d(np.asarray(Y_raw, dtype=np.float64))

    if Y.ndim != 2:
        raise ValueError(f"y_raw 는 2차원이어야 함: {Y.shape}")
    if n_obj is not None and Y.shape[1] != n_obj:
        raise ValueError(f"목적 수 불일치 — {Y.shape[1]} ≠ 기대 {n_obj}")
    if not np.isfinite(Y).all():
        r, c = map(int, np.argwhere(~np.isfinite(Y))[0])
        raise ValueError(f"y_raw 에 비유한값 — 행 {r} 목적 {c}: {Y[r, c]}")
    return Y


# ──────────────────────────────────────────────────────────────────────────────
# 공유 score 파이프라인 — raw y0 → 정규화 z → 스칼라 점수
# 탐색 구동(OptimizerBase.tell)과 리포트(benchmark.py 의 pooled 재점수)가
# 같은 구현을 공유해야 랭킹이 유효하다. sense(max/min 방향) 적용은 시스템
# 전체에서 RobustScaler.transform 한 곳뿐이다.
# ──────────────────────────────────────────────────────────────────────────────

class RobustScaler:
    """raw y0 (n, 6) → 정규화 z (n, 6), 모든 목적을 '1 = best' 방향으로 통일.

    - 노이즈/outlier 에 강하도록 min-max 대신 p5–p95 quantile 을 쓰고,
      범위 밖 값은 [0, 1] 로 클리핑한다.
    - 최소화 목적(y13, y23)은 뒤집어서 z 가 클수록 좋게 만든다.
    - 관측이 적거나 값이 퇴화(상수)한 목적은 z=0.5 로 중립 처리한다.
    """

    def __init__(self, q_low: float = 0.05, q_high: float = 0.95):
        self.q_low = q_low
        self.q_high = q_high
        self.lo: np.ndarray | None = None  # (6,)
        self.hi: np.ndarray | None = None

    def fit(self, Y_raw: np.ndarray) -> "RobustScaler":
        Y_raw = np.atleast_2d(Y_raw)
        self.lo = np.quantile(Y_raw, self.q_low, axis=0)
        self.hi = np.quantile(Y_raw, self.q_high, axis=0)
        return self

    def transform(self, Y_raw: np.ndarray) -> np.ndarray:
        assert self.lo is not None, "transform 전에 fit 필요"
        Y_raw = np.atleast_2d(Y_raw)
        span = self.hi - self.lo
        z = np.empty_like(Y_raw, dtype=np.float64)
        for j in range(Y_raw.shape[1]):
            if span[j] < 1e-15:  # 퇴화: 아직 정보 없음 → 중립값
                z[:, j] = 0.5
                continue
            zj = (Y_raw[:, j] - self.lo[j]) / span[j]
            if OBJECTIVE_SENSES[j] < 0:  # 최소화 목적은 뒤집는다
                zj = 1.0 - zj
            z[:, j] = np.clip(zj, 0.0, 1.0)
        return z


def score_sum(z: np.ndarray) -> np.ndarray:
    """단순 평균 (정규화 합과 순서 동일). baseline — 한 목적 폭락을 못 막는다."""
    return z.mean(axis=1)


def score_chebyshev(z: np.ndarray, rho: float = 0.01) -> np.ndarray:
    """augmented Chebyshev (ideal = 1). 최악 목적이 점수를 지배한다.

    고전형은 max_j(1 − z_j) + ρ·Σ(1 − z_j) 를 최소화하는 것 —
    여기서는 '클수록 좋음' 방향으로 등가 변환해 [0, 1] 범위로 맞춘다:
        score = (min_j z_j + ρ·mean_j z_j) / (1 + ρ)
    ρ 항은 '최악이 같은 해'들 사이의 순위를 나머지 목적으로 갈라주는 보정.
    """
    return (z.min(axis=1) + rho * z.mean(axis=1)) / (1.0 + rho)


def score_owa_bottom_k(z: np.ndarray, k: int = 2) -> np.ndarray:
    """bottom-k OWA: 가장 나쁜 k개 목적의 평균. Chebyshev(=k1)보다 완만한 안전장치."""
    return np.sort(z, axis=1)[:, :k].mean(axis=1)


SCORERS = {
    "sum": score_sum,
    "chebyshev": score_chebyshev,
    "owa": score_owa_bottom_k,
}


# ──────────────────────────────────────────────────────────────────────────────
# 베이스 클래스
# ──────────────────────────────────────────────────────────────────────────────

class OptimizerBase:
    """모든 optimizer 의 공통 인터페이스.

    서브클래스가 구현할 것:
        init_state(seed) -> dict               (super() 호출 후 자기 키 추가)
        ask(state)       -> (X_batch, state)
        _update(state, X_hist, scores_hist) -> state   # 알고리즘 갱신 훅

    tell 은 베이스가 구현한다 — 전 알고리즘 공통 ingest + 점수화:
        1. 이번 batch 의 (X, raw y0) 증분을 state 내부 히스토리에 누적
        2. 전체 raw 히스토리로 RobustScaler 재적합 → 전 관측 재점수
           (값 범위 미지 가정. rescore_interval > 1 이면 주기적으로만 재적합)
        3. self._update(state, X_hist, scores_hist) 호출

    RandomSearch 처럼 점수를 안 쓰는 알고리즘도 ingest 는 베이스에서 항상
    일어난다 — 히스토리는 체크포인트/사후(anytime) 분석의 유일한 소스다.

    주의: ask/tell 은 state 를 **수정해서 반환**한다(반환값 사용 필수).
    인스턴스 속성에는 절대 탐색 상태를 저장하지 않는다.
    """

    name: str = "base"

    def __init__(
        self,
        space: SearchSpace,
        total_budget: int = 800,
        scorer_name: str = "chebyshev",
        rescore_interval: int = 1,
    ) -> None:
        self.space = space
        # 일부 알고리즘(SA 온도 스케줄 등)이 진행률 계산에 예산을 사용한다.
        self.total_budget = total_budget
        self.scorer_name = scorer_name
        self.scorer = SCORERS[scorer_name]
        # 1(기본) = 매 tell 전체 재적합·재점수. 장기 실행(true-optimum 100K 등)
        # 은 크게 잡아 O(N²) 비용을 피한다 — 그 사이 새 관측만 기존 스케일러로
        # 점수화한다 (단위: 평가 횟수).
        self.rescore_interval = rescore_interval

    def init_state(self, seed: int) -> dict:
        """탐색 상태를 초기화한다. RNG 상태와 빈 히스토리 버퍼를 포함한다."""
        state: dict = {}
        _rng_save(state, np.random.default_rng(seed))
        n_obj = len(OBJECTIVE_SENSES)
        cap = 64  # 시작 용량 — 부족하면 tell 에서 2배씩 확장 (amortized O(N))
        state["_X_buf"] = np.empty((cap, self.space.n_cols), dtype=np.int64)
        state["_Y_buf"] = np.empty((cap, n_obj), dtype=np.float64)
        state["_s_buf"] = np.empty(cap, dtype=np.float64)
        state["n_evals"] = 0
        state["_since_refit"] = 0
        state["_lo"] = None  # 마지막 재적합 시점의 스케일러 파라미터
        state["_hi"] = None
        self._sync_views(state)
        return state

    @staticmethod
    def _sync_views(state: dict) -> None:
        """버퍼의 채워진 구간을 가리키는 공개 뷰를 갱신한다 (복사 없음)."""
        n = state["n_evals"]
        state["X_hist"] = state["_X_buf"][:n]
        state["Y_raw_hist"] = state["_Y_buf"][:n]
        state["scores_hist"] = state["_s_buf"][:n]

    def ask(self, state: dict) -> tuple[np.ndarray, dict]:
        """다음에 평가할 후보 X 들을 (batch, 30) 정수 배열로 반환한다."""
        raise NotImplementedError

    def tell(self, state: dict, X_new: np.ndarray, Y_raw_new: np.ndarray) -> dict:
        """이번 batch 의 (X, raw y0) 증분을 통보받는다 — ingest 는 공통.

        히스토리 누적 → 스케일러 재적합 → 전 관측 재점수 → `_update` 호출.
        """
        X_new = np.atleast_2d(np.asarray(X_new, dtype=np.int64))
        # y_raw 원형(구조화 dict 또는 수치 배열) → 표준 (b, K) — 변환 이음새 경유
        Y_new = convert_y_raw(Y_raw_new, n_obj=len(OBJECTIVE_SENSES))
        assert len(X_new) == len(Y_new) and len(X_new) >= 1
        b = len(X_new)
        n0 = state["n_evals"]
        n = n0 + b

        if n > len(state["_X_buf"]):  # 용량 2배 확장
            new_cap = max(2 * len(state["_X_buf"]), n)
            for key in ("_X_buf", "_Y_buf", "_s_buf"):
                buf = state[key]
                grown = np.empty((new_cap,) + buf.shape[1:], dtype=buf.dtype)
                grown[:n0] = buf[:n0]
                state[key] = grown
        state["_X_buf"][n0:n] = X_new
        state["_Y_buf"][n0:n] = Y_new
        state["n_evals"] = n

        # 값 범위를 모르므로 기본은 매 tell 전체 재적합·재점수 — 과거 관측의
        # 점수도 매번 바뀐다 (알고리즘은 인덱스로 기억하고 최신 점수 재조회).
        state["_since_refit"] += b
        scaler = RobustScaler()
        if state["_lo"] is None or state["_since_refit"] >= self.rescore_interval:
            scaler.fit(state["_Y_buf"][:n])
            state["_lo"], state["_hi"] = scaler.lo, scaler.hi
            state["_s_buf"][:n] = self.scorer(scaler.transform(state["_Y_buf"][:n]))
            state["_since_refit"] = 0
        else:  # 장기 실행 경로: 새 관측만 기존 스케일러로 점수화
            scaler.lo, scaler.hi = state["_lo"], state["_hi"]
            state["_s_buf"][n0:n] = self.scorer(scaler.transform(Y_new))

        self._sync_views(state)
        return self._update(state, state["X_hist"], state["scores_hist"])

    def _update(self, state: dict, X_hist: np.ndarray, scores_hist: np.ndarray) -> dict:
        """알고리즘별 상태 갱신 훅. 최신 재정규화 점수의 전체 히스토리를 받는다."""
        raise NotImplementedError

    # ─── 서브클래스 공용 헬퍼 ──────────────────────────────────────────────

    def _random_batch(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.space.sample(rng, n)

    def _mutate(
        self, rng: np.random.Generator, x: np.ndarray, rate: float = 1.0 / 30
    ) -> np.ndarray:
        """ordinal 변수용 mutation: 대체로 ±1 스텝, 가끔 랜덤 값 점프.

        각 컬럼이 확률 rate 로 변이된다. 최소 1개 컬럼은 반드시 변이시켜
        '아무 변화 없는 자식'이 나오는 것을 막는다.
        signed 범위: 점프는 [x_min, x_max] 균등 — [0, card) 산술 금지.
        """
        x = x.copy()
        mask = rng.random(self.space.n_cols) < rate
        if not mask.any():
            mask[rng.integers(self.space.n_cols)] = True
        for c in np.flatnonzero(mask):
            if rng.random() < 0.8:  # ordinal 구조 활용: 이웃 값으로 이동
                x[c] += rng.choice([-1, 1])
            else:  # 탈출용 랜덤 점프
                x[c] = rng.integers(self.space.x_min[c], self.space.x_max[c] + 1)
        return self.space.clip(x)


# ──────────────────────────────────────────────────────────────────────────────
# 1) Random Search — 모든 비교의 baseline
# ──────────────────────────────────────────────────────────────────────────────

class RandomSearchOptimizer(OptimizerBase):
    """균등 랜덤 샘플링. 다른 method 가 이걸 못 이기면 문제가 있는 것."""

    name = "random"

    def __init__(self, space: SearchSpace, total_budget: int = 800,
                 batch_size: int = 10, **base_kwargs):
        super().__init__(space, total_budget, **base_kwargs)
        self.batch_size = batch_size

    def ask(self, state: dict) -> tuple[np.ndarray, dict]:
        rng = _rng_load(state)
        batch = self._random_batch(rng, self.batch_size)
        _rng_save(state, rng)
        return batch, state

    def _update(self, state: dict, X_hist: np.ndarray, scores_hist: np.ndarray) -> dict:
        return state  # 점수를 사용하지 않는다 (ingest 는 베이스 tell 이 이미 수행)


# ──────────────────────────────────────────────────────────────────────────────
# 2) Blockwise Coordinate Selection
# ──────────────────────────────────────────────────────────────────────────────

class BlockwiseCoordinateOptimizer(OptimizerBase):
    """블록-인지 좌표 local search (random-restart hill climbing).

    설계:
      - **초기점**: marginal-balanced 설계(컬럼별 레벨이 균등하게 나오는 설계)
        n_init 개를 관측하고, 그중 best 를 incumbent 로 삼는다.
      - **스윕**: 라운드마다 block_order(기본 common → set2 → set1)를 따라
        각 변수를 1-hop(ordinal ±1) 스윕하며 변수별 best-improvement 를
        채택한다. common 을 매 라운드 재방문해 블록 간 결합을 흡수한다.
      - **재시작**: 라운드 내 개선이 없으면 수렴으로 보고, marginal-balanced
        새 점(지금까지 restart 에 덜 쓰인 레벨 우선)으로 random-restart 하여
        남은 예산을 다른 basin 탐색에 쓴다.
      - **캐시**: 같은 X 의 재평가는 캐시로 회피해 예산을 아낀다.
        노이즈 관측 점수로 탐색하며, 참 점수의 anytime 평가는 calculator 가
        그대로 담당한다 (optimizer 는 관측 점수만 사용).
    """

    name = "blockwise_coord"

    def __init__(
        self,
        space: SearchSpace,
        total_budget: int = 800,
        n_init: int = 32,
        block_order: tuple[str, ...] = ("common", "set2", "set1"),
        **base_kwargs,
    ):
        super().__init__(space, total_budget, **base_kwargs)
        self.n_init = n_init
        # 라운드 내 변수 방문 순서 (블록 순서 → 블록 내 컬럼 순서)
        self._var_order = np.concatenate([space.block_cols(b) for b in block_order])

    # ─── marginal-balanced 설계 ────────────────────────────────────────────

    def _balanced_design(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """컬럼별로 모든 값이 최대한 균등하게 등장하는 n 개 설계점."""
        X = np.empty((n, self.space.n_cols), dtype=np.int64)
        for c, card in enumerate(self.space.cardinalities):
            # signed 범위: 값 = x_min + 레벨 (0 기준 산술 금지)
            levels = np.arange(self.space.x_min[c], self.space.x_max[c] + 1)
            reps = np.tile(levels, int(np.ceil(n / card)))[:n]
            rng.shuffle(reps)
            X[:, c] = reps
        return X

    def _restart_point(self, state: dict, rng: np.random.Generator) -> np.ndarray:
        """restart 이력에서 덜 쓰인 값을 우선 선택해 marginal 균형을 유지."""
        usage = state["restart_usage"]  # (30, max_card) — 무효 슬롯은 +inf
        x = np.empty(self.space.n_cols, dtype=np.int64)
        for c in range(self.space.n_cols):
            u = usage[c, : self.space.cardinalities[c]]
            cand = np.flatnonzero(u == u.min())
            lvl = rng.choice(cand)  # 최소 사용 슬롯 중 랜덤 (탐색 다양성)
            x[c] = self.space.x_min[c] + lvl  # 슬롯 인덱스 → signed 값
            usage[c, lvl] += 1
        return x

    # ─── 상태 관리 ─────────────────────────────────────────────────────────

    def init_state(self, seed: int) -> dict:
        state = super().init_state(seed)
        state["phase"] = "init"        # "init" → "sweep"
        state["cache"] = {}            # X bytes → 히스토리 인덱스 (재평가 회피)
        state["incumbent_idx"] = -1    # incumbent 의 히스토리 인덱스
        state["incumbent_x"] = None
        state["cursor"] = 0            # _var_order 상의 현재 위치
        state["round_improved"] = False
        state["need_restart"] = False
        state["pending"] = None        # ("init"|"sweep"|"restart", 스윕 변수)
        state["pending_start"] = 0
        # restart 점의 marginal 균형 유지용 레벨 사용 횟수
        max_card = int(self.space.cardinalities.max())
        usage = np.zeros((self.space.n_cols, max_card))
        invalid = np.arange(max_card)[None, :] >= self.space.cardinalities[:, None]
        usage[invalid] = np.inf  # 무효 레벨은 절대 선택되지 않게
        state["restart_usage"] = usage
        return state

    def _advance_cursor(self, state: dict) -> None:
        """다음 변수로 이동. 라운드가 끝나면 개선 여부로 restart 를 판단."""
        state["cursor"] += 1
        if state["cursor"] >= len(self._var_order):
            state["cursor"] = 0
            if not state["round_improved"]:  # 라운드 내 무개선 = 수렴
                state["need_restart"] = True
            state["round_improved"] = False

    # ─── ask / tell ────────────────────────────────────────────────────────

    def ask(self, state: dict) -> tuple[np.ndarray, dict]:
        rng = _rng_load(state)
        if state["phase"] == "init":
            batch = self._balanced_design(rng, self.n_init)
            state["pending"] = ("init", -1)
        else:
            batch = None
            # 최대 한 라운드 분량을 훑으며 '캐시에 없는 1-hop 이웃'을 찾는다
            for _ in range(len(self._var_order) + 1):
                if state["need_restart"]:
                    break
                c = int(self._var_order[state["cursor"]])
                x0 = state["incumbent_x"]
                cands = []
                for delta in (-1, +1):  # 1-hop: ordinal 이웃 값
                    v = x0[c] + delta
                    if self.space.x_min[c] <= v <= self.space.x_max[c]:
                        x = x0.copy()
                        x[c] = v
                        if x.tobytes() not in state["cache"]:
                            cands.append(x)
                if cands:
                    batch = np.array(cands)
                    state["pending"] = ("sweep", c)
                    break
                # 이 변수의 이웃이 전부 기관측 → 평가 없이 다음 변수로
                self._advance_cursor(state)

            if batch is None:  # 수렴(또는 전부 캐시) → marginal-balanced restart
                for _ in range(10):  # 캐시 충돌 시 몇 번 재추첨
                    x = self._restart_point(state, rng)
                    if x.tobytes() not in state["cache"]:
                        break
                batch = x[None, :]
                state["pending"] = ("restart", -1)
                state["need_restart"] = False
        _rng_save(state, rng)
        return batch, state

    def _update(self, state: dict, X_hist: np.ndarray, scores_hist: np.ndarray) -> dict:
        n = len(scores_hist)
        new_idx = np.arange(state["pending_start"], n)
        for i in new_idx:  # 재평가 회피 캐시 갱신
            state["cache"][X_hist[i].tobytes()] = int(i)

        kind, swept_var = state["pending"]
        if kind == "init":
            # 설계점 중 관측 best 를 incumbent 로, 스윕 시작
            best = new_idx[int(np.argmax(scores_hist[new_idx]))]
            state["incumbent_idx"] = int(best)
            state["incumbent_x"] = X_hist[best].copy()
            state["phase"] = "sweep"
        elif kind == "restart":
            # 새 basin 탐색: 점수와 무관하게 restart 점을 incumbent 로 삼는다
            # (전역 best 는 runner 의 best-so-far 가 이미 보존한다)
            i = int(new_idx[-1])
            state["incumbent_idx"] = i
            state["incumbent_x"] = X_hist[i].copy()
            state["cursor"] = 0
            state["round_improved"] = False
        else:  # "sweep" — 변수별 best-improvement 채택
            best = new_idx[int(np.argmax(scores_hist[new_idx]))]
            # incumbent 점수도 재정규화된 최신 값으로 다시 조회해 비교한다
            if scores_hist[best] > scores_hist[state["incumbent_idx"]]:
                state["incumbent_idx"] = int(best)
                state["incumbent_x"] = X_hist[best].copy()
                state["round_improved"] = True
            self._advance_cursor(state)

        state["pending_start"] = n
        return state


# ──────────────────────────────────────────────────────────────────────────────
# 3) Genetic Algorithm
# ──────────────────────────────────────────────────────────────────────────────

class GAOptimizer(OptimizerBase):
    """(μ+λ) 유전 알고리즘. 블록 단위 crossover 로 조건부 독립 구조를 활용.

    - population/자식 크기 λ = pop_size (ask 1회 = 1세대)
    - 선택: 토너먼트(k=3), crossover: 블록 단위 donor 선택 + 컬럼 uniform 혼합
    - tell: (부모 ∪ 자식) 에서 최신 점수 상위 μ 를 다음 세대 부모로.
      부모는 히스토리 인덱스로 기억한다 (재정규화 대응).
    """

    name = "ga"

    def __init__(
        self,
        space: SearchSpace,
        total_budget: int = 800,
        pop_size: int = 20,
        mutation_rate: float = 2.0 / 30,
        **base_kwargs,
    ):
        super().__init__(space, total_budget, **base_kwargs)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

    def init_state(self, seed: int) -> dict:
        state = super().init_state(seed)
        state["parent_idx"] = None  # 부모 개체들의 히스토리 인덱스 (init 전 None)
        state["pending_start"] = 0  # 이번 batch 가 히스토리에서 시작될 인덱스
        return state

    def _crossover(
        self, rng: np.random.Generator, pa: np.ndarray, pb: np.ndarray
    ) -> np.ndarray:
        """블록 단위 crossover: 블록마다 donor 부모를 고르고, 그 위에
        낮은 확률의 컬럼 단위 uniform 혼합을 얹는다."""
        child = pa.copy()
        for name in self.space.blocks:
            if rng.random() < 0.5:
                cols = self.space.block_cols(name)
                child[cols] = pb[cols]
        swap = rng.random(self.space.n_cols) < 0.1  # 미세 혼합
        child[swap] = pb[swap]
        return child

    def ask(self, state: dict) -> tuple[np.ndarray, dict]:
        rng = _rng_load(state)
        if state["parent_idx"] is None:
            batch = self._random_batch(rng, self.pop_size)  # 초기 세대는 랜덤
        else:
            scores = state["parent_scores"]  # 직전 tell 에서 조회해 둔 부모 점수
            parents_X = state["parent_X"]
            batch = np.empty((self.pop_size, self.space.n_cols), dtype=np.int64)
            for i in range(self.pop_size):
                # 토너먼트 선택 (k=3) 2회 → 부모 쌍
                def pick() -> int:
                    cand = rng.integers(0, len(scores), 3)
                    return int(cand[np.argmax(scores[cand])])
                pa, pb = parents_X[pick()], parents_X[pick()]
                child = self._crossover(rng, pa, pb)
                batch[i] = self._mutate(rng, child, self.mutation_rate)
        _rng_save(state, rng)
        return batch, state

    def _update(self, state: dict, X_hist: np.ndarray, scores_hist: np.ndarray) -> dict:
        n = len(scores_hist)
        new_idx = np.arange(state["pending_start"], n)  # 이번 batch 의 인덱스
        if state["parent_idx"] is None:
            pool = new_idx
        else:
            pool = np.concatenate([state["parent_idx"], new_idx])
        # (μ+λ) 환경 선택: 최신 재정규화 점수 기준 상위 pop_size 생존
        order = pool[np.argsort(scores_hist[pool])[::-1]]
        survivors = order[: self.pop_size]
        state["parent_idx"] = survivors
        # ask 에서 다시 조회하지 않도록 X/점수 스냅샷을 함께 저장
        state["parent_X"] = X_hist[survivors].copy()
        state["parent_scores"] = scores_hist[survivors].copy()
        state["pending_start"] = n
        return state


# ──────────────────────────────────────────────────────────────────────────────
# 4) Simulated Annealing
# ──────────────────────────────────────────────────────────────────────────────

class SAOptimizer(OptimizerBase):
    """Simulated Annealing. batch 크기 1 (한 번에 이웃 하나 제안).

    - 이웃: 1~3개 컬럼을 ordinal ±1 스텝(가끔 랜덤 점프)으로 변경
    - 온도: 지수 감쇠 T = T0·(T_end/T0)^(evals/budget)
      점수가 [0,1] 로 정규화되므로 T0=0.1, T_end=1e-3 이 무난하다.
    - 수락 판정은 '현재 해'와 '제안 해'의 **최신 재정규화 점수**로 한다.
    """

    name = "sa"

    def __init__(
        self,
        space: SearchSpace,
        total_budget: int = 800,
        t_start: float = 0.1,
        t_end: float = 1e-3,
        **base_kwargs,
    ):
        super().__init__(space, total_budget, **base_kwargs)
        self.t_start = t_start
        self.t_end = t_end

    def init_state(self, seed: int) -> dict:
        state = super().init_state(seed)
        state["current_idx"] = None  # 현재 해의 히스토리 인덱스 (init 전 None)
        state["pending_start"] = 0
        return state

    def _temperature(self, n_evals: int) -> float:
        frac = min(1.0, n_evals / max(1, self.total_budget))
        return self.t_start * (self.t_end / self.t_start) ** frac

    def ask(self, state: dict) -> tuple[np.ndarray, dict]:
        rng = _rng_load(state)
        if state["current_idx"] is None:
            batch = self._random_batch(rng, 1)  # 초기해
        else:
            n_moves = int(rng.integers(1, 4))  # 1~3개 컬럼 변경
            batch = self._mutate(
                rng, state["current_X"], rate=n_moves / self.space.n_cols
            )[None, :]
        _rng_save(state, rng)
        return batch, state

    def _update(self, state: dict, X_hist: np.ndarray, scores_hist: np.ndarray) -> dict:
        rng = _rng_load(state)
        n = len(scores_hist)
        prop_idx = n - 1  # 이번에 평가된 제안 해 (batch=1)
        if state["current_idx"] is None:
            accept = True
        else:
            cur = scores_hist[state["current_idx"]]  # 최신 점수로 재조회
            prop = scores_hist[prop_idx]
            delta = float(prop - cur)
            t = self._temperature(n)
            # 개선이면 무조건, 악화면 Metropolis 확률로 수락
            accept = delta >= 0 or rng.random() < np.exp(delta / max(t, 1e-12))
        if accept:
            state["current_idx"] = prop_idx
            state["current_X"] = X_hist[prop_idx].copy()
        state["pending_start"] = n
        _rng_save(state, rng)
        return state


# ──────────────────────────────────────────────────────────────────────────────
# 5) Particle Swarm Optimization (이산화 버전)
# ──────────────────────────────────────────────────────────────────────────────

class PSOOptimizer(OptimizerBase):
    """연속 완화(continuous relaxation) 기반 이산 PSO.

    - 입자 위치 p ∈ [0,1]^30 를 연속으로 유지하고, 평가 시에만
      레벨 인덱스로 반올림한다 (ordinal 이라 등간격 매핑이 자연스럽다).
    - 표준 속도 갱신: v ← w·v + c1·r1·(pbest−p) + c2·r2·(gbest−p)
    - pbest/gbest 는 히스토리 인덱스로 기억하고 매 tell 에서 최신 점수로 갱신.
    """

    name = "pso"

    def __init__(
        self,
        space: SearchSpace,
        total_budget: int = 800,
        swarm_size: int = 20,
        inertia: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        **base_kwargs,
    ):
        super().__init__(space, total_budget, **base_kwargs)
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2

    def init_state(self, seed: int) -> dict:
        state = super().init_state(seed)
        rng = _rng_load(state)
        state["pos"] = rng.uniform(0, 1, (self.swarm_size, self.space.n_cols))
        state["vel"] = rng.uniform(-0.1, 0.1, (self.swarm_size, self.space.n_cols))
        state["pbest_idx"] = None  # 입자별 personal best 의 히스토리 인덱스
        state["pending_start"] = 0
        _rng_save(state, rng)
        return state

    def _decode(self, pos: np.ndarray) -> np.ndarray:
        """연속 위치 [0,1] → signed 정수 값 (반올림). decode(0) == x_min."""
        x = self.space.x_min + np.rint(
            pos * (self.space.x_max - self.space.x_min)
        ).astype(np.int64)
        return self.space.clip(x)

    def ask(self, state: dict) -> tuple[np.ndarray, dict]:
        rng = _rng_load(state)
        if state["pbest_idx"] is not None:
            # 속도/위치 갱신 (첫 세대는 초기 위치 그대로 평가)
            pos, vel = state["pos"], state["vel"]
            pbest_u = state["pbest_u"]                  # (S, 30) 단위 좌표
            gbest_u = state["gbest_u"][None, :]         # (1, 30)
            r1 = rng.uniform(0, 1, pos.shape)
            r2 = rng.uniform(0, 1, pos.shape)
            vel = (
                self.inertia * vel
                + self.c1 * r1 * (pbest_u - pos)
                + self.c2 * r2 * (gbest_u - pos)
            )
            # 속도 상한: 한 스텝에 공간의 30% 이상 움직이지 않게
            vel = np.clip(vel, -0.3, 0.3)
            pos = np.clip(pos + vel, 0.0, 1.0)
            state["pos"], state["vel"] = pos, vel
        batch = self._decode(state["pos"])
        _rng_save(state, rng)
        return batch, state

    def _update(self, state: dict, X_hist: np.ndarray, scores_hist: np.ndarray) -> dict:
        n = len(scores_hist)
        new_idx = np.arange(state["pending_start"], n)  # 입자 순서와 동일
        if state["pbest_idx"] is None:
            state["pbest_idx"] = new_idx.copy()
        else:
            # 최신 재정규화 점수로 personal best 갱신
            improved = scores_hist[new_idx] > scores_hist[state["pbest_idx"]]
            state["pbest_idx"][improved] = new_idx[improved]
        # personal/global best 의 단위 좌표 스냅샷 (ask 의 속도 갱신에 사용)
        pbest_X = X_hist[state["pbest_idx"]]
        state["pbest_u"] = self.space.to_unit(pbest_X)
        g = state["pbest_idx"][int(np.argmax(scores_hist[state["pbest_idx"]]))]
        state["gbest_u"] = self.space.to_unit(X_hist[g])
        state["pending_start"] = n
        return state


# ──────────────────────────────────────────────────────────────────────────────
# 6) Ant Colony Optimization
# ──────────────────────────────────────────────────────────────────────────────

class ACOOptimizer(OptimizerBase):
    """컬럼×레벨 페로몬 테이블 기반 ACO.

    - 페로몬 τ[c][l]: 컬럼 c 에서 레벨 l 을 고를 상대적 선호도.
      cardinality 가 컬럼마다 다르므로 (30, max_card) 배열 + 마스크로 관리.
    - ask: 개미 n_ants 마리가 τ 비례 확률로 레벨을 독립 샘플링.
    - tell: 증발(1−ρ) 후, 전체 히스토리 상위 elite 해들이 rank 가중으로 침착.
      점수 크기 자체는 쓰지 않고 rank 만 쓰므로 스케일-프리다.
    """

    name = "aco"

    def __init__(
        self,
        space: SearchSpace,
        total_budget: int = 800,
        n_ants: int = 20,
        evaporation: float = 0.1,
        n_elite: int = 5,
        explore_floor: float = 0.05,
        **base_kwargs,
    ):
        super().__init__(space, total_budget, **base_kwargs)
        self.n_ants = n_ants
        self.evaporation = evaporation
        self.n_elite = n_elite
        self.explore_floor = explore_floor  # 확률 하한 (조기 수렴 방지)
        # 유효 레벨 마스크: (30, max_card), 유효하지 않은 레벨은 False
        max_card = int(space.cardinalities.max())
        self._level_mask = (
            np.arange(max_card)[None, :] < space.cardinalities[:, None]
        )

    def init_state(self, seed: int) -> dict:
        state = super().init_state(seed)
        tau = np.where(self._level_mask, 1.0, 0.0)  # 균등 초기 페로몬
        state["tau"] = tau
        return state

    def ask(self, state: dict) -> tuple[np.ndarray, dict]:
        rng = _rng_load(state)
        tau = state["tau"]
        # 컬럼별 선택 확률 = 정규화된 τ 에 탐험 하한을 섞은 것
        probs = tau / tau.sum(axis=1, keepdims=True)
        uniform = self._level_mask / self._level_mask.sum(axis=1, keepdims=True)
        probs = (1 - self.explore_floor) * probs + self.explore_floor * uniform
        batch = np.empty((self.n_ants, self.space.n_cols), dtype=np.int64)
        for c in range(self.space.n_cols):
            # 페로몬 슬롯 인덱스로 샘플링한 뒤 signed 값으로 변환
            lvl = rng.choice(len(probs[c]), size=self.n_ants, p=probs[c])
            batch[:, c] = self.space.x_min[c] + lvl
        _rng_save(state, rng)
        return batch, state

    def _update(self, state: dict, X_hist: np.ndarray, scores_hist: np.ndarray) -> dict:
        tau = state["tau"] * (1 - self.evaporation)  # 증발
        # 전체 히스토리 상위 elite 가 rank 가중(1위가 가장 크게)으로 침착
        elite = np.argsort(scores_hist)[::-1][: self.n_elite]
        for rank, idx in enumerate(elite):
            deposit = self.evaporation * (self.n_elite - rank) / self.n_elite
            lvl = X_hist[idx] - self.space.x_min  # signed 값 → 슬롯 인덱스
            tau[np.arange(self.space.n_cols), lvl] += deposit
        state["tau"] = np.where(self._level_mask, tau, 0.0)
        return state


# ──────────────────────────────────────────────────────────────────────────────
# 7) TPE (Tree-structured Parzen Estimator) — 직접 구현
# ──────────────────────────────────────────────────────────────────────────────

class TPEOptimizer(OptimizerBase):
    """밀도비 기반 SMBO. 히스토리를 good/bad 로 나누고 컬럼별 이산 밀도를 추정.

    - 분할: 점수 상위 γ(기본 20%) 를 good, 나머지를 bad.
    - 밀도: 컬럼별 레벨 히스토그램 + ordinal smoothing(이웃 레벨로 [0.25,0.5,0.25]
      커널 확산) + Laplace 평활. ordinal 구조를 활용하는 부분이 이 smoothing.
    - acquisition: good 분포에서 후보 n_candidates 개를 샘플링한 뒤
      log p_good − log p_bad 가 최대인 것 하나를 제안 (batch=1).
    - 상태는 RNG 뿐 — 모델은 매번 히스토리에서 다시 만든다 (stateless 에 최적).
    """

    name = "tpe"

    def __init__(
        self,
        space: SearchSpace,
        total_budget: int = 800,
        gamma: float = 0.2,
        n_candidates: int = 50,
        n_startup: int = 20,
        **base_kwargs,
    ):
        super().__init__(space, total_budget, **base_kwargs)
        self.gamma = gamma
        self.n_candidates = n_candidates
        self.n_startup = n_startup  # 이 관측 수 전까지는 랜덤 탐색

    def _column_density(self, levels: np.ndarray, card: int) -> np.ndarray:
        """한 컬럼의 관측 슬롯 인덱스(값 − x_min)로부터 smoothing 된 분포를 만든다."""
        counts = np.bincount(levels, minlength=card).astype(np.float64)
        if card >= 3:
            # ordinal smoothing: 이웃 레벨로 질량을 퍼뜨린다 (경계는 반사 없이 잘림)
            smoothed = 0.5 * counts.copy()
            smoothed[1:] += 0.25 * counts[:-1]
            smoothed[:-1] += 0.25 * counts[1:]
            counts = smoothed
        counts += 0.5  # Laplace 평활 (미관측 레벨 확률 0 방지)
        return counts / counts.sum()

    def ask(self, state: dict) -> tuple[np.ndarray, dict]:
        rng = _rng_load(state)
        X, s = state["X_hist"], state["scores_hist"]  # 베이스가 유지하는 최신 뷰
        if len(s) < self.n_startup:
            batch = self._random_batch(rng, 1)  # 시동 구간: 랜덤
            _rng_save(state, rng)
            return batch, state

        # good/bad 분할 (good 은 최소 5개 확보)
        n_good = max(5, int(np.ceil(self.gamma * len(s))))
        order = np.argsort(s)[::-1]
        good, bad = X[order[:n_good]], X[order[n_good:]]

        # 컬럼별 밀도 추정 → good 분포에서 후보 샘플링 → 밀도비 최대 후보 선택
        cands = np.empty((self.n_candidates, self.space.n_cols), dtype=np.int64)
        log_ratio = np.zeros(self.n_candidates)
        for c in range(self.space.n_cols):
            card = int(self.space.cardinalities[c])
            x_min = int(self.space.x_min[c])  # signed 값 ↔ 슬롯 인덱스 오프셋
            p_g = self._column_density(good[:, c] - x_min, card)
            p_b = self._column_density(bad[:, c] - x_min, card)
            lvl = rng.choice(card, size=self.n_candidates, p=p_g)
            cands[:, c] = x_min + lvl
            log_ratio += np.log(p_g[lvl]) - np.log(p_b[lvl])
        batch = cands[int(np.argmax(log_ratio))][None, :]
        _rng_save(state, rng)
        return batch, state

    def _update(self, state: dict, X_hist: np.ndarray, scores_hist: np.ndarray) -> dict:
        return state  # 모형은 매 ask 에서 히스토리로 다시 만든다 — 갱신할 상태 없음


# ──────────────────────────────────────────────────────────────────────────────
# 8) XGBoost Surrogate + Novelty Acquisition
# ──────────────────────────────────────────────────────────────────────────────

class XGBSurrogateOptimizer(OptimizerBase):
    """XGBoost 회귀 surrogate 로 점수를 예측하고, novelty 보너스를 더한
    acquisition 으로 후보를 고르는 SMBO.

    TPE 와의 차이: TPE 는 컬럼별 독립 밀도(주효과)만 보지만, tree 회귀는
    컬럼 간 상호작용(특히 common 블록의 trade-off 결합)을 학습할 수 있다.

    - acquisition = μ̂(x) + κ · novelty(x)
      novelty = 기존 관측점까지의 최소 해밍거리 / 30  (탐험 유도, 이산 공간에서
      예측 분산 추정을 대신하는 실용적 장치)
    - 후보 풀 = 랜덤 n_random + 상위 해 mutation n_local (exploitation)
    - 모델 재학습은 refit_interval tell 마다 (비용 절약). 모델 객체는 state 에
      저장되며 pickle 가능하므로 파일 체크포인트에도 문제없다.
    """

    name = "xgb_surrogate"

    def __init__(
        self,
        space: SearchSpace,
        total_budget: int = 800,
        n_startup: int = 30,
        n_random: int = 150,
        n_local: int = 150,
        kappa: float = 0.15,
        refit_interval: int = 4,
        batch_size: int = 4,
        **base_kwargs,
    ):
        super().__init__(space, total_budget, **base_kwargs)
        self.n_startup = n_startup
        self.n_random = n_random
        self.n_local = n_local
        self.kappa = kappa
        self.refit_interval = refit_interval
        self.batch_size = batch_size

    def init_state(self, seed: int) -> dict:
        state = super().init_state(seed)
        state["model"] = None
        state["tell_count"] = 0
        return state

    def _fit_model(self, X: np.ndarray, s: np.ndarray):
        # import 를 지연시켜 xgboost 미설치 환경에서도 다른 optimizer 는 동작
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=2,
            verbosity=0,
        )
        # ordinal 이므로 레벨 인덱스를 수치 피처로 바로 사용해도 무방
        model.fit(X.astype(np.float32), s.astype(np.float32))
        return model

    def _candidate_pool(
        self, rng: np.random.Generator, X: np.ndarray, s: np.ndarray
    ) -> np.ndarray:
        """랜덤 후보 + 상위 해 주변 mutation 후보를 합친 풀을 만든다."""
        pool = [self._random_batch(rng, self.n_random)]
        top = X[np.argsort(s)[::-1][:10]]
        local = np.array(
            [self._mutate(rng, top[rng.integers(len(top))], rate=2.0 / 30)
             for _ in range(self.n_local)]
        )
        pool.append(local)
        return np.vstack(pool)

    def ask(self, state: dict) -> tuple[np.ndarray, dict]:
        rng = _rng_load(state)
        X, s = state["X_hist"], state["scores_hist"]  # 베이스가 유지하는 최신 뷰
        if len(s) < self.n_startup or state["model"] is None:
            batch = self._random_batch(rng, self.batch_size)  # 시동 구간
            _rng_save(state, rng)
            return batch, state

        cands = self._candidate_pool(rng, X, s)
        mu = state["model"].predict(cands.astype(np.float32))

        # novelty: 각 후보에서 기존 관측점까지의 최소 해밍거리 (0~1 정규화)
        # (N×C 비교라 비용이 있지만 N≤800, C≤300 수준이라 충분히 빠르다)
        diff = (cands[:, None, :] != X[None, :, :]).sum(axis=2)  # (C, N)
        novelty = diff.min(axis=1) / self.space.n_cols

        acq = mu + self.kappa * novelty
        # 중복 제안 방지를 위해 acquisition 상위에서 서로 다른 해를 고른다
        order = np.argsort(acq)[::-1]
        batch, seen = [], set()
        for i in order:
            key = cands[i].tobytes()
            if key not in seen:
                seen.add(key)
                batch.append(cands[i])
            if len(batch) == self.batch_size:
                break
        _rng_save(state, rng)
        return np.array(batch), state

    def _update(self, state: dict, X_hist: np.ndarray, scores_hist: np.ndarray) -> dict:
        state["tell_count"] += 1
        # 재정규화로 과거 점수도 바뀌므로 주기적으로 전체 재학습한다
        need_refit = (
            len(scores_hist) >= self.n_startup
            and (state["model"] is None
                 or state["tell_count"] % self.refit_interval == 0)
        )
        if need_refit:
            state["model"] = self._fit_model(X_hist, scores_hist)
        return state


# ──────────────────────────────────────────────────────────────────────────────
# 9) Chow-Liu 의존성 트리 EDA (MIMIC/COMIT 계열)
# ──────────────────────────────────────────────────────────────────────────────

class ChowLiuTreeEDAOptimizer(OptimizerBase):
    """pairwise 의존성 트리 기반 EDA. 배경/설계 근거는 doc/algo/chow_liu_eda.md.

    한 세대(ask)의 흐름:
      1. 전체 히스토리 상위 γ 를 elite 로 선정 (하한/상한 캡)
      2. elite 로부터 모든 컬럼 쌍의 상호정보량(MI) 추정 (Laplace 평활)
      3. MI 최대 신장 트리(Chow-Liu) 구성 → 조건부 확률표 P(x_v | x_parent)
      4. 루트 marginal 부터 트리 순서로 조건부 샘플링해 batch 생성
         (탐험 하한 ε 만큼 uniform 을 섞어 조기 수렴 방지)

    pool 의 다른 method 와의 관계:
      - ACO/TPE 는 univariate(컬럼 독립) 모형 — 이 클래스는 변수 쌍 결합을
        '샘플링 분포 자체'로 표현하는 유일한 method.
      - BM3 의 교차-블록 상호작용(common↔set)을 직접 겨냥한다.
    상태는 히스토리 스냅샷 + RNG 뿐 — 모형은 매 ask 재구축 (stateless 최적).
    """

    name = "eda_tree"

    def __init__(
        self,
        space: SearchSpace,
        total_budget: int = 800,
        batch_size: int = 20,
        gamma: float = 0.25,
        min_elite: int = 30,
        max_elite: int = 400,   # 100K 스케일에서도 세대당 비용을 상수로 유지
        alpha: float = 0.5,     # Laplace 평활
        explore_floor: float = 0.05,
        n_startup: int = 40,
        **base_kwargs,
    ):
        super().__init__(space, total_budget, **base_kwargs)
        self.batch_size = batch_size
        self.gamma = gamma
        self.min_elite = min_elite
        self.max_elite = max_elite
        self.alpha = alpha
        self.explore_floor = explore_floor
        self.n_startup = n_startup

    # ─── 모형 추정 ─────────────────────────────────────────────────────────

    def _joint_counts(self, elite: np.ndarray, u: int, v: int) -> np.ndarray:
        """컬럼 쌍 (u, v) 의 Laplace 평활된 joint 히스토그램 (ca, cb)."""
        ca = int(self.space.cardinalities[u])
        cb = int(self.space.cardinalities[v])
        lu = elite[:, u] - self.space.x_min[u]  # signed 값 → 슬롯 인덱스
        lv = elite[:, v] - self.space.x_min[v]
        flat = lu * cb + lv  # 2D 인덱스를 1D 로 접어 bincount
        counts = np.bincount(flat, minlength=ca * cb).astype(np.float64)
        return counts.reshape(ca, cb) + self.alpha

    def _mutual_information(self, joint: np.ndarray) -> float:
        """평활된 joint 히스토그램에서 MI 를 추정한다."""
        p = joint / joint.sum()
        pu = p.sum(axis=1, keepdims=True)
        pv = p.sum(axis=0, keepdims=True)
        return float((p * np.log(p / (pu * pv))).sum())

    def _build_tree(self, elite: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
        """Chow-Liu: MI 를 가중치로 한 최대 신장 트리(Prim) → (root, 방향 간선).

        Returns:
            root  : MI 합이 가장 큰 노드 (샘플링 시작점)
            edges : BFS 순서의 (parent, child) 목록 — 이 순서대로 샘플링 가능
        """
        n = self.space.n_cols
        mi = np.zeros((n, n))
        for u in range(n):
            for v in range(u + 1, n):
                mi[u, v] = mi[v, u] = self._mutual_information(
                    self._joint_counts(elite, u, v)
                )

        # Prim 으로 최대 신장 트리 (n=30 이라 O(n²) 로 충분)
        in_tree = np.zeros(n, dtype=bool)
        in_tree[0] = True
        best_w = mi[0].copy()      # 트리 밖 노드가 트리에 붙는 최적 가중치
        best_from = np.zeros(n, dtype=np.int64)
        undirected: list[tuple[int, int]] = []
        for _ in range(n - 1):
            cand = np.where(in_tree, -np.inf, best_w)
            v = int(np.argmax(cand))
            undirected.append((int(best_from[v]), v))
            in_tree[v] = True
            improve = mi[v] > best_w
            best_w[improve] = mi[v][improve]
            best_from[improve] = v

        # 루트 = 트리 간선 MI 합이 가장 큰 노드 (정보 허브에서 샘플링 시작)
        adj: list[list[int]] = [[] for _ in range(n)]
        for u, v in undirected:
            adj[u].append(v)
            adj[v].append(u)
        strength = np.zeros(n)
        for u, v in undirected:
            strength[u] += mi[u, v]
            strength[v] += mi[u, v]
        root = int(np.argmax(strength))

        # BFS 로 방향 부여 → edges 순서대로 조건부 샘플링하면 된다
        edges: list[tuple[int, int]] = []
        visited = {root}
        queue = [root]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    edges.append((u, v))
                    queue.append(v)
        return root, edges

    # ─── ask / tell ────────────────────────────────────────────────────────

    def ask(self, state: dict) -> tuple[np.ndarray, dict]:
        rng = _rng_load(state)
        X, s = state["X_hist"], state["scores_hist"]  # 베이스가 유지하는 최신 뷰
        if len(s) < self.n_startup:
            batch = self._random_batch(rng, self.batch_size)  # 시동 구간
            _rng_save(state, rng)
            return batch, state

        # 1) elite 선정: 상위 γ, [min_elite, max_elite] 로 캡
        n_elite = int(np.clip(int(self.gamma * len(s)),
                              self.min_elite, self.max_elite))
        elite = X[np.argsort(s)[::-1][:n_elite]]

        # 2~3) Chow-Liu 트리 + 조건부 샘플링
        root, edges = self._build_tree(elite)
        batch = np.empty((self.batch_size, self.space.n_cols), dtype=np.int64)

        def _mix(p: np.ndarray) -> np.ndarray:
            """탐험 하한: (1−ε)·모형 + ε·uniform — 확률 포화로 인한 조기 수렴 방지."""
            return (1 - self.explore_floor) * p + self.explore_floor / len(p)

        # 루트: elite marginal 에서 샘플링 (슬롯 인덱스 → signed 값)
        card_r = int(self.space.cardinalities[root])
        p_root = np.bincount(elite[:, root] - self.space.x_min[root],
                             minlength=card_r) + self.alpha
        p_root = _mix(p_root / p_root.sum())
        lvl_r = rng.choice(card_r, size=self.batch_size, p=p_root)
        batch[:, root] = self.space.x_min[root] + lvl_r

        # 자식들: BFS 순서로 P(child | parent) 조건부 샘플링
        for parent, child in edges:
            joint = self._joint_counts(elite, parent, child)  # (cp, cc) 평활됨
            cond = joint / joint.sum(axis=1, keepdims=True)
            for i in range(self.batch_size):
                p = _mix(cond[batch[i, parent] - self.space.x_min[parent]])
                batch[i, child] = self.space.x_min[child] + rng.choice(len(p), p=p)

        _rng_save(state, rng)
        return batch, state

    def _update(self, state: dict, X_hist: np.ndarray, scores_hist: np.ndarray) -> dict:
        return state  # 모형은 매 ask 재구축 (TPE 와 동일) — 갱신할 상태 없음


# ──────────────────────────────────────────────────────────────────────────────
# 10) XGB Trust-Region (CASMOPOLITAN-lite)
# ──────────────────────────────────────────────────────────────────────────────

class XGBTrustRegionOptimizer(OptimizerBase):
    """trust region + 앙상블 불확실성으로 무장한 XGB surrogate (xgb 의 후속).

    xgb_surrogate 의 실측 약점은 init seed 에 따른 고분산이었다 (전역 랜덤
    후보 풀 → 초기 모델이 나쁜 basin 을 가리키면 그대로 표류). 조합공간 BO
    의 trust-region 기법(CASMOPOLITAN/Bounce 계열)을 XGB 에 접목해 고친다:

      - **후보 생성을 국소화**: 전역 랜덤 대신, 현재 trajectory 의 best
        (incumbent) 로부터 해밍 거리 ≤ R 인 이웃만 후보로 생성.
      - **R 의 동적 조절**: 연속 개선(succ_tol)이면 R 2배(최대 r_max),
        연속 정체(fail_tol)면 R 절반. R < 1 로 떨어지면 **restart** —
        새 랜덤 지점에서 trajectory 를 다시 시작 (모델/히스토리는 유지).
        고분산의 원인인 "나쁜 초기 basin 고착"을 restart 가 끊어준다.
      - **앙상블 불확실성**: 서로 다른 시드/부표본의 XGB 4개로 μ, σ 를
        추정하고 UCB(μ + κσ) 로 후보를 고른다 (기존의 novelty 휴리스틱을
        원칙적 acquisition 으로 대체 — SMAC 계열의 교훈).

    모델은 전체 히스토리로 학습(전역 정보 유지)하되, acquisition 은 trust
    region 안에서만 최적화한다(국소 탐색). 상태는 pickle 가능.
    """

    name = "xgb_tr"

    def __init__(
        self,
        space: SearchSpace,
        total_budget: int = 800,
        n_startup: int = 30,
        batch_size: int = 4,
        n_candidates: int = 300,
        kappa: float = 1.0,        # UCB 탐험 계수
        r_init: int = 8,           # trust region 초기 해밍 반경
        r_max: int = 15,
        succ_tol: int = 3,         # 연속 개선 → R 확대
        fail_tol: int = 8,         # 연속 정체(batch 단위) → R 축소
        refit_interval: int = 4,
        n_ensemble: int = 4,
        max_train_size: int = 4000,
        **base_kwargs,
    ):
        super().__init__(space, total_budget, **base_kwargs)
        self.n_startup = n_startup
        self.batch_size = batch_size
        self.n_candidates = n_candidates
        self.kappa = kappa
        self.r_init = r_init
        self.r_max = r_max
        self.succ_tol = succ_tol
        self.fail_tol = fail_tol
        self.refit_interval = refit_interval
        self.n_ensemble = n_ensemble
        # 장기 실행(예: 100K true-optimum) 대응: 학습 표본 상한.
        # 800 evals 수준에서는 상한에 안 걸리므로 동작이 완전히 동일하다.
        self.max_train_size = max_train_size

    def init_state(self, seed: int) -> dict:
        state = super().init_state(seed)
        state["models"] = None       # XGB 앙상블 (pickle 가능)
        state["tell_count"] = 0
        state["seen"] = set()        # 기관측 X (tell 에서 증분 갱신)
        state["radius"] = self.r_init
        state["succ"] = 0
        state["fail"] = 0
        state["restart_start"] = 0   # 현재 trajectory 가 시작된 히스토리 인덱스
        state["reseed"] = False      # True 면 다음 ask 는 랜덤 re-seed batch
        return state

    # ─── surrogate ─────────────────────────────────────────────────────────

    def _fit_ensemble(self, rng: np.random.Generator, X: np.ndarray, s: np.ndarray):
        from xgboost import XGBRegressor

        # 히스토리가 상한을 넘으면 'elite 절반 + 랜덤 절반' 으로 서브샘플 —
        # 좋은 지역의 해상도는 지키면서 전역 형상 정보도 남긴다.
        if len(s) > self.max_train_size:
            half = self.max_train_size // 2
            order = np.argsort(s)[::-1]
            elite = order[:half]
            rest = rng.choice(order[half:], size=half, replace=False)
            keep = np.concatenate([elite, rest])
            X, s = X[keep], s[keep]

        models = []
        for _ in range(self.n_ensemble):
            m = XGBRegressor(
                n_estimators=80, max_depth=4, learning_rate=0.1,
                subsample=0.7, colsample_bytree=0.8,
                random_state=int(rng.integers(2**31)), n_jobs=2, verbosity=0,
            )
            m.fit(X.astype(np.float32), s.astype(np.float32))
            models.append(m)
        return models

    def _ucb(self, models, cands: np.ndarray) -> np.ndarray:
        preds = np.stack([m.predict(cands.astype(np.float32)) for m in models])
        return preds.mean(axis=0) + self.kappa * preds.std(axis=0)

    # ─── trust region 후보 생성 ────────────────────────────────────────────

    def _tr_candidates(
        self, rng: np.random.Generator, incumbent: np.ndarray, radius: int
    ) -> np.ndarray:
        """incumbent 로부터 해밍 거리 1~radius 인 이웃 후보들을 생성한다."""
        cands = np.tile(incumbent, (self.n_candidates, 1))
        for i in range(self.n_candidates):
            d = int(rng.integers(1, radius + 1))
            cols = rng.choice(self.space.n_cols, size=d, replace=False)
            for c in cols:
                lo, hi = int(self.space.x_min[c]), int(self.space.x_max[c])
                if rng.random() < 0.7:  # ordinal 이웃 스텝 위주
                    cands[i, c] = np.clip(
                        cands[i, c] + rng.choice([-1, 1]), lo, hi)
                else:                   # 가끔 값 점프 (deceptive 탈출용)
                    cands[i, c] = rng.integers(lo, hi + 1)
        return cands

    # ─── ask / tell ────────────────────────────────────────────────────────

    def ask(self, state: dict) -> tuple[np.ndarray, dict]:
        rng = _rng_load(state)
        X, s = state["X_hist"], state["scores_hist"]  # 베이스가 유지하는 최신 뷰
        need_random = (
            len(s) < self.n_startup
            or state["models"] is None or state["reseed"]
        )
        if need_random:  # 시동 구간 또는 restart 직후 re-seed
            state["reseed"] = False
            batch = self._random_batch(rng, self.batch_size)
            _rng_save(state, rng)
            return batch, state

        # incumbent = 현재 trajectory(restart 이후) 안의 best
        traj = np.arange(state["restart_start"], len(s))
        incumbent = X[traj[int(np.argmax(s[traj]))]]

        cands = self._tr_candidates(rng, incumbent, state["radius"])
        acq = self._ucb(state["models"], cands)

        # 기평가/중복 제안 제외하고 acquisition 상위 batch_size 개 선택.
        # (기관측 set 은 tell 에서 증분 유지 — 매 ask 재구축하면 장기 실행에서
        #  O(N) × ask 횟수 = O(N²) 이 된다)
        seen = set(state["seen"])
        batch = []
        for i in np.argsort(acq)[::-1]:
            key = cands[i].tobytes()
            if key not in seen:
                seen.add(key)
                batch.append(cands[i])
            if len(batch) == self.batch_size:
                break
        while len(batch) < self.batch_size:  # 후보가 모자라면 랜덤 보충
            batch.append(self._random_batch(rng, 1)[0])
        _rng_save(state, rng)
        return np.array(batch), state

    def _update(self, state: dict, X_hist: np.ndarray, scores_hist: np.ndarray) -> dict:
        rng = _rng_load(state)
        n = len(scores_hist)
        new_idx = np.arange(state["pending_start"], n) \
            if "pending_start" in state else np.arange(n)
        state["pending_start"] = n
        state["tell_count"] += 1
        for i in new_idx:  # 기관측 set 증분 갱신 (ask 의 중복 제안 방지용)
            state["seen"].add(X_hist[i].tobytes())

        # ── trust region 갱신: trajectory best 가 이번 batch 에서 나왔는가?
        # (같은 스케일러 하의 비교라 재정규화에 안전한 판정 방식)
        traj = np.arange(state["restart_start"], n)
        if len(traj) and n >= self.n_startup:
            best_i = traj[int(np.argmax(scores_hist[traj]))]
            if best_i in new_idx:  # 개선
                state["succ"] += 1
                state["fail"] = 0
                if state["succ"] >= self.succ_tol:
                    state["radius"] = min(state["radius"] * 2, self.r_max)
                    state["succ"] = 0
            else:                  # 정체
                state["fail"] += 1
                state["succ"] = 0
                if state["fail"] >= self.fail_tol:
                    state["fail"] = 0
                    new_r = state["radius"] // 2
                    if new_r < 1:  # 수렴 → restart (모델/히스토리는 유지)
                        state["radius"] = self.r_init
                        state["restart_start"] = n
                        state["reseed"] = True
                    else:
                        state["radius"] = new_r

        # ── 앙상블 재학습 (주기적) ──
        # 장기 실행에서는 재학습 간격을 히스토리 크기에 비례해 늘린다
        # (100K 시점엔 ~50 tell 마다 — 800 evals 수준에서는 기본값 그대로).
        interval = max(self.refit_interval, n // 2000)
        need_refit = (
            n >= self.n_startup
            and (state["models"] is None
                 or state["tell_count"] % interval == 0)
        )
        if need_refit:
            state["models"] = self._fit_ensemble(rng, X_hist, scores_hist)
        _rng_save(state, rng)
        return state


# ──────────────────────────────────────────────────────────────────────────────
# 11) 블록-FOS GOMEA (Gene-pool Optimal Mixing, 고정 linkage)
# ──────────────────────────────────────────────────────────────────────────────

class GOMEABlockOptimizer(OptimizerBase):
    """블록 구조를 FOS 로 고정 주입한 GOMEA. 배경은 doc/algo/ 리서치 문서 참조.

    GOMEA 의 variation = optimal mixing:
        member 의 FOS 부분집합 하나를 donor 의 유전자로 통째로 덮어써 보고,
        평가해서 나빠지지 않았으면 채택, 나빠졌으면 롤백. (시도 1회 = 평가 1회)

    표준 GOMEA 는 FOS(linkage)를 population 에서 학습하지만, 이 문제는
    의존성 골격(set1 ⫫ set2 | common)을 **이미 알고 있으므로** FOS 를
    고정한다 — eda_tree 가 예산을 태워 배우려다 실패한 구조를 공짜로 얻는다.
    FOS = 블록 3개 + 5컬럼 하위 블록들 (거친 이식과 미세 이식의 혼합):
        {common(10), set1(5), set2(15),
         common 전/후반(5+5), set2 3등분(5+5+5)}   → 8개

    추가 장치:
      - forced improvement: 한 pass 동안 아무것도 채택되지 않은 member 는
        elitist 를 donor 로 한 번 더 pass (정체 탈출, 표준 GOMEA 기법).
      - 평가 캐시: 이미 평가한 X 와 동일한 제안은 평가 없이 건너뛴다.
      - 노이즈: 채택 판정은 매 tell 재정규화된 최신 점수로 한다.
    batch=1 (ask 하나 = mixing 시도 하나) — 순차 평가 가정과 정합.
    """

    name = "gomea_block"

    def __init__(self, space: SearchSpace, total_budget: int = 800,
                 pop_size: int = 16, **base_kwargs):
        super().__init__(space, total_budget, **base_kwargs)
        self.pop_size = pop_size
        common = space.block_cols("common")
        set1 = space.block_cols("set1")
        set2 = space.block_cols("set2")
        # 고정 FOS: 블록 전체(거친 이식) + 5컬럼 하위 블록(미세 이식)
        self.fos: list[np.ndarray] = [
            common, set1, set2,
            common[:5], common[5:],
            set2[:5], set2[5:10], set2[10:],
        ]

    def init_state(self, seed: int) -> dict:
        state = super().init_state(seed)
        state["phase"] = "init"       # "init" → "mix"
        state["member_X"] = None      # (P, 30) 각 member 의 현재 해
        state["member_sidx"] = None   # (P,) 각 member 점수의 히스토리 인덱스
        state["elitist"] = -1         # 최고 member 의 인덱스 (0..P-1)
        state["m"] = 0                # 현재 mixing 중인 member
        state["fos_perm"] = None      # 이 member pass 의 FOS 방문 순서
        state["f"] = 0                # fos_perm 상의 위치
        state["pass_accepted"] = False
        state["in_fi"] = False        # forced improvement pass 여부
        state["cache"] = set()        # 평가된 X (재평가 회피)
        state["pending"] = None       # ("init"|"mix"|"rand", member)
        state["pending_start"] = 0
        return state

    def _next_member(self, state: dict) -> None:
        """현재 member 의 pass 를 끝내고 다음 member 로 넘어간다."""
        state["m"] = (state["m"] + 1) % self.pop_size
        state["fos_perm"] = None
        state["in_fi"] = False

    def ask(self, state: dict) -> tuple[np.ndarray, dict]:
        rng = _rng_load(state)
        if state["phase"] == "init":
            batch = self._random_batch(rng, self.pop_size)  # 초기 population
            state["pending"] = ("init", -1)
            _rng_save(state, rng)
            return batch, state

        # '평가할 가치가 있는'(member 와 다르고 미관측인) 제안이 나올 때까지
        # member/FOS 커서를 전진시킨다. 최대 두 바퀴 안에 반드시 끝난다.
        for _ in range(2 * self.pop_size * (len(self.fos) + 1)):
            if state["fos_perm"] is None:  # 새 member pass 시작
                state["fos_perm"] = rng.permutation(len(self.fos)).tolist()
                state["f"] = 0
                state["pass_accepted"] = False

            if state["f"] >= len(self.fos):  # pass 종료
                if (not state["pass_accepted"] and not state["in_fi"]
                        and state["m"] != state["elitist"]):
                    # forced improvement: elitist 를 donor 로 한 pass 더
                    state["in_fi"] = True
                    state["fos_perm"] = rng.permutation(len(self.fos)).tolist()
                    state["f"] = 0
                else:
                    self._next_member(state)
                continue

            m = state["m"]
            F = self.fos[state["fos_perm"][state["f"]]]
            state["f"] += 1
            if state["in_fi"]:
                donor = state["elitist"]
            else:
                donor = int(rng.integers(self.pop_size - 1))
                donor += donor >= m  # 자기 자신 제외
            x = state["member_X"][m].copy()
            x[F] = state["member_X"][donor][F]
            if (x != state["member_X"][m]).any() and x.tobytes() not in state["cache"]:
                state["pending"] = ("mix", m)
                _rng_save(state, rng)
                return x[None, :], state

        # population 이 수렴해 새 제안이 없음 → 랜덤 이민자로 다양성 주입
        batch = self._random_batch(rng, 1)
        state["pending"] = ("rand", -1)
        _rng_save(state, rng)
        return batch, state

    def _update(self, state: dict, X_hist: np.ndarray, scores_hist: np.ndarray) -> dict:
        n = len(scores_hist)
        new_idx = np.arange(state["pending_start"], n)
        for i in new_idx:
            state["cache"].add(X_hist[i].tobytes())

        kind, m = state["pending"]
        if kind == "init":
            state["member_X"] = X_hist[new_idx].copy()
            state["member_sidx"] = new_idx.copy()
            state["phase"] = "mix"
        elif kind == "mix":
            i = int(new_idx[-1])
            cur = scores_hist[state["member_sidx"][m]]  # 최신 재정규화 점수
            # GOMEA 관례: 나빠지지 않으면 채택 (plateau 표류 허용).
            # forced improvement 는 목적상 '엄격한 개선'만 채택한다.
            accept = (scores_hist[i] > cur) if state["in_fi"] \
                else (scores_hist[i] >= cur)
            if accept:
                state["member_X"][m] = X_hist[i].copy()
                state["member_sidx"][m] = i
                state["pass_accepted"] = True
                if state["in_fi"]:  # FI 성공 → 그 member pass 종료
                    self._next_member(state)
        else:  # "rand" 이민자: 최약체 member 보다 좋으면 교체
            i = int(new_idx[-1])
            worst = int(np.argmin(scores_hist[state["member_sidx"]]))
            if scores_hist[i] > scores_hist[state["member_sidx"][worst]]:
                state["member_X"][worst] = X_hist[i].copy()
                state["member_sidx"][worst] = i

        if state["member_sidx"] is not None:  # elitist 갱신 (최신 점수 기준)
            state["elitist"] = int(np.argmax(scores_hist[state["member_sidx"]]))
        state["pending_start"] = n
        return state


# ──────────────────────────────────────────────────────────────────────────────
# 레지스트리 — runner 가 이름으로 optimizer 를 생성한다
# ──────────────────────────────────────────────────────────────────────────────

OPTIMIZERS: dict[str, type[OptimizerBase]] = {
    RandomSearchOptimizer.name: RandomSearchOptimizer,
    BlockwiseCoordinateOptimizer.name: BlockwiseCoordinateOptimizer,
    GAOptimizer.name: GAOptimizer,
    SAOptimizer.name: SAOptimizer,
    PSOOptimizer.name: PSOOptimizer,
    ACOOptimizer.name: ACOOptimizer,
    TPEOptimizer.name: TPEOptimizer,
    XGBSurrogateOptimizer.name: XGBSurrogateOptimizer,
    ChowLiuTreeEDAOptimizer.name: ChowLiuTreeEDAOptimizer,
    GOMEABlockOptimizer.name: GOMEABlockOptimizer,
    XGBTrustRegionOptimizer.name: XGBTrustRegionOptimizer,
}


if __name__ == "__main__":
    # 간단한 자가 점검: 모든 optimizer 가 ask-tell 사이클을 돌 수 있는지 확인.
    # (진짜 벤치마크 대신 임의 raw 관측을 tell 해도 인터페이스는 성립해야 한다)
    import pickle

    space = SearchSpace()
    rng = np.random.default_rng(0)
    n_obj = len(OBJECTIVE_SENSES)
    for name, cls in OPTIMIZERS.items():
        opt = cls(space, total_budget=100)
        state = opt.init_state(seed=7)
        n = 0
        for it in range(6):
            batch, state = opt.ask(state)
            assert batch.ndim == 2 and batch.shape[1] == space.n_cols
            assert (batch >= space.x_min).all() and (batch <= space.x_max).all()
            # 스케일이 제각각인 임의 raw 관측 (은닉 스케일 가정 흉내)
            Y_raw = rng.normal(0.0, 1.0, (len(batch), n_obj)) * 100.0
            state = opt.tell(state, batch, Y_raw)
            n += len(batch)
        assert state["n_evals"] == n == len(state["scores_hist"])
        # stateless 요건: 상태가 pickle 직렬화 가능해야 한다 (체크포인트)
        blob = pickle.dumps(state)
        state2 = pickle.loads(blob)
        batch, _ = opt.ask(state2)
        print(f"[OK] {name:>16s} — {n} evals, "
              f"state pickle {len(blob)} bytes, resume ask batch={len(batch)}")

    # 파일 교환 셸 점검: x.txt 왕복 무손실 + fail-loud
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.txt"
        X = space.sample(rng, n=10)
        write_x(p, X, eval_index=123)
        X2, idx = read_x(p, space=space)
        assert np.array_equal(X, X2) and idx == 123
        assert (X2.min(axis=0) < 0).any(), "signed 값이 실제로 왕복되어야 함"
        for content in [
            "[1,2,3]\n",                                # 헤더 없음
            "# eval_index=0\n[1,2.5,3]\n",              # 정수 아님
            "# eval_index=0\n[1,2]\n[1,2,3]\n",         # 길이 불일치
            "# eval_index=0\n[" + ",".join(["999"] * space.n_cols) + "]\n",  # 범위 밖
        ]:
            p.write_text(content)
            try:
                read_x(p, space=space)
            except ValueError:
                pass
            else:
                raise AssertionError(f"raise 됐어야 함: {content!r}")
    print(f"[OK] {'x.txt 셸':>16s} — 왕복 무손실 ({X.shape}), fail-loud 4종 통과")

    # y_raw.bin 셸 점검: 구조화 왕복 + convert 측정 이음새 + fail-loud
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "y_raw.bin"
        g = 32
        raw = {
            "mask1": rng.random((10, g, g)) < 0.3,
            "mask2": rng.random((10, g, g)) < 0.5,
            "y13": rng.normal(0, 1, 10) * 5e-3,
            "y23": rng.normal(0, 1, 10) * 5e-3,
        }
        write_y_raw(p, raw, eval_index=123)
        raw2, idx = read_y_raw(p)
        assert idx == 123
        for k in raw:
            assert np.array_equal(raw[k], raw2[k]), f"{k} 왕복 불일치"
        Y = convert_y_raw(raw2, n_obj=n_obj)
        assert Y.shape == (10, n_obj)
        # 측정 정의 확인: 알려진 직사각형 마스크 → height/width 정확
        rect = np.zeros((1, g, g), dtype=bool)
        rect[0, 5:15, 3:7] = True  # height 10, width 4
        Yr = convert_y_raw({"mask1": rect, "mask2": rect,
                            "y13": [0.0], "y23": [0.0]}, n_obj=n_obj)
        assert Yr[0, 0] == 10 and Yr[0, 1] == 4 and Yr[0, 3] == 10 and Yr[0, 4] == 4
        p.write_bytes(p.read_bytes()[:-8])  # 잘린 파일 → raise
        try:
            read_y_raw(p)
        except ValueError:
            pass
        else:
            raise AssertionError("잘린 y_raw.bin 은 raise 됐어야 함")
        try:
            convert_y_raw({"mask1": raw["mask1"], "mask2": raw["mask2"],
                           "y13": [np.nan] * 10, "y23": raw["y23"]})  # NaN → raise
        except ValueError:
            pass
        else:
            raise AssertionError("NaN 은 raise 됐어야 함")
    print(f"[OK] {'y_raw.bin 셸':>16s} — 구조화 왕복 동일, 측정 이음새, fail-loud 통과")

    # 체크포인트 점검: history.jsonl + state.pkl 로 중단·재개 = 무중단과 동일 궤적
    with tempfile.TemporaryDirectory() as d:
        hist = Path(d) / "history.jsonl"
        st = Path(d) / "state.pkl"
        W = np.random.default_rng(1).normal(size=(space.n_cols, n_obj))

        def f(X):  # 결정적 합성 관측 (재개 검증엔 노이즈 불필요)
            return X.astype(np.float64) @ W

        for name in ["sa", "ga"]:  # batch=1 과 batch=20 대표
            opt = OPTIMIZERS[name](space, total_budget=200)
            s = opt.init_state(seed=3)  # ── 무중단 12 tells (기준 궤적)
            for _ in range(12):
                b, s = opt.ask(s)
                s = opt.tell(s, b, f(b))
            ref_X, ref_s = s["X_hist"].copy(), s["scores_hist"].copy()

            hist.unlink(missing_ok=True)  # ── 6 tells → 체크포인트 → 재개 6 tells
            s2 = opt.init_state(seed=3)
            for _ in range(6):
                b, s2 = opt.ask(s2)
                n0 = s2["n_evals"]
                s2 = opt.tell(s2, b, f(b))
                append_history(hist, b, f(b), eval_index=n0)
            save_state(st, s2)
            s3 = load_state(st, hist, space=space)
            for _ in range(6):
                b, s3 = opt.ask(s3)
                s3 = opt.tell(s3, b, f(b))
            assert np.array_equal(s3["X_hist"], ref_X), f"{name}: 재개 궤적 불일치"
            assert np.allclose(s3["scores_hist"], ref_s), f"{name}: 재개 점수 불일치"

        # 정합성 fail-loud: 히스토리에 여분 batch → n_evals 불일치 → raise
        append_history(hist, space.sample(rng, 1), np.zeros((1, n_obj)),
                       eval_index=s2["n_evals"])
        try:
            load_state(st, hist, space=space)
        except ValueError:
            pass
        else:
            raise AssertionError("n_evals 불일치는 raise 됐어야 함")
    print(f"[OK] {'체크포인트':>16s} — sa/ga 중단·재개 동일 궤적, 정합성 fail-loud 통과")
