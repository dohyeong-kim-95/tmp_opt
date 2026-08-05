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
    n = slim["n_evals"]
    scores = slim.pop("_scores")
    n_obj = len(OBJECTIVE_SENSES)
    if n == 0:
        # 첫 스텝(아직 관측 0) — history.jsonl 이 없을 수 있으므로 읽지 않는다.
        if space is None:
            raise ValueError("n_evals=0 상태 로드에는 space 가 필요하다 (버퍼 폭)")
        X = np.empty((0, space.n_cols), dtype=np.int64)
        Y = np.empty((0, n_obj), dtype=np.float64)
    else:
        X, Y = load_history(history_path, space=space)
        if len(X) != n:
            raise ValueError(
                f"체크포인트 정합성 위반 — state.pkl n_evals={n} ≠ history.jsonl {len(X)}")
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
    개수 기반이라 가장자리 픽셀이 흔들려도 ±수 픽셀 수준으로만 반응한다.
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


# ─── 프로세스 분리 실행: optimizer 한 스텝 (매 호출 = 새 프로세스) ─────────────
#
# 파일 기반 계약의 optimizer 측. runner 가 이 함수를 서브프로세스로 반복 호출하고,
# calculator 는 별도 프로세스로 y_raw.bin 을 채운다. 두 프로세스는 **공유 메모리가
# 없으므로** state.pkl + history.jsonl 이 스텝 간 유일한 기억이다 (in-process
# 우회가 구조적으로 불가능 — 파일이 유일한 통신 수단).
#
# 교환 디렉토리 파일:
#   x.txt        optimizer→calculator 다음 후보 (매 스텝 덮어씀)
#   y_raw.bin    calculator→optimizer 직전 x 의 관측 (매 스텝 덮어씀)
#   history.jsonl / state.pkl   optimizer 의 지속 상태
#   done         예산 소진 시 optimizer 가 남기는 종료 마커
# ──────────────────────────────────────────────────────────────────────────────

def serve_step(optimizer_name: str, exchange_dir: str | Path,
               seed: int, budget: int) -> str:
    """optimizer 한 스텝: 상태 로드 → (직전 y_raw 있으면) tell → ask → 저장.

    반환: "proposed"(x.txt 새로 씀) 또는 "done"(예산 소진, done 마커 씀).
    """
    d = Path(exchange_dir)
    space = SearchSpace()
    opt = OPTIMIZERS[optimizer_name](space, total_budget=budget)
    assert opt.name == optimizer_name, \
        f"dispatch 불일치: 요청 {optimizer_name!r} → 생성 {opt.name!r}"
    st_p, hist_p = d / "state.pkl", d / "history.jsonl"
    x_p, y_p, done_p = d / "x.txt", d / "y_raw.bin", d / "done"

    if st_p.exists():
        state = load_state(st_p, hist_p, space=space)
    else:
        state = opt.init_state(seed)  # 첫 스텝

    n = state["n_evals"]
    # 직전 제안(x.txt)에 대한 응답(y_raw.bin)이 있으면 ingest
    if y_p.exists():
        raw, y_idx = read_y_raw(y_p)
        if y_idx == n:  # 우리가 마지막에 낸 배치에 대한 fresh 응답
            X_last, x_idx = read_x(x_p, space=space)
            if x_idx != n:
                raise ValueError(f"대응 위반: x.txt idx={x_idx} ≠ 기대 {n}")
            state = opt.tell(state, X_last, raw)
            append_history(hist_p, X_last,
                           state["Y_raw_hist"][n:state["n_evals"]], eval_index=n)
            n = state["n_evals"]
        elif y_idx > n:
            raise ValueError(f"y_raw.bin idx={y_idx} > n_evals={n} — 손상")
        # y_idx < n: 이미 소비됨 (엄격 순서면 발생 안 함)

    if n >= budget:  # 예산 소진 → 종료 마커
        save_state(st_p, state)
        done_p.write_text(f"n_evals={n}\n")
        return "done"

    batch, state = opt.ask(state)
    batch = batch[: budget - n]  # 예산 초과분 절단
    write_x(x_p, batch, eval_index=n)
    save_state(st_p, state)
    return "proposed"


# ──────────────────────────────────────────────────────────────────────────────
# 공유 score 파이프라인 — raw y0 → 정규화 z → 스칼라 점수
# 탐색 구동(OptimizerBase.tell)과 사후 리포트가
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


def get_scores(Y_raw: np.ndarray, kind: str = "chebyshev",
               scaler: "RobustScaler | None" = None, **scorer_kw) -> np.ndarray:
    """raw 관측 (n, 6) → 점수 (n,). **점수 파이프라인의 단일 진입점.**

    정규화(robust p5–p95) → sense 통일(최소화 목적 뒤집기) → scalarization 을
    한 번에 한다. `OptimizerBase.tell` 도, 밖에서 직접 도는 루프도, 사후 분석도
    전부 이 함수를 지나야 한다 — 구현이 두 벌이 되는 순간 알고리즘 간 비교가
    조용히 무의미해진다 (실행은 안 깨지므로 아무도 눈치채지 못한다).

    Args:
        Y_raw     : (n, 6) 열 순서 = OBJECTIVE_NAMES
        kind      : SCORERS 키 ("chebyshev" 기본 / "sum" / "owa")
        scaler    : 이미 적합된 RobustScaler. None 이면 Y_raw 로 새로 적합한다
                    (= 전체 히스토리 재적합. 기본 동작)
        scorer_kw : scalarization 파라미터 (chebyshev 의 rho, owa 의 k 등)

    사용 예 (OptimizerBase 없이 직접 루프를 도는 경우):
        Y = convert_y_raw(calc.evaluate(X))
        scores = get_scores(Y)            # 알고리즘에 넘길 [0,1] 점수
    """
    if kind not in SCORERS:
        raise ValueError(f"알 수 없는 scorer {kind!r} — 사용 가능: {sorted(SCORERS)}")
    sc = RobustScaler().fit(Y_raw) if scaler is None else scaler
    return SCORERS[kind](sc.transform(Y_raw), **scorer_kw)


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
        # 점수는 get_scores 한 곳에서만 만들어진다 (밖에서 직접 루프를 돌 때도
        # 같은 함수를 쓰면 결과가 정확히 일치한다 — 구현이 갈라질 여지가 없다)
        scaler = RobustScaler()
        if state["_lo"] is None or state["_since_refit"] >= self.rescore_interval:
            scaler.fit(state["_Y_buf"][:n])
            state["_lo"], state["_hi"] = scaler.lo, scaler.hi
            state["_s_buf"][:n] = get_scores(state["_Y_buf"][:n], self.scorer_name,
                                             scaler=scaler)
            state["_since_refit"] = 0
        else:  # 장기 실행 경로: 새 관측만 기존 스케일러로 점수화
            scaler.lo, scaler.hi = state["_lo"], state["_hi"]
            state["_s_buf"][n0:n] = get_scores(Y_new, self.scorer_name, scaler=scaler)

        self._sync_views(state)
        return self._update(state, state["X_hist"], state["scores_hist"])

    def _update(self, state: dict, X_hist: np.ndarray, scores_hist: np.ndarray) -> dict:
        """알고리즘별 상태 갱신 훅. 최신 재정규화 점수의 전체 히스토리를 받는다."""
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# 알고리즘 — 전부 `@simple_algorithm` 함수다 (인자 하나, 반환 하나).
#
# 클래스가 필요한 것은 상태 기계가 큰 두 종뿐이다 (blockwise_coord,
# gomea_block — 아래에 따로 있다). 새 알고리즘은 이 형태로 쓰면 된다:
#   doc/algo/adding_an_algorithm.md · 복사용 템플릿 algo_template.py
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# 확장 지점 — **함수 하나로 알고리즘 추가하기**
#
# 위의 11종은 클래스로 쓰였지만, 클래스는 요구사항이 아니다. 새 알고리즘은
# 함수 하나로 충분하다:
#
#     @algorithm("my_sa", state={"cur": None}, t_start=0.1)
#     def my_sa(X, scores, state, rng, ctx):
#         if len(X) == 0:
#             return ctx.sample(rng, 1)              # 첫 호출: 히스토리가 비어 있다
#         i = len(scores) - 1                        # 방금 평가된 점
#         ...
#         return [ctx.mutate(rng, X[state["cur"]])], state
#
# 계약은 하나다: **"지금까지의 관측을 보고 다음에 평가할 X 들을 돌려준다."**
# 매 라운드 한 번 호출되며, 그때의 전체 히스토리를 받는다.
#
# 인자는 **선언한 것만** 넘어온다 (이름으로 매칭). 쓸 수 있는 이름:
#     X       (n, 30) int64   지금까지 평가된 X (평가 순서)
#     Y       (n, 6)  float   대응 raw 관측 (열 순서 = OBJECTIVE_NAMES)
#     scores  (n,)    float   [0,1] 점수, 클수록 좋음. 스케일 정규화·sense 통일·
#                             scalarization 이 이미 적용돼 있다 (RobustScaler +
#                             SCORERS). 알고리즘이 다시 구현할 필요가 없고,
#                             다시 구현하면 알고리즘 간 비교가 무의미해진다.
#     state   dict            라운드 간 기억. @algorithm(state=...) 가 초기값.
#     rng     Generator       난수. 이것만 쓰면 재현성이 보장된다.
#     ctx     Ctx             space / budget / cfg + 헬퍼(sample, mutate, top_k)
#
# 반환은 `xs` 또는 `(xs, state)`. xs 는 (30,) 하나, 그 리스트, 또는 (b, 30) 배열.
# state 는 dict 라 제자리 수정도 반영되므로 굳이 반환하지 않아도 된다.
#
# 지켜야 할 것 하나: **state 에는 pickle 가능한 값만 넣는다.** 프로세스 분리
# 실행(`--serve-step`)은 매 스텝이 새 프로세스라 state.pkl 이 유일한 기억이다.
# (그래서 제너레이터/코루틴 스타일은 쓸 수 없다 — 실행 상태는 직렬화 불가)
#
# 다른 파일에 정의했다면 프로세스 분리 실행에 `--plugin 모듈명` 을 붙인다.
# 자세한 안내와 템플릿: `algo_template.py`, `doc/algo/adding_an_algorithm.md`
# ──────────────────────────────────────────────────────────────────────────────


class Ctx:
    """알고리즘이 쓰는 공간·예산·설정과 헬퍼. **탐색 상태는 담지 않는다.**"""

    def __init__(self, space: SearchSpace, budget: int, cfg: dict) -> None:
        self.space = space
        self.budget = budget
        self.cfg = cfg

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """균등 랜덤 X (n, 30)."""
        return self.space.sample(rng, n)

    def mutate(self, rng: np.random.Generator, x: np.ndarray,
               rate: float = 1.0 / 30) -> np.ndarray:
        """ordinal 이웃: 대체로 ±1 스텝, 가끔 랜덤 점프. 최소 1컬럼은 변한다."""
        x = np.asarray(x, dtype=np.int64).copy()
        m = rng.random(self.space.n_cols) < rate
        if not m.any():
            m[rng.integers(self.space.n_cols)] = True
        for c in np.flatnonzero(m):
            if rng.random() < 0.8:
                x[c] += rng.choice([-1, 1])
            else:
                x[c] = rng.integers(self.space.x_min[c], self.space.x_max[c] + 1)
        return self.space.clip(x)

    @staticmethod
    def top_k(X: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
        """점수 상위 k 개의 X (내림차순)."""
        return X[np.argsort(scores)[::-1][:k]]

    def clip(self, x: np.ndarray) -> np.ndarray:
        return self.space.clip(x)


def algorithm(name: str, state: dict | None = None, **cfg):
    """함수 하나를 optimizer 로 등록하는 데코레이터 (`OPTIMIZERS[name]`).

    Args:
        name  : 레지스트리 키 (runner 의 --optimizer 값)
        state : 라운드 간 기억의 초기값. 매 run 마다 깊은 복사되어 들어간다.
        **cfg : 하이퍼파라미터 기본값 → `ctx.cfg`. 생성 시 덮어쓸 수 있다.
    """
    import copy
    import inspect

    if state is not None and not isinstance(state, dict):
        raise TypeError("state 는 dict 여야 한다 (pickle 가능한 값만)")

    def decorate(fn):
        params = list(inspect.signature(fn).parameters)
        unknown = set(params) - {"X", "Y", "scores", "state", "rng", "ctx"}
        if unknown:
            raise TypeError(
                f"{name}: 알 수 없는 인자 {sorted(unknown)} — "
                "쓸 수 있는 이름은 X, Y, scores, state, rng, ctx")

        class _FunctionOptimizer(OptimizerBase):
            __doc__ = fn.__doc__ or f"함수 스타일 optimizer {name!r}."

            def __init__(self, space, total_budget: int = 800, **kw):
                base = {k: kw.pop(k) for k in ("scorer_name", "rescore_interval")
                        if k in kw}
                super().__init__(space, total_budget, **base)
                merged = dict(cfg)
                merged.update(kw)  # 나머지 키워드는 하이퍼파라미터 덮어쓰기
                self.ctx = Ctx(space, total_budget, merged)

            def init_state(self, seed: int) -> dict:
                st = super().init_state(seed)
                st.update(copy.deepcopy(state) if state else {})
                return st

            def ask(self, st: dict) -> tuple[np.ndarray, dict]:
                rng = _rng_load(st)
                pool = {"X": st["X_hist"], "Y": st["Y_raw_hist"],
                        "scores": st["scores_hist"], "state": st,
                        "rng": rng, "ctx": self.ctx}
                out = fn(**{k: pool[k] for k in params})
                if isinstance(out, tuple) and len(out) == 2 and \
                        isinstance(out[1], dict):
                    xs, new_state = out
                    if new_state is not st:
                        st.update(new_state)
                else:
                    xs = out
                batch = np.atleast_2d(np.asarray(xs, dtype=np.int64))
                if batch.ndim != 2 or batch.shape[1] != self.space.n_cols:
                    raise ValueError(
                        f"{name}: 반환 형상 {batch.shape} — (b, {self.space.n_cols}) 여야 함")
                if len(batch) == 0:
                    raise ValueError(f"{name}: 빈 배치를 반환했다")
                batch = self.space.clip(batch)  # 범위 밖은 조용히 어긋나지 않게 클램프
                _rng_save(st, rng)
                return batch, st

            def _update(self, st, X_hist, scores_hist):
                return st  # 히스토리 누적은 베이스 tell 이 이미 했다

        _FunctionOptimizer.name = name
        _FunctionOptimizer.__name__ = f"{name}_FunctionOptimizer"
        if name in OPTIMIZERS:
            raise ValueError(f"optimizer 이름 중복: {name!r}")
        OPTIMIZERS[name] = _FunctionOptimizer
        fn.optimizer_cls = _FunctionOptimizer  # 테스트에서 직접 꺼내 쓸 수 있게
        return fn

    return decorate


def simple_algorithm(name: str, state: dict | None = None, **cfg):
    """`algo(data) -> [다음 X, ...]` — **인자 하나, 반환 하나.**

    `examples/owner_minimal.py` 와 같은 모양이다. 인자 이름을 고를 필요도,
    무엇을 선언할지 고민할 필요도 없다. 필요한 건 전부 `data` 안에 있다:

        data["X"]      (n, 30) int64   지금까지 평가된 X
        data["Y"]      (n, 6)  float   대응 raw 관측
        data["scores"] (n,)    float   [0,1] 점수, 클수록 좋음
        data["state"]  dict            라운드 간 기억 (**제자리 수정**하면 된다)
        data["rng"]    Generator       난수 — 반드시 이것만
        data["space"]  SearchSpace     x_min / x_max / n_cols / sample / clip
        data["cfg"]    dict            @simple_algorithm 에 넘긴 하이퍼파라미터
        data["ctx"]    Ctx             헬퍼 (sample / mutate / top_k / clip)

    히스토리만 보면 되는 알고리즘은 `data["state"]` 를 아예 안 써도 된다
    (그때는 `examples/owner_minimal.py` 처럼 파일만으로도 돌릴 수 있다).
    SA 의 '현재 해', PSO 의 속도처럼 관측에 안 적히는 기억이 필요하면
    `data["state"]` 에 넣는다 — pickle 가능한 값만.

        @simple_algorithm("my_ga", n_elite=8, batch=4)
        def my_ga(data):
            X, s, rng = data["X"], data["scores"], data["rng"]
            if len(X) < data["cfg"]["n_elite"]:
                return data["ctx"].sample(rng, data["cfg"]["n_elite"])
            elite = data["ctx"].top_k(X, s, data["cfg"]["n_elite"])
            return [data["ctx"].mutate(rng, elite[rng.integers(len(elite))])
                    for _ in range(data["cfg"]["batch"])]

    `@algorithm` 과 완전히 같은 계약이다 — 인자를 이름으로 받느냐 dict 하나로
    받느냐의 차이뿐이고, runner·체크포인트·프로세스 분리 전부 동일하게 돈다.
    """
    def decorate(fn):
        @algorithm(name, state=state, **cfg)
        def _bundled(X, Y, scores, state, rng, ctx):
            return fn({"X": X, "Y": Y, "scores": scores, "state": state,
                       "rng": rng, "space": ctx.space, "cfg": ctx.cfg,
                       "budget": ctx.budget, "ctx": ctx})

        _bundled.__name__ = fn.__name__
        _bundled.__doc__ = fn.__doc__
        fn.optimizer_cls = _bundled.optimizer_cls
        return fn

    return decorate


def load_plugins(modules) -> list[str]:
    """알고리즘이 정의된 모듈을 import 해 레지스트리에 올린다.

    프로세스 분리 실행은 매 스텝 새 프로세스라, 외부 파일에 정의한 알고리즘은
    그 프로세스에서 다시 import 되어야 한다 (`--plugin` 이 이 함수를 부른다).
    """
    import importlib
    import sys

    added = []
    for m in modules or ():
        before = set(OPTIMIZERS)
        importlib.import_module(m)
        # `python optimizer.py` 로 실행하면 이 코드는 __main__ 모듈에 있고,
        # 플러그인의 `from optimizer import algorithm` 은 **별개의 모듈 객체**를
        # 만든다 (sys.modules["optimizer"] ≠ __main__). 그래서 등록이 저쪽
        # 레지스트리로 가버린다 — 여기서 다시 합쳐 준다.
        other = getattr(sys.modules.get("optimizer"), "OPTIMIZERS", None)
        if other is not None and other is not OPTIMIZERS:
            for k, v in other.items():
                OPTIMIZERS.setdefault(k, v)
        added += sorted(set(OPTIMIZERS) - before)
    return added

OPTIMIZERS: dict = {}   # @algorithm / 아래 클래스들이 채운다

# ── 1) random ────────────────────────────────────────────────────────────
@simple_algorithm("random", batch=10)
def random(data):
    """균등 랜덤 샘플링. 모든 비교의 baseline."""
    return data["ctx"].sample(data["rng"], data["cfg"]["batch"])


# ── 2) ga ────────────────────────────────────────────────────────────────
@simple_algorithm("ga", state={"parent": None, "seen_n": 0},
                  pop=20, mut=2.0 / 30)
def ga(data):
    """(μ+λ) GA — 블록 단위 crossover + 토너먼트 선택(k=3)."""
    X, s, st, rng, ctx, cfg = (data["X"], data["scores"], data["state"],
                               data["rng"], data["ctx"], data["cfg"])
    sp, pop = ctx.space, cfg["pop"]

    # ── 환경 선택: (부모 ∪ 이번 자식) 상위 pop 생존 ──
    n = len(s)
    if n > st["seen_n"]:
        new_idx = np.arange(st["seen_n"], n)
        pool = new_idx if st["parent"] is None else \
            np.concatenate([st["parent"], new_idx])
        st["parent"] = pool[np.argsort(s[pool])[::-1]][:pop]
        st["seen_n"] = n
    if st["parent"] is None:
        return ctx.sample(rng, pop)

    p_X, p_s = X[st["parent"]], s[st["parent"]]
    out = np.empty((pop, sp.n_cols), dtype=np.int64)
    for i in range(pop):
        def pick():
            cand = rng.integers(0, len(p_s), 3)          # 토너먼트 k=3
            return int(cand[np.argmax(p_s[cand])])
        pa, pb = p_X[pick()], p_X[pick()]
        child = pa.copy()
        for name in sp.blocks:                            # 블록 단위 crossover
            if rng.random() < 0.5:
                cols = sp.block_cols(name)
                child[cols] = pb[cols]
        swap = rng.random(sp.n_cols) < 0.1                # 미세 혼합
        child[swap] = pb[swap]
        out[i] = ctx.mutate(rng, child, cfg["mut"])
    return out


# ── 3) sa ────────────────────────────────────────────────────────────────
@simple_algorithm("sa", state={"cur": None, "seen_n": 0},
                  t_start=0.1, t_end=1e-3)
def sa(data):
    """Simulated Annealing (batch=1). 수락 판정은 최신 재정규화 점수로."""
    X, s, st, rng, ctx, cfg = (data["X"], data["scores"], data["state"],
                               data["rng"], data["ctx"], data["cfg"])
    n = len(s)
    if n > st["seen_n"]:                                  # Metropolis 수락 판정
        prop = n - 1
        if st["cur"] is None:
            st["cur"] = prop
        else:
            frac = min(1.0, n / max(1, data["budget"]))
            T = cfg["t_start"] * (cfg["t_end"] / cfg["t_start"]) ** frac
            d = float(s[prop] - s[st["cur"]])
            if d >= 0 or rng.random() < np.exp(d / max(T, 1e-12)):
                st["cur"] = prop
        st["seen_n"] = n
    if st["cur"] is None:
        return ctx.sample(rng, 1)
    k = int(rng.integers(1, 4))                           # 1~3 컬럼 변경
    return [ctx.mutate(rng, X[st["cur"]], rate=k / ctx.space.n_cols)]


# ── 4) pso ───────────────────────────────────────────────────────────────
@simple_algorithm("pso", state={"pbest": None, "seen_n": 0, "pos": None,
                                   "vel": None},
                  swarm=20, w=0.7, c1=1.5, c2=1.5)
def pso(data):
    """이산화 PSO — 연속 위치 [0,1] 을 유지하다가 반올림해 평가한다."""
    X, s, st, rng, ctx, cfg = (data["X"], data["scores"], data["state"],
                               data["rng"], data["ctx"], data["cfg"])
    sp, S = ctx.space, cfg["swarm"]
    if st["pos"] is None:                                 # 최초 1회 초기화
        st["pos"] = rng.uniform(0, 1, (S, sp.n_cols))
        st["vel"] = rng.uniform(-0.1, 0.1, (S, sp.n_cols))

    n = len(s)
    if n > st["seen_n"]:                                  # personal/global best 갱신
        new_idx = np.arange(st["seen_n"], n)
        if st["pbest"] is None:
            st["pbest"] = new_idx.copy()
        else:
            imp = s[new_idx] > s[st["pbest"]]
            st["pbest"][imp] = new_idx[imp]
        st["pbest_u"] = sp.to_unit(X[st["pbest"]])
        g = st["pbest"][int(np.argmax(s[st["pbest"]]))]
        st["gbest_u"] = sp.to_unit(X[g])
        st["seen_n"] = n

    if st["pbest"] is not None:                           # 속도/위치 갱신
        pos, vel = st["pos"], st["vel"]
        r1, r2 = rng.uniform(0, 1, pos.shape), rng.uniform(0, 1, pos.shape)
        vel = (cfg["w"] * vel + cfg["c1"] * r1 * (st["pbest_u"] - pos)
               + cfg["c2"] * r2 * (st["gbest_u"][None, :] - pos))
        vel = np.clip(vel, -0.3, 0.3)                     # 한 스텝 이동 상한
        st["pos"], st["vel"] = np.clip(pos + vel, 0.0, 1.0), vel

    x = sp.x_min + np.rint(st["pos"] * (sp.x_max - sp.x_min)).astype(np.int64)
    return sp.clip(x)


# ── 5) aco ───────────────────────────────────────────────────────────────
@simple_algorithm("aco", state={"tau": None, "seen_n": 0},
                  n_ants=20, evap=0.1, n_elite=5, floor=0.05)
def aco(data):
    """컬럼×레벨 페로몬 ACO. 침착은 점수 크기가 아니라 rank 로 — 스케일-프리."""
    X, s, st, rng, ctx, cfg = (data["X"], data["scores"], data["state"],
                               data["rng"], data["ctx"], data["cfg"])
    sp = ctx.space
    max_card = int(sp.cardinalities.max())
    mask = np.arange(max_card)[None, :] < sp.cardinalities[:, None]
    if st["tau"] is None:
        st["tau"] = np.where(mask, 1.0, 0.0)              # 균등 초기 페로몬

    n = len(s)
    if n > st["seen_n"]:                                  # 증발 + elite 침착
        tau = st["tau"] * (1 - cfg["evap"])
        elite = np.argsort(s)[::-1][: cfg["n_elite"]]
        for rank, idx in enumerate(elite):
            dep = cfg["evap"] * (cfg["n_elite"] - rank) / cfg["n_elite"]
            tau[np.arange(sp.n_cols), X[idx] - sp.x_min] += dep
        st["tau"] = np.where(mask, tau, 0.0)
        st["seen_n"] = n

    probs = st["tau"] / st["tau"].sum(axis=1, keepdims=True)
    uni = mask / mask.sum(axis=1, keepdims=True)
    probs = (1 - cfg["floor"]) * probs + cfg["floor"] * uni   # 탐험 하한
    out = np.empty((cfg["n_ants"], sp.n_cols), dtype=np.int64)
    for c in range(sp.n_cols):
        out[:, c] = sp.x_min[c] + rng.choice(len(probs[c]),
                                             size=cfg["n_ants"], p=probs[c])
    return out


# ── 6) tpe ───────────────────────────────────────────────────────────────
def _density(levels, card, alpha=0.5):
    """관측 슬롯 인덱스 → ordinal smoothing + Laplace 평활된 분포."""
    counts = np.bincount(levels, minlength=card).astype(np.float64)
    if card >= 3:
        sm = 0.5 * counts.copy()
        sm[1:] += 0.25 * counts[:-1]
        sm[:-1] += 0.25 * counts[1:]
        counts = sm
    counts += alpha
    return counts / counts.sum()


@simple_algorithm("tpe", gamma=0.2, n_cand=50, n_startup=20)
def tpe(data):
    """밀도비 SMBO. state 불필요 — 모형을 매 호출 히스토리에서 다시 만든다."""
    X, s, rng, ctx, cfg = (data["X"], data["scores"], data["rng"],
                           data["ctx"], data["cfg"])
    sp = ctx.space
    if len(s) < cfg["n_startup"]:
        return ctx.sample(rng, 1)

    n_good = max(5, int(np.ceil(cfg["gamma"] * len(s))))
    order = np.argsort(s)[::-1]
    good, bad = X[order[:n_good]], X[order[n_good:]]

    C = cfg["n_cand"]
    cands = np.empty((C, sp.n_cols), dtype=np.int64)
    log_ratio = np.zeros(C)
    for c in range(sp.n_cols):
        card, xm = int(sp.cardinalities[c]), int(sp.x_min[c])
        p_g = _density(good[:, c] - xm, card)
        p_b = _density(bad[:, c] - xm, card)
        lvl = rng.choice(card, size=C, p=p_g)
        cands[:, c] = xm + lvl
        log_ratio += np.log(p_g[lvl]) - np.log(p_b[lvl])
    return [cands[int(np.argmax(log_ratio))]]


# ── 7) xgb_surrogate ─────────────────────────────────────────────────────
@simple_algorithm("xgb_surrogate",
                  state={"model": None, "rounds": 0, "seen_n": 0},
                  n_startup=30, n_random=150, n_local=150, kappa=0.15,
                  refit=4, batch=4)
def xgb_surrogate(data):
    """XGB 회귀 + novelty 보너스 acquisition."""
    X, s, st, rng, ctx, cfg = (data["X"], data["scores"], data["state"],
                               data["rng"], data["ctx"], data["cfg"])
    sp, n = ctx.space, len(s)
    if n > st.get("seen_n", 0):          # 새 관측 도착 = 클래스판의 tell 시점
        st["rounds"] += 1
        if n >= cfg["n_startup"] and (st["model"] is None
                                      or st["rounds"] % cfg["refit"] == 0):
            from xgboost import XGBRegressor
            m = XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.1,
                             subsample=0.9, colsample_bytree=0.9, n_jobs=2,
                             verbosity=0)
            m.fit(X.astype(np.float32), s.astype(np.float32))
            st["model"] = m
        st["seen_n"] = n

    if n < cfg["n_startup"] or st["model"] is None:
        return ctx.sample(rng, cfg["batch"])

    # 후보 풀: 전역 랜덤 먼저, 그 다음 상위해 mutation (rng 소비 순서 고정)
    rand_pool = ctx.sample(rng, cfg["n_random"])
    top = ctx.top_k(X, s, 10)
    local = np.array([ctx.mutate(rng, top[rng.integers(len(top))], rate=2.0 / 30)
                      for _ in range(cfg["n_local"])])
    cands = np.vstack([rand_pool, local])
    mu = st["model"].predict(cands.astype(np.float32))
    novelty = (cands[:, None, :] != X[None, :, :]).sum(axis=2).min(axis=1) / sp.n_cols
    acq = mu + cfg["kappa"] * novelty

    out, seen = [], set()
    for i in np.argsort(acq)[::-1]:
        key = cands[i].tobytes()
        if key not in seen:
            seen.add(key)
            out.append(cands[i])
        if len(out) == cfg["batch"]:
            break
    return out


# ── 8) eda_tree ──────────────────────────────────────────────────────────
@simple_algorithm("eda_tree", batch=20, gamma=0.25, min_elite=30,
                  max_elite=400, alpha=0.5, floor=0.05, n_startup=40)
def eda_tree(data):
    """Chow-Liu 의존성 트리 EDA. state 불필요 — 트리를 매 호출 다시 만든다."""
    X, s, rng, ctx, cfg = (data["X"], data["scores"], data["rng"],
                           data["ctx"], data["cfg"])
    sp, B, a = ctx.space, cfg["batch"], cfg["alpha"]
    if len(s) < cfg["n_startup"]:
        return ctx.sample(rng, B)

    n_elite = int(np.clip(int(cfg["gamma"] * len(s)),
                          cfg["min_elite"], cfg["max_elite"]))
    elite = X[np.argsort(s)[::-1][:n_elite]]

    def joint(u, v):
        ca, cb = int(sp.cardinalities[u]), int(sp.cardinalities[v])
        lu, lv = elite[:, u] - sp.x_min[u], elite[:, v] - sp.x_min[v]
        cnt = np.bincount(lu * cb + lv, minlength=ca * cb).astype(np.float64)
        return cnt.reshape(ca, cb) + a

    N = sp.n_cols
    mi = np.zeros((N, N))
    for u in range(N):
        for v in range(u + 1, N):
            p = joint(u, v)
            p = p / p.sum()
            mi[u, v] = mi[v, u] = float(
                (p * np.log(p / (p.sum(1, keepdims=True) * p.sum(0, keepdims=True)))).sum())

    in_tree = np.zeros(N, dtype=bool)                     # Prim 최대신장트리
    in_tree[0] = True
    best_w, best_from = mi[0].copy(), np.zeros(N, dtype=np.int64)
    und = []
    for _ in range(N - 1):
        v = int(np.argmax(np.where(in_tree, -np.inf, best_w)))
        und.append((int(best_from[v]), v))
        in_tree[v] = True
        imp = mi[v] > best_w
        best_w[imp], best_from[imp] = mi[v][imp], v

    adj = [[] for _ in range(N)]
    strength = np.zeros(N)
    for u, v in und:
        adj[u].append(v)
        adj[v].append(u)
        strength[u] += mi[u, v]
        strength[v] += mi[u, v]
    root = int(np.argmax(strength))

    edges, visited, queue = [], {root}, [root]            # BFS 로 방향 부여
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                edges.append((u, v))
                queue.append(v)

    def mix(p):
        return (1 - cfg["floor"]) * p + cfg["floor"] / len(p)

    out = np.empty((B, sp.n_cols), dtype=np.int64)
    cr = int(sp.cardinalities[root])
    p_root = np.bincount(elite[:, root] - sp.x_min[root], minlength=cr) + a
    out[:, root] = sp.x_min[root] + rng.choice(cr, size=B, p=mix(p_root / p_root.sum()))
    for parent, child in edges:
        cond = joint(parent, child)
        cond = cond / cond.sum(axis=1, keepdims=True)
        for i in range(B):
            p = mix(cond[out[i, parent] - sp.x_min[parent]])
            out[i, child] = sp.x_min[child] + rng.choice(len(p), p=p)
    return out


# ── 9) xgb_tr ────────────────────────────────────────────────────────────
@simple_algorithm("xgb_tr",
                  state={"models": None, "seen_n": 0, "radius": 8, "succ": 0,
                         "fail": 0, "traj": 0, "reseed": False, "rounds": 0,
                         "seen": None},
                  n_startup=30, batch=4, n_cand=300, kappa=1.0, r_init=8,
                  r_max=15, succ_tol=3, fail_tol=8, refit=4, n_ens=4,
                  max_train=4000)
def xgb_tr(data):
    """XGB 앙상블 + 해밍 신뢰영역. UCB(μ+κσ) 로 후보 선택."""
    X, s, st, rng, ctx, cfg = (data["X"], data["scores"], data["state"],
                               data["rng"], data["ctx"], data["cfg"])
    sp, n = ctx.space, len(s)
    if st["seen"] is None:
        st["seen"] = set()

    # ── 새 관측 반영: 기관측 set + 신뢰영역 반경 ──
    if n > st["seen_n"]:
        new_idx = np.arange(st["seen_n"], n)
        for i in new_idx:
            st["seen"].add(X[i].tobytes())
        st["rounds"] += 1
        traj = np.arange(st["traj"], n)
        if len(traj) and n >= cfg["n_startup"]:
            best_i = traj[int(np.argmax(s[traj]))]
            if best_i in new_idx:                          # 개선
                st["succ"] += 1
                st["fail"] = 0
                if st["succ"] >= cfg["succ_tol"]:
                    st["radius"] = min(st["radius"] * 2, cfg["r_max"])
                    st["succ"] = 0
            else:                                          # 정체
                st["fail"] += 1
                st["succ"] = 0
                if st["fail"] >= cfg["fail_tol"]:
                    st["fail"] = 0
                    r = st["radius"] // 2
                    if r < 1:                              # 수렴 → restart
                        st["radius"], st["traj"] = cfg["r_init"], n
                        st["reseed"] = True
                    else:
                        st["radius"] = r
        # ── 앙상블 재학습 (주기적) ──
        interval = max(cfg["refit"], n // 2000)
        if n >= cfg["n_startup"] and (st["models"] is None
                                      or st["rounds"] % interval == 0):
            from xgboost import XGBRegressor
            Xt, stt = X, s
            if len(s) > cfg["max_train"]:                  # 상한 초과 시 서브샘플
                half = cfg["max_train"] // 2
                order = np.argsort(s)[::-1]
                keep = np.concatenate(
                    [order[:half], rng.choice(order[half:], size=half, replace=False)])
                Xt, stt = X[keep], s[keep]
            st["models"] = [
                XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.1,
                             subsample=0.7, colsample_bytree=0.8, n_jobs=2,
                             verbosity=0, random_state=int(rng.integers(2**31)))
                .fit(Xt.astype(np.float32), stt.astype(np.float32))
                for _ in range(cfg["n_ens"])]
        st["seen_n"] = n

    if n < cfg["n_startup"] or st["models"] is None or st["reseed"]:
        st["reseed"] = False
        return ctx.sample(rng, cfg["batch"])

    traj = np.arange(st["traj"], n)
    inc = X[traj[int(np.argmax(s[traj]))]]
    R = max(1, int(st["radius"]))
    cands = np.tile(inc, (cfg["n_cand"], 1))               # 신뢰영역 안 후보
    for i in range(cfg["n_cand"]):
        d = int(rng.integers(1, R + 1))
        for c in rng.choice(sp.n_cols, size=d, replace=False):
            lo, hi = int(sp.x_min[c]), int(sp.x_max[c])
            if rng.random() < 0.7:
                cands[i, c] = np.clip(cands[i, c] + rng.choice([-1, 1]), lo, hi)
            else:
                cands[i, c] = rng.integers(lo, hi + 1)
    P = np.stack([m.predict(cands.astype(np.float32)) for m in st["models"]])
    acq = P.mean(axis=0) + cfg["kappa"] * P.std(axis=0)

    seen, out = set(st["seen"]), []
    for i in np.argsort(acq)[::-1]:
        key = cands[i].tobytes()
        if key not in seen:
            seen.add(key)
            out.append(cands[i])
        if len(out) == cfg["batch"]:
            break
    while len(out) < cfg["batch"]:
        out.append(ctx.sample(rng, 1)[0])
    return out



# ──────────────────────────────────────────────────────────────────────────────
# 클래스로 남긴 2종 — 상태 기계가 커서 함수 하나로는 오히려 읽기 나쁘다.
# (blockwise_coord: phase/sweep 커서 + 캐시 · gomea_block: FOS 순회 상태)
# 계약은 함수판과 완전히 같다 — init_state / ask / _update.
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
            batch = self.space.sample(rng, self.pop_size)  # 초기 population
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
        batch = self.space.sample(rng, 1)
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

OPTIMIZERS[BlockwiseCoordinateOptimizer.name] = BlockwiseCoordinateOptimizer
OPTIMIZERS[GOMEABlockOptimizer.name] = GOMEABlockOptimizer


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="optimizer — 자가 점검 또는 프로세스 분리 스텝")
    _ap.add_argument("--plugin", action="append", default=[], metavar="MODULE",
                     help="알고리즘이 정의된 모듈 (예: algo_template). 반복 지정 가능")
    _ap.add_argument("--serve-step", action="store_true",
                     help="파일 기반 프로세스 분리 실행의 optimizer 한 스텝을 수행")
    # choices 를 걸지 않는다 — 플러그인 알고리즘은 파서 생성 후에 등록되므로
    # argparse 가 먼저 거부해 버린다. 검증은 플러그인 로드 뒤에 직접 한다.
    _ap.add_argument("--optimizer", default="random")
    _ap.add_argument("--dir", type=str, default=None, help="교환 디렉토리")
    _ap.add_argument("--seed", type=int, default=0)
    _ap.add_argument("--budget", type=int, default=780)
    _args = _ap.parse_args()

    _added = load_plugins(_args.plugin)
    if _args.optimizer not in OPTIMIZERS:
        raise SystemExit(
            f"알 수 없는 optimizer {_args.optimizer!r}. 사용 가능: "
            f"{sorted(OPTIMIZERS)}"
            + (f" (플러그인 {_args.plugin} 이 추가한 것: {_added})" if _added else ""))

    if _args.serve_step:  # runner 가 서브프로세스로 호출하는 경로
        assert _args.dir, "--serve-step 에는 --dir 필요"
        result = serve_step(_args.optimizer, _args.dir, _args.seed, _args.budget)
        print(f"[opt-step] {_args.optimizer} → {result}")
        raise SystemExit(0)

    # 인자 없음 → 자가 점검: 모든 optimizer 가 ask-tell 사이클을 돌 수 있는지 확인.
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
