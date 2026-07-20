"""protocol.py — 모듈 간 파일 교환 형식의 레퍼런스 구현 (x.txt).

optimizer → calculator 로 넘어가는 후보 X 는 텍스트 파일 하나다:

    # eval_index=123
    [15,0,0,0,-1,-3,5,3]
    [-15,0,3,0,-1,3,2,9]

- 1행: `# eval_index=<int>` 헤더 — 이 배치의 첫 평가가 갖는 전역 평가 카운터.
  calculator 는 노이즈 시딩 (noise_seed, eval_index) 에 쓰고, optimizer 는
  회신(y_raw)이 자기가 낸 배치와 대응하는지 검증하는 데 쓴다.
- 2행부터: 한 줄 = 해 하나. 대괄호로 감싼 콤마 구분 정수 (signed).
  X 는 정수라 텍스트 왕복이 무손실이다.

규율:
- **원자적 쓰기**: 임시 파일에 쓴 뒤 os.replace — 반쯤 쓰인 파일을 읽는
  레이스를 막는다 (POSIX rename 은 원자적).
- **fail-loud**: 헤더 누락, 정수 아님, 행 길이 불일치, (space 를 주면)
  범위 밖 값 — 전부 즉시 raise. 기본값 대체·건너뛰기 금지.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


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


def read_x(path: str | Path, space=None) -> tuple[np.ndarray, int]:
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


if __name__ == "__main__":
    # 자가 점검: 왕복 무손실 + fail-loud 확인
    import tempfile

    from space import SearchSpace

    ss = SearchSpace()
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.txt"

        # 왕복: signed 값 그대로 보존
        X = ss.sample(rng, n=10)
        write_x(p, X, eval_index=123)
        X2, idx = read_x(p, space=ss)
        assert np.array_equal(X, X2) and idx == 123
        assert (X2.min(axis=0) < 0).any(), "signed 값이 실제로 왕복되어야 함"

        # fail-loud: 헤더 누락 / 정수 아님 / 행 길이 불일치 / 범위 밖
        for content in [
            "[1,2,3]\n",                                # 헤더 없음
            "# eval_index=0\n[1,2.5,3]\n",              # 정수 아님
            "# eval_index=0\n[1,2]\n[1,2,3]\n",         # 길이 불일치
            "# eval_index=0\n[" + ",".join(["999"] * ss.n_cols) + "]\n",  # 범위 밖
        ]:
            p.write_text(content)
            try:
                read_x(p, space=ss)
            except ValueError:
                pass
            else:
                raise AssertionError(f"raise 됐어야 함: {content!r}")

        print(f"[OK] protocol — x.txt 왕복 무손실 ({X.shape}), fail-loud 4종 통과")
