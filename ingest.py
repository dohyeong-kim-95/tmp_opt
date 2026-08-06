"""ingest.py — 측정 파일 디렉토리 → obs.jsonl 한 파일로 머지.

측정은 **파일 하나에 하나씩** 나온다. 그 파일들을 한 디렉토리에 모아두면 이
도구가 훑어서 표준 레코드로 바꿔 `obs.jsonl` 끝에 이어 붙인다.

    measurements/            data/obs.jsonl
      m0001.npz   ─┐
      m0002.npz   ─┼─ ingest ─→  한 줄 = 한 측정 (append-only)
      m0003.npz   ─┘

**멱등하다.** 이미 들어간 파일은 `src` 로 알아보고 건너뛴다. 그래서 측정이
계속 쌓이는 동안 같은 명령을 몇 번을 돌려도 새로 생긴 것만 들어간다:

    python ingest.py --src measurements/ --out data/obs.jsonl

┌──────────────────────────────────────────────────────────────────────────┐
│ 이음새 — reader                                                           │
│   원본 측정 파일의 형식은 이 리포가 모른다. 아는 척하지 않고 **읽는       │
│   함수 하나**로 분리한다:                                                 │
│                                                                          │
│     def my_reader(path) -> dict:                                         │
│         return {"x":       (30,) 정수,                                    │
│                 "masks":   [mask1, mask2]  # 각각 2D boolean, 크기 달라도 됨│
│                 "scalars": (2,) float}                                    │
│                                                                          │
│   이 함수만 채우면 하류(형식·검증·머지·점수·표면)는 전부 그대로 돈다.     │
│   짜고 나서 `--check <파일>` 로 계약을 먼저 확인할 것 — 머지 전에 틀린    │
│   걸 잡는 게 훨씬 싸다.                                                   │
└──────────────────────────────────────────────────────────────────────────┘

`read_npz` 는 **참고 구현**이다. 실제 형식이 다르면 이걸 본떠 새로 쓰고
`register()` 로 등록한다.

자가 점검:
    python ingest.py --selfcheck
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from record import N_MASKS, N_SCALARS, Blob, Record, append, count, sources, validate
from space import SearchSpace

Reader = Callable[[Path], dict]


# ──────────────────────────────────────────────────────────────────────────────
# reader — 원본 측정 파일 하나를 읽는다 (이음새)
# ──────────────────────────────────────────────────────────────────────────────


def read_npz(path: Path) -> dict:
    """**참고 구현** — `np.savez` 로 저장된 측정 파일 하나.

    기대 키: `x`(30 정수), `mask1`/`mask2`(2D boolean), `scalars`(2 float).
    실제 형식이 다르면 이 함수를 본떠 새로 쓰고 `register()` 로 등록한다.
    """
    with np.load(path) as z:
        return {"x": z["x"], "masks": [z["mask1"], z["mask2"]],
                "scalars": z["scalars"]}


READERS: dict[str, Reader] = {"npz": read_npz}


def register(name: str, reader: Reader) -> None:
    """직접 짠 reader 를 등록한다 (`--reader <name>` 으로 쓰인다)."""
    READERS[name] = reader


def get_reader(name: str) -> Reader:
    if name not in READERS:
        raise ValueError(
            f"알 수 없는 reader {name!r} — {sorted(READERS)}. "
            "원본 형식이 다르면 read_npz 를 본떠 짜고 register() 로 등록할 것")
    return READERS[name]


# ──────────────────────────────────────────────────────────────────────────────
# 계약 검증 — 머지 전에 reader 가 옳은지 확인한다
# ──────────────────────────────────────────────────────────────────────────────


def to_record(raw: dict, src: str, space: SearchSpace) -> Record:
    """reader 의 출력 → Record. 계약 위반은 여기서 전부 fail-loud 로 잡는다."""
    missing = {"x", "masks", "scalars"} - set(raw)
    if missing:
        raise ValueError(f"reader 출력에 키 없음: {sorted(missing)}")

    x = np.asarray(raw["x"])
    if x.shape != (space.n_cols,):
        raise ValueError(f"x 는 {space.n_cols}개여야 함: {x.shape}")
    if not np.issubdtype(x.dtype, np.integer):
        if not np.allclose(x, np.rint(x)):
            raise ValueError(f"x 가 정수가 아님: {x.tolist()[:5]}…")
        x = np.rint(x).astype(np.int64)   # float 로 저장된 정수는 받아준다
    x = x.astype(np.int64)

    masks = list(raw["masks"])
    if len(masks) != N_MASKS:
        raise ValueError(f"masks 는 {N_MASKS}장이어야 함: {len(masks)}장")
    blobs = []
    for k, m in enumerate(masks, start=1):
        m = np.asarray(m)
        if m.ndim != 2:
            raise ValueError(f"mask{k} 는 2차원이어야 함: {m.shape}")
        if m.dtype != bool:
            uniq = np.unique(m)
            if not set(uniq.tolist()) <= {0, 1}:
                raise ValueError(
                    f"mask{k} 가 boolean 이 아니고 0/1 도 아님 — 값 {uniq[:5].tolist()}…")
            m = m.astype(bool)
        blobs.append(Blob.from_mask(m))

    s = np.asarray(raw["scalars"], dtype=np.float64).ravel()
    if s.shape != (N_SCALARS,):
        raise ValueError(f"scalars 는 {N_SCALARS}개여야 함: {s.shape}")
    if not np.isfinite(s).all():
        raise ValueError(f"scalars 에 비유한값: {s.tolist()}")

    return Record(0, x, tuple(blobs), s, src)


def check_reader(reader: Reader, path, *, space: SearchSpace | None = None) -> dict:
    """reader 를 파일 하나에 걸어보고 계약을 확인한다. 머지 전에 쓸 것."""
    space = space or SearchSpace()
    path = Path(path)
    r = to_record(reader(path), path.name, space)
    from record import _check_x
    _check_x(r.x, space)   # 탐색 공간 범위까지 확인
    return {"file": path.name,
            "mask_shapes": [b.shape for b in r.blobs],
            "areas": [b.area for b in r.blobs],
            "fill_rate": [round(b.area / (b.shape[0] * b.shape[1]), 4)
                          for b in r.blobs],
            "n_runs": [len(b.runs) for b in r.blobs],
            "n_rows": [b.height for b in r.blobs],
            "scalars": r.scalars.tolist()}


# ──────────────────────────────────────────────────────────────────────────────
# 머지
# ──────────────────────────────────────────────────────────────────────────────


def ingest(src_dir, out, *, reader: Reader = read_npz, pattern: str = "*",
           space: SearchSpace | None = None, dry_run: bool = False,
           verbose: bool = True) -> dict:
    """`src_dir` 의 측정 파일들을 `out` 에 이어 붙인다 (멱등).

    이미 들어간 파일은 `src` 로 알아보고 건너뛴다. 파일 순서는 이름순으로
    고정해 같은 입력이면 같은 결과가 나오게 한다.

    한 파일이 계약을 어기면 **그 파일만 건너뛰고 나머지는 넣는다** — 측정이
    비싸서 한 점 때문에 전체를 막을 이유가 없다. 건너뛴 목록은 반환값과
    출력에 남으므로 조용히 사라지지 않는다.
    """
    space = space or SearchSpace()
    src_dir, out = Path(src_dir), Path(out)
    if not src_dir.is_dir():
        raise NotADirectoryError(f"측정 디렉토리 없음: {src_dir}")

    files = sorted(p for p in src_dir.glob(pattern) if p.is_file())
    done = sources(out)
    todo = [p for p in files if p.name not in done]

    new: list[Record] = []
    failed: list[tuple[str, str]] = []
    for p in todo:
        try:
            new.append(to_record(reader(p), p.name, space))
        except Exception as e:                      # noqa: BLE001 — 어떤 실패든 기록
            failed.append((p.name, f"{type(e).__name__}: {e}"))

    n_before = count(out)
    n_after = n_before
    if new and not dry_run:
        n_after = append(out, new, space=space)

    rep = {"scanned": len(files), "skipped_done": len(files) - len(todo),
           "ingested": 0 if dry_run else len(new), "failed": failed,
           "n_before": n_before, "n_after": n_after, "dry_run": dry_run}

    if verbose:
        print(f"[머지] {src_dir} → {out}")
        print(f"  파일 {rep['scanned']}개 스캔 · 이미 들어감 {rep['skipped_done']} · "
              f"신규 {len(new)} · 실패 {len(failed)}")
        for name, err in failed[:10]:
            print(f"    ✗ {name} — {err}")
        if len(failed) > 10:
            print(f"    … 외 {len(failed) - 10}개")
        print(f"  {out.name}: {n_before} → {n_after}줄"
              + ("  (dry-run — 쓰지 않음)" if dry_run else ""))
    return rep


# ──────────────────────────────────────────────────────────────────────────────


def _selfcheck() -> None:
    """가짜 측정 파일 디렉토리를 만들어 머지 전 경로를 확인한다."""
    import tempfile

    rng = np.random.default_rng(0)
    ss = SearchSpace()
    H1, W1, H2, W2 = 64, 48, 96, 120      # 두 장의 크기가 서로 다르다

    def _diamond(h, w, ah, aw):
        yy, xx = np.ogrid[:h, :w]
        return ((np.abs(yy - h / 2) / ah) ** 1.5
                + (np.abs(xx - w / 2) / aw) ** 1.5) <= 1.0

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "measurements"
        src.mkdir()
        out = Path(td) / "data" / "obs.jsonl"

        def _write(i, x=None):
            np.savez(src / f"m{i:04d}.npz",
                     x=(ss.sample(rng, 1)[0] if x is None else x),
                     mask1=_diamond(H1, W1, rng.uniform(6, 26), rng.uniform(6, 20)),
                     mask2=_diamond(H2, W2, rng.uniform(6, 42), rng.uniform(6, 54)),
                     scalars=rng.uniform(0, 1, N_SCALARS))

        for i in range(12):
            _write(i)

        # reader 계약을 먼저 확인 (실전에서도 이 순서)
        info = check_reader(read_npz, src / "m0000.npz")
        assert info["mask_shapes"] == [(H1, W1), (H2, W2)]
        print(f"[OK] reader 계약 — 마스크 {info['mask_shapes']}, "
              f"면적 {info['areas']}, run {info['n_runs']}개/{info['n_rows']}행")

        r1 = ingest(src, out, verbose=False)
        assert r1["ingested"] == 12 and r1["n_after"] == 12

        # 멱등: 그대로 다시 돌려도 아무것도 안 들어간다
        r2 = ingest(src, out, verbose=False)
        assert r2["ingested"] == 0 and r2["n_after"] == 12 and r2["skipped_done"] == 12
        print(f"[OK] 멱등 — 재실행 시 신규 0개, {r2['n_after']}줄 유지")

        # 증분: 측정 5개가 더 생기면 그것만 들어간다
        for i in range(12, 17):
            _write(i)
        r3 = ingest(src, out, verbose=False)
        assert r3["ingested"] == 5 and r3["n_after"] == 17 and r3["skipped_done"] == 12
        print(f"[OK] 증분 — 신규 5개만 추가되어 {r3['n_after']}줄")

        # 반복측정(같은 x 를 다시 잰 파일)이 그대로 들어가는가
        x_rep = np.load(src / "m0000.npz")["x"]
        _write(90, x=x_rep)
        r4 = ingest(src, out, verbose=False)
        assert r4["ingested"] == 1
        v = validate(out)
        assert v["n"] == 18 and v["n_unique_x"] == 17 and v["n_repeated"] == 2
        print(f"[OK] 반복측정 — {v['n']}줄, 유일 x {v['n_unique_x']}, 반복 {v['n_repeated']}줄")

        # 깨진 파일 하나가 나머지를 막지 않는가
        np.savez(src / "m0099.npz", x=np.zeros(7), mask1=np.zeros((4, 4)),
                 mask2=np.zeros((4, 4)), scalars=np.zeros(2))
        for i in range(20, 23):
            _write(i)
        r5 = ingest(src, out, verbose=False)
        assert r5["ingested"] == 3 and len(r5["failed"]) == 1
        assert "m0099.npz" == r5["failed"][0][0] and "30개여야" in r5["failed"][0][1]
        print(f"[OK] 부분 실패 — 깨진 1개는 사유와 함께 건너뛰고 나머지 3개 머지 "
              f"({r5['n_after']}줄)")

        # 깨진 파일이 고쳐지면 다음 머지에 들어간다 (건너뛴 게 영구 손실이 아님)
        _write(99)
        r6 = ingest(src, out, verbose=False)
        assert r6["ingested"] == 1 and not r6["failed"]
        print(f"[OK] 복구 — 고친 파일이 다음 머지에 들어감 ({r6['n_after']}줄)")

        # dry-run 은 파일을 건드리지 않는다
        _write(30)
        n = count(out)
        r7 = ingest(src, out, dry_run=True, verbose=False)
        assert count(out) == n and r7["ingested"] == 0
        print(f"[OK] dry-run — {n}줄 그대로")

        # 점수는 머지된 파일에서 언제든 다시 잰다
        import score
        from record import read
        recs = read(out)
        for name in score.SCORERS:
            Y = score.get(name)(recs)
            assert Y.shape == (len(recs), score.get(name).n_obj)
        kb = out.stat().st_size / 1e3
        print(f"[OK] 채점 — {len(recs)}줄에 scorer {len(score.SCORERS)}종 전부 적용 "
              f"(파일 {kb:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="측정 파일 디렉토리 → obs.jsonl 머지 (멱등)")
    ap.add_argument("--src", type=Path, help="측정 파일들이 모인 디렉토리")
    ap.add_argument("--out", type=Path, default=Path("data/obs.jsonl"),
                    help="누적 관측 파일 (여기에 append)")
    ap.add_argument("--reader", default="npz", help=f"reader 이름 — {sorted(READERS)}")
    ap.add_argument("--pattern", default="*", help="파일 glob (예: '*.npz')")
    ap.add_argument("--dry-run", action="store_true", help="쓰지 않고 결과만 본다")
    ap.add_argument("--check", type=Path,
                    help="측정 파일 하나에 reader 를 걸어 계약만 확인한다")
    ap.add_argument("--validate", action="store_true",
                    help="--out 파일의 형식·일관성을 검사한다")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        return
    if args.check is not None:
        info = check_reader(get_reader(args.reader), args.check)
        print(f"[계약 OK] {info['file']}")
        for k, v in info.items():
            if k != "file":
                print(f"  {k:<12} {v}")
        return
    if args.validate:
        rep = validate(args.out)
        print(f"[검증 OK] {args.out}")
        for k, v in rep.items():
            print(f"  {k:<14} {v}")
        return
    if args.src is None:
        ap.error("--src 가 필요하다 (또는 --check / --validate / --selfcheck)")
    ingest(args.src, args.out, reader=get_reader(args.reader),
           pattern=args.pattern, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
