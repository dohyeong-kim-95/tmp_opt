"""accept.py — 시뮬레이션 환경의 완료조건 검사.

반응표면이 "실제 TEST 의 대역"으로 쓸 만한지를 네 가지로 판정한다. 각 항목은
주장이 아니라 **반증 가능한 측정**이다 — 시뮬레이션 장치(`SimulatedInstrument`)
를 참값으로 두고, 표면의 출력과 참값을 직접 대조한다.

  (1) 근방에 샘플이 충분하면 boolean array 와 스칼라 y 를 신뢰성 있게 내는가
      → 관측 근처 X 에서 표면 출력 vs 참값 오차가 관측 노이즈 수준인가

  (2) 근방에 샘플이 없으면 "대변하지 못한다"고 근거를 들어 설명하는가
      → no_data 로 표시하고, 그 판단이 옳았음을 참값으로 확인.
        (게이트를 무시하고 억지로 보간했다면 실제로 크게 틀렸어야 한다)

  (3) 최적화 알고리즘과 roundtrip 이 되는가
      → in-process 전 optimizer 완주 + 프로세스 분리 파일 교환 + 체크포인트 재개

  (4) 샘플이 추가되면 즉시 반영되는가
      → no_data 이던 X 를 측정해 append → 재로드 시 exact 로 바뀌고 값이 일치

실행:
    python accept.py                 # 전체 (프로세스 분리 포함, 수십 초)
    python accept.py --quick         # 프로세스 분리/전 optimizer 생략
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from calculator import (OBJECTIVE_NAMES, SurfaceCalculator,
                        load_observations, save_observations)
from make_dataset import SimulatedInstrument, build_dataset
from optimizer import OPTIMIZERS, convert_y_raw, load_history
from runner import run_single, run_separated
from space import SearchSpace

_HERE = Path(__file__).resolve().parent
_PX = [0, 1, 3, 4]  # 픽셀 목적 (나머지 [2, 5] 는 스칼라)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def _mutate(x, n_flip, space, rng):
    v = np.asarray(x).copy()
    for c in rng.choice(space.n_cols, n_flip, replace=False):
        v[c] = rng.integers(space.x_min[c], space.x_max[c] + 1)
    return v


# ──────────────────────────────────────────────────────────────────────────────

def criterion_1(surf, inst, obs_X, space, rng, n=120) -> bool:
    """근방 샘플이 충분할 때 마스크·스칼라를 신뢰성 있게 내는가."""
    print("\n[1] 근방에 샘플이 있을 때 — 신뢰성 있게 생성하는가")
    # 관측에서 1~2 컬럼만 흔든 점들 (표면이 커버한다고 주장할 영역)
    probe = np.array([_mutate(obs_X[rng.integers(len(obs_X))],
                              int(rng.integers(1, 3)), space, rng) for _ in range(n)])
    _, status, info = surf.predict(probe)
    got_raw = surf.evaluate(probe, noisy=False)
    got = convert_y_raw(got_raw)
    truth_raw = inst.evaluate(probe, noisy=False)
    true = convert_y_raw(truth_raw)

    # 노이즈 바닥 (같은 X 반복측정). 픽셀 목적은 결정적이라 sd=0 —
    # 그 경우의 바닥은 격자 양자화 단위인 1px 로 본다.
    rep = convert_y_raw(inst.evaluate(np.repeat(probe[:1], 30, axis=0)))
    floor = rep.std(axis=0, ddof=1)
    floor[_PX] = np.maximum(floor[_PX], 1.0)
    span = true.max(axis=0) - true.min(axis=0) + 1e-12

    # **판정은 목적 그룹별로** — mask1/y13 은 group1 상태, mask2/y23 은 group2 상태가
    # 지배한다. 한 그룹이 no_data 인 점의 다른 그룹 출력까지 섞으면 측정이 무의미하다.
    ok = True
    for gi, (g_objs, mkey) in enumerate(((( 0, 1, 2), "mask1"), ((3, 4, 5), "mask2"))):
        st = status[:, gi]
        for tag in ("exact", "exact_block", "interp"):
            sel = np.flatnonzero(st == tag)
            if len(sel) == 0:
                continue
            err = np.abs(got[sel][:, g_objs] - true[sel][:, g_objs])
            rel = (err / span[list(g_objs)]).mean(axis=0)
            ious = [_iou(got_raw[mkey][i], truth_raw[mkey][i]) for i in sel[:60]]
            names = [OBJECTIVE_NAMES[k] for k in g_objs]
            print(f"    group{gi+1} {tag:<12} n={len(sel):>3}  "
                  f"상대오차 " + " ".join(f"{n_}={r:.3f}" for n_, r in zip(names, rel))
                  + f"  |  마스크 IoU 평균 {np.mean(ious):.3f} 최소 {np.min(ious):.3f}")
            if tag in ("exact", "exact_block"):
                ok &= np.mean(ious) > 0.98 and rel.max() < 0.02   # 재생은 거의 정확해야
            else:
                ok &= np.mean(ious) > 0.80 and rel.max() < 0.10   # 보간은 근사면 충분
    cov = np.array([s != "no_data" for s in status[:, 0]]).mean()
    print(f"    group1 커버율 {cov:.0%} · 노이즈 바닥 "
          + " ".join(f"{n_}={floor[k]:.3g}" for k, n_ in enumerate(OBJECTIVE_NAMES)))
    ok &= cov >= 0.5
    print(f"    → {'PASS' if ok else 'FAIL'} "
          f"(재생: IoU>0.98·상대오차<2% / 보간: IoU>0.80·상대오차<10%)")
    return bool(ok)


def criterion_2(surf, inst, obs_X, space, rng, n=200) -> bool:
    """근방에 샘플이 없을 때 — 근거를 들어 '대변 못 한다'고 말하는가."""
    print("\n[2] 근방에 샘플이 없을 때 — 근거를 들어 설명하는가")
    far = space.sample(rng, n)
    _, status, info = surf.predict(far)
    nod = np.array([s == "no_data" for s in status[:, 0]])
    print(f"    무작위 {n}점 중 group1 no_data {int(nod.sum())} "
          f"(최근접 해밍거리 중앙값 {np.median(info['d_min'][:, 0]):.3f}, "
          f"게이트 {surf.d_gate:.3f})")

    # 판단이 옳았는가 — 게이트를 열고 억지 보간했을 때의 오차를,
    # **커버리지 안 보간 오차**와 비교한다. 절대 임계는 목적 스케일에 좌우되지만
    # 이 비율은 "거부한 영역이 실제로 더 못 맞는가"라는 주장 그 자체다.
    forced = SurfaceCalculator(surf._X, surf._Y,
                               {"mask1": surf._masks["mask1"],
                                "mask2": surf._masks["mask2"]},
                               d_gate=1.0, spread_gate=1e9)  # 게이트 개방
    true = convert_y_raw(inst.evaluate(far, noisy=False))
    forced_y = convert_y_raw(forced.evaluate(far, noisy=False))
    span = true.max(axis=0) - true.min(axis=0) + 1e-12
    rel_out = (np.abs(forced_y - true) / span)[nod].mean()

    # 기준선: 커버리지 안(관측 1~2컬럼 이웃)에서의 보간 오차
    near = np.array([_mutate(obs_X[rng.integers(len(obs_X))], 2, space, rng)
                     for _ in range(120)])
    _, st_in, _ = surf.predict(near)
    sel = np.flatnonzero([s == "interp" for s in st_in[:, 0]])
    t_in = convert_y_raw(inst.evaluate(near[sel], noisy=False))
    g_in = convert_y_raw(surf.evaluate(near[sel], noisy=False))
    rel_in = (np.abs(g_in - t_in) / span).mean()
    ratio = rel_out / max(rel_in, 1e-12)
    print(f"    커버리지 안 보간 상대오차 {rel_in:.3f} (n={len(sel)}) vs "
          f"게이트 열고 억지 보간 {rel_out:.3f} → {ratio:.1f}배")

    # 근거 제시: explain() 이 최근접 관측·거리·게이트를 실제로 내놓는가
    txt = surf.explain(far[int(np.flatnonzero(nod)[0])], k=2)
    need = ("no_data", "d_min", "게이트")
    has_evidence = all(t in txt for t in need) and "d=" in txt
    print("    explain() 출력 예시:")
    for line in txt.splitlines()[:4]:
        print("      " + line[:96])

    ok = nod.mean() > 0.5 and has_evidence and ratio > 2.0
    print(f"    → {'PASS' if ok else 'FAIL'} (no_data {nod.mean():.0%}, 근거 제시, "
          f"거부 영역 오차가 커버 영역의 {ratio:.1f}배 > 2.0)")
    return bool(ok)


def criterion_3(obs_path, quick: bool) -> bool:
    """최적화 알고리즘과 roundtrip 이 되는가."""
    print("\n[3] 최적화 알고리즘과 roundtrip")
    names = list(OPTIMIZERS) if not quick else ["random", "sa", "xgb_tr"]
    done = []
    for name in names:
        s = SurfaceCalculator.from_jsonl(obs_path, policy="pessimistic")
        r = run_single(name, s, seed=0, budget=60, source=str(obs_path))
        assert len(r.X) == 60 and r.Y_raw.shape == (60, 6)
        done.append(name)
    print(f"    in-process 완주 {len(done)}/{len(names)}")

    ok_sep = True
    if not quick:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            r = run_separated("ga", obs_path, seed=0, budget=20,
                              exchange_dir=d, verbose=False)
            files = {f.name for f in d.iterdir()}
            need = {"x.txt", "y_raw.bin", "history.jsonl", "state.pkl",
                    "done", "coverage.jsonl"}
            missing = need - files
            X, Y = load_history(d / "history.jsonl", space=SearchSpace())
            cov = sum(1 for _ in (d / "coverage.jsonl").open())
            ok_sep = not missing and len(X) == 20 and cov == 20
            print(f"    프로세스 분리: 교환파일 {sorted(files)}")
            print(f"      history {len(X)}행, coverage.jsonl {cov}줄"
                  + (f", 누락 {missing}" if missing else ""))

        # 체크포인트 재개 = 무중단과 동일 궤적
        with tempfile.TemporaryDirectory() as td:
            ck = Path(td) / "ck"
            s = SurfaceCalculator.from_jsonl(obs_path)
            a = run_single("sa", s, seed=0, budget=40, checkpoint_dir=ck)
            Xc, Yc = load_history(ck / "history.jsonl", space=SearchSpace())
            same = np.array_equal(Xc, a.X)
            print(f"    체크포인트 history 왕복 일치: {same}")
            ok_sep &= bool(same)

    ok = len(done) == len(names) and ok_sep
    print(f"    → {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def criterion_4(obs_path, inst, space, rng) -> bool:
    """샘플이 추가되면 즉시 반영되는가."""
    print("\n[4] 샘플 추가 시 즉시 반영")
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "obs.jsonl"
        work.write_text(Path(obs_path).read_text())
        surf0 = SurfaceCalculator.from_jsonl(work)
        n0 = len(surf0._X)

        # 현재 커버리지 밖인 점 하나를 고른다
        cand = space.sample(rng, 300)
        _, st, _ = surf0.predict(cand)
        far_i = int(np.flatnonzero([s == "no_data" for s in st[:, 0]])[0])
        x_new = cand[far_i:far_i + 1]
        print(f"    대상 X: 추가 전 status = {st[far_i, 0]} / {st[far_i, 1]}")

        # 실제로 측정해서 append
        raw = inst.evaluate(x_new)
        Y_new = convert_y_raw(raw)
        save_observations(work, x_new, Y_new, block=["added"],
                          masks={"mask1": raw["mask1"], "mask2": raw["mask2"]},
                          append=True)

        surf1 = SurfaceCalculator.from_jsonl(work)
        _, st1, _ = surf1.predict(x_new)
        back = convert_y_raw(surf1.evaluate(x_new, noisy=False))
        mask_same = np.array_equal(surf1.evaluate(x_new, noisy=False)["mask1"][0],
                                   raw["mask1"][0])
        print(f"    추가 후 status = {st1[0, 0]} / {st1[0, 1]}, "
              f"관측 수 {n0} → {len(surf1._X)}")
        print(f"    값 일치: 측정치 {np.array_equal(back, Y_new)}, "
              f"마스크 바이트 동일 {mask_same}")

        # 기존 관측이 훼손되지 않았는가
        old_ok = np.array_equal(surf1._X[:n0], surf0._X) and \
            np.array_equal(surf1._Y[:n0], surf0._Y)
        # 주변도 커버리지로 편입되었는가
        near = np.array([_mutate(x_new[0], 1, space, rng) for _ in range(30)])
        _, st_b, _ = surf0.predict(near)
        _, st_a, _ = surf1.predict(near)
        gained = int(sum(1 for b, a in zip(st_b[:, 0], st_a[:, 0])
                         if b == "no_data" and a != "no_data"))
        print(f"    기존 {n0}행 무결: {old_ok} · 주변 1-hop 30점 중 "
              f"커버리지 편입 {gained}개")

        ok = (st1[0, 0] == "exact" and st1[0, 1] == "exact"
              and np.array_equal(back, Y_new) and mask_same and old_ok
              and gained > 0)
    print(f"    → {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def main() -> None:
    ap = argparse.ArgumentParser(description="시뮬레이션 환경 완료조건 검사")
    ap.add_argument("--obs", type=Path, default=_HERE / "obs.jsonl",
                    help="관측 파일 (없으면 즉석 생성)")
    ap.add_argument("--quick", action="store_true",
                    help="프로세스 분리·전 optimizer 검사 생략")
    ap.add_argument("--instrument-seed", type=int, default=303)
    args = ap.parse_args()

    space = SearchSpace()
    inst = SimulatedInstrument(seed=args.instrument_seed)
    obs_path = args.obs
    tmp = None
    if not obs_path.exists():
        tmp = tempfile.TemporaryDirectory()
        obs_path = Path(tmp.name) / "obs.jsonl"
        p = build_dataset(inst, space)
        save_observations(obs_path, p["X"], p["Y"], block=p["block"],
                          masks={"mask1": p["mask1"], "mask2": p["mask2"]})
        print(f"[준비] 관측 파일이 없어 즉석 생성: {len(p['X'])}점")

    d = load_observations(obs_path)
    surf = SurfaceCalculator.from_jsonl(obs_path)
    print(f"[환경] 관측 {len(d['X'])}점, 마스크 "
          f"{'있음' if d['mask1'] is not None else '없음'}, "
          f"d_gate={surf.d_gate}, spread_gate={surf.spread_gate}")

    rng = np.random.default_rng(7)
    res = {
        "(1) 근방 샘플 충분 → 신뢰성 있는 생성": criterion_1(surf, inst, d["X"], space, rng),
        "(2) 근방 샘플 부족 → 근거 있는 거부": criterion_2(surf, inst, d["X"], space, rng),
        "(3) 최적화 알고리즘 roundtrip": criterion_3(obs_path, args.quick),
        "(4) 샘플 추가 즉시 반영": criterion_4(obs_path, inst, space, rng),
    }
    print("\n" + "=" * 62)
    for k, v in res.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    n_ok = sum(res.values())
    print(f"  완료조건 {n_ok}/{len(res)}")
    print("=" * 62)
    if tmp is not None:
        tmp.cleanup()
    raise SystemExit(0 if n_ok == len(res) else 1)


if __name__ == "__main__":
    main()
