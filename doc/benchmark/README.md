# doc/benchmark — 벤치마크 설계 스터디

bm1~bm5 가 X → raw_y 를 어떻게 만드는지, 난이도를 어떤 factor 로 조절하는지 정리.

- [`stage.md`](stage.md) — **먼저 읽기.** 공통 파이프라인(X→u→latent→raw_y),
  난이도 factor 6종(단봉/블록내상호작용/교차블록/기만/rugged/목적충돌),
  벤치마크별 배합표, 실측 결과.
- [`bm1_easy.md`](bm1_easy.md) — 분리가능 단봉 (기준선)
- [`bm2_medium.md`](bm2_medium.md) — 블록 내 상호작용 + 완만한 다봉
- [`bm3_hard.md`](bm3_hard.md) — 교차-블록 + 기만 + rugged 복합
- [`bm4_deceptive.md`](bm4_deceptive.md) — 진짜 trap 지배 (탈출 능력)
- [`bm5_epistasis.md`](bm5_epistasis.md) — 강한 상호작용 지배 (비분리)

구현: `calculator.py` (각 `Benchmark*` 클래스의 `_latent`).
관측 장치(타원 마스크)·측정: `calculator.py` 베이스 + `optimizer.py` 의 `convert_y_raw`.
