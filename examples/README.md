# examples/ — 참조 아티팩트 (정답 산출물)

이 디렉토리는 실제 파일 프로토콜을 그대로 돌린 벤치마크 run 의 산출물이다.
Agent 가 만든 시스템이 **같은 파일들을 같은 형식으로** 생성하는지 대조하는
정답 세트로 쓴다.

## bm1_easy_ga_seed0/

`ga` optimizer × `bm1_easy` × seed 0 × budget 780 의 산출물.
**프로세스 분리 실행**으로 생성됐다 — runner 가 optimizer 와 calculator 를
별도 서브프로세스로 번갈아 띄우고 이 디렉토리의 파일로만 통신했다
(39 handshake 라운드 = 배치 20 × 39). 재현:

```
python runner.py --optimizer ga --benchmark bm1_easy --budget 780 \
                 --separate examples/bm1_easy_ga_seed0
```

| 파일 | 내용 | 형식 |
|---|---|---|
| `x.txt` | **마지막** 배치의 후보 X (eval_index=760, 20×30) | 텍스트, `# eval_index=` 헤더 + 행별 signed 정수 |
| `y_raw.bin` | **마지막** 배치의 관측 원형 (마스크 2장 + 스칼라 2개) | 바이너리, int64 헤더 + uint8 마스크 + float64 스칼라 |
| `history.jsonl` | 전체 히스토리 780 평가 (append-only) | jsonl, 한 줄 = tell 한 번 `{eval_index, X, y_raw}` |
| `state.pkl` | 최종 optimizer 상태 (알고리즘 상태 + RNG + 스케일러 + 점수 캐시) | pickle dict |

주의: `x.txt` / `y_raw.bin` 은 프로세스 간 교환 파일이라 **매 스텝 덮어쓴다** —
저장된 것은 마지막(39번째) 배치다. `history.jsonl` 만 전체 이력을 담는다.

### 재현 / 검증

```python
from pathlib import Path
from optimizer import read_x, read_y_raw, load_history, load_state
from space import SearchSpace

d = Path("examples/bm1_easy_ga_seed0")
ss = SearchSpace()

X, idx = read_x(d / "x.txt", space=ss)          # (20, 30), idx == 760
raw, _ = read_y_raw(d / "y_raw.bin")            # mask1/mask2 (20,128,128), y13/y23 (20,)
Xh, Yh = load_history(d / "history.jsonl", ss)  # (780, 30), (780, 6)
st = load_state(d / "state.pkl", d / "history.jsonl", ss)  # n_evals == 780
```

이 run 의 최종 drive best score ≈ 0.7196 (per-run 스케일러 기준).
