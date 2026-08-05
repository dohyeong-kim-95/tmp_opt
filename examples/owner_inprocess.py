"""owner_inprocess.py — **calculator 가 optimizer 를 소유하는** 최소 코드.

당신(측정하는 쪽)이 루프의 주인이다. optimizer 는 "다음에 뭘 재볼까"에만 답하고
측정 함수를 절대 부르지 않는다. 이게 ask-and-tell 이다.

    python examples/owner_inprocess.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calculator import SurfaceCalculator          # ← 당신의 측정기로 교체
from runner import Session

BUDGET = 800

# 측정기 (실제로는 장비/외부 서비스)
calc = SurfaceCalculator.from_jsonl(
    Path(__file__).resolve().parent.parent / "obs.jsonl", policy="pessimistic")

# ── 여기가 전부 ──────────────────────────────────────────────────────────
sess = Session("xgb_tr", budget=BUDGET, seed=0)

while not sess.done:
    x = sess.next_x()                  # (30,) 다음에 측정할 X
    y_raw = calc.evaluate(x[None, :])  # ← 측정. 루프의 주인은 당신
    sess.report(x, y_raw)              # 결과 통보

best = sess.best()
# ────────────────────────────────────────────────────────────────────────

print(f"{sess.n_evals} evals, best_score={best['best_score']:.4f}")
print(f"best_x = {best['best_x'].tolist()}")
print(f"best_y = {best['best_y'].round(4).tolist()}")
