# 작업 규칙 — 이 저장소를 참조하는 Agent 필독

이 문서는 tmp_opt 를 레퍼런스로 삼아 별도 시스템을 구현하는 Agent 에게
적용되는 작업 규칙이다. 아래 규칙은 전부 **기계적으로 검수**되며,
자가보고는 검수를 대체하지 않는다.

## 0. 이 저장소의 지위

tmp_opt 는 **읽기 전용 참조 구현**이다. 여기서 가져갈 것은 코드가 아니라
**계약**이다: ask-tell 시그니처, 점수 파이프라인의 수학(robust p5–p95 →
sense 통일 → chebyshev), 상태 dict 불변식, 체크포인트 정합성 규칙,
파일 교환 형식. 이 계약들을 **합의된 파일 구조 안에** 다시 구현하라.

## 1. 실행 의존 금지

tmp_opt 에 대한 어떤 실행 의존도 만들지 않는다:
- `import` (직접/간접), `sys.path` 추가, 서브프로세스 호출 — 금지
- 파일을 복사한 뒤 경로/이름만 바꾸는 것 — 금지

이유: 이 프로젝트는 사람 협업자들과 **파일 구조가 합의**되어 있다.
구현은 합의된 파일들 안에 존재해야 하며, 그 파일들이 팀이 리뷰하고
유지보수하는 대상이다. tmp_opt 를 import 하면 합의된 코드는 더미가 되고
산출물로서 무효다.

## 2. 성공의 정의

> **tmp_opt 디렉토리가 존재하지 않는 환경에서, 합의된 파일 구조만으로
> 자가 점검과 벤치마크가 완주하는 것.**

검수는 실제로 그 조건(레퍼런스 제거 환경)에서 실행된다. import 의존이
있으면 그 자리에서 ImportError 로 불합격이다.

## 3. 완전성 게이트 — "일부 구현 + 통과 주장" 금지

optimizer 는 **11종 전원**이 요구사항이다. 판정문에 완전성이 포함된다:

```python
EXPECTED = {"random", "blockwise_coord", "ga", "sa", "pso", "aco",
            "tpe", "xgb_surrogate", "eda_tree", "gomea_block", "xgb_tr"}
assert set(OPTIMIZERS) == EXPECTED   # 하나라도 빠지면 불합격
```

- 계약 테스트 스위트는 이 명단으로 **parametrize** 된다 — 미구현은
  skip 이 아니라 **fail** 이다.
- 벤치마크 "통과" 판정 = 순위표에 11개 이름 전원 출석 **그리고**
  random 이 압도적 최하 (README: "random 을 못 이기면 문제가 있는 것").
- 격자 결과 파일(matrix.json)의 이름 목록이 검수 대상이다.

## 3.5 파일 구조 게이트 — "한 파일에 전부" 금지

합의된 파일 구조는 요구사항이지 제안이 아니다. 각 파일이 **존재**하고,
각 책임이 **그 파일 안에** 있어야 한다:

| 파일 | 책임 | 있으면 안 되는 것 |
|---|---|---|
| `space.py` | 탐색 공간 표준 명세 | 문제/점수/알고리즘 |
| `calculator.py` | X → y_raw (문제 정의·노이즈) | 점수·스케일링 개념 일체 |
| `optimizer.py` | 알고리즘 + score 파이프라인 + 셸 | 실행 루프, 플로팅 |
| `runner.py` | ask→평가→tell 반복 기계 | 점수 계산, 비교, 시각화 |
| `benchmark.py` | 여러 run 비교·랭킹·시각화 | 알고리즘 구현 |

기계 검수:
1. 5개 파일 전부 존재 + 각 파일의 진입점(`python <file>` 자가 점검/CLI)이
   단독으로 동작
2. 공개 API 가 명세된 파일에 존재:
   `from space import SearchSpace` / `from calculator import BENCHMARKS` /
   `from optimizer import OPTIMIZERS, RobustScaler, SCORERS` /
   `from runner import run_single` / `python benchmark.py --matrix ...`
3. 책임 경계 grep: runner 에 스케일링·scalarization 없음, calculator 에
   점수 개념 없음, optimizer 에 matplotlib 없음, benchmark 에 알고리즘 없음
4. 벤치마크 결과는 **benchmark.py 진입점에서 생성된 것만 인정**
   (실행 커맨드와 산출 경로가 검수 대상)

한 파일에 욱여넣은 것도, 파일만 만들어 두고 빈 껍데기인 것도(로직이 다른
파일에 있으면 3번 grep 에 걸린다) 불합격이다.

## 3.6 파일 교환 게이트 — "데이터가 안 남는" 벤치마크 금지

모듈 간 계약은 **파일을 주고받으며 다음 iteration 을 만드는 것**이다. 벤치마크가
돌았는데 x.txt / y_raw.bin / history.jsonl / state.pkl 이 **한 번도 디스크에
닿지 않았다면**, 그것은 파일 교환이 아니라 in-process 함수 호출이며 계약
미구현이다. (참고: 레퍼런스는 두 모드를 제공한다 — 빠른 in-process 와
`--separate` 프로세스 분리. 계약이 파일 매개라면 후자가 기준이다.)

기계 검수:
1. 프로세스 분리 실행 중 교환 디렉토리를 감시 → x.txt / y_raw.bin 이 매 라운드
   갱신되고 헤더 `eval_index` 가 배치 크기만큼 전진해야 한다 (churn 지문).
2. **강한 강제**: optimizer 와 calculator 가 **별도 프로세스**로 도는가.
   같은 프로세스면 공유 메모리로 우회 가능 — 별도 프로세스면 파일이 유일한
   통신 수단이라 우회가 물리적으로 불가능하다.
3. 완주 후 산출물이 레퍼런스 reader(`read_x`/`read_y_raw`/`load_history`/
   `load_state`)로 그대로 읽히고 정합해야 한다. 정답 세트: `examples/`.

"벤치마크가 통과했다"의 증거는 콘솔 로그가 아니라 **디스크에 남은(또는 run
중 churn 한) 교환 파일**이다.

## 4. 보고 양식

"통과했다" / "완료했다" 문장은 보고로 인정하지 않는다. 보고는 분수로:

```
구현        N / 11
계약 통과   N / 11   (pytest -v 로그 첨부)
격자 출석   N / 11   (matrix.json 첨부)
random 순위 k / 11
```

## 5. 태스크 단위

한 번에 **optimizer 하나**. "다음은 X. 계약 스위트 green + 격자 출석
확인 후 다음"의 반복이다. 여러 개를 한꺼번에 진행하지 않는다.

## 6. 금지 목록 (발견 즉시 diff 폐기)

- **테스트 완화**: 허용오차·임계·슬랙을 낮춰 통과시키는 것
- **silent fallback**: 실패·미구현을 랜덤/기본값으로 조용히 대체하는 것.
  에러는 삼키지 말고 즉시 raise 하라 — silent fallback 은 그 자체가 버그다
- **측정/판정 정의 변경**: convert_y_raw 의 측정 정의, 점수 파이프라인의
  수학을 바꾸는 것은 버그 수정이 아니라 문제 재정의다 — 감독 승인 없이 금지
- **acceptance 테스트 수정**: 테스트는 감독 소유다. production 만 수정하라

## 7. 이전에 실제로 있었던 일 (반복하지 말 것)

1. tmp_opt 를 import 해서 벤치마크를 돌리고 "통과"라고 보고 → 폐기됨 (§1)
2. optimizer 1개만 구현하고 "벤치마크 통과"라고 보고 → 폐기됨 (§3, §4)
3. runner/calculator/benchmark 를 만들지 않고 optimizer 한 파일에 전부
   욱여넣은 뒤 "벤치마크 통과"라고 보고 → 폐기됨 (§3.5)
4. 벤치마크를 in-process 로만 돌려 x.txt/y_raw.bin/jsonl/pkl 이 아무것도
   생기지 않음 → 파일 교환 계약 미구현 (§3.6)

네 경우 모두 거짓 보고가 아니라 판정문의 빈틈을 통과한 것이었다.
그래서 판정문을 위처럼 고쳤다. 빈틈을 찾는 노력을 구현에 쓰라.
