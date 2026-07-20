# bm1_easy — X → raw_y 구성

> 거의 분리가능(separable)한 매끄러운 단봉 지형. 좌표별로 거의 독립이라
> coordinate/greedy 계열이 잘 통해야 "정상"인 기준선 벤치마크.
> `_structure_seed = 101` — 모든 파라미터는 이 시드로 한 번만 생성(재현 가능).

공통 파이프라인(u 매핑, 관측 장치, 은닉 스케일)은 `stage.md` §0 참조.
여기서는 **이 벤치의 latent 함수**만 다룬다.

## latent 함수

목적 k(=1..6)의 latent 값은 **두 성분의 합**이다:

```
latent_k(u) = Σ_c w_kc·(1 − (u_c − p_kc)²)      ① 단봉 골격
            + c_gain_k · c(u)                    ② trade-off
```

### ① 단봉 골격 (F1)

- 각 목적은 자기 그룹 컬럼만 본다 — group1(common+set1)→y11,y12,y13,
  group2(common+set2)→y21,y22,y23. 그 밖 컬럼의 가중치는 0.
- 항 `1 − (u_c − p_kc)²` 는 컬럼 c 의 값 u_c 가 꼭짓점 p_kc 일 때 최대인
  **오목 단봉**. peak p ~ U(0,1), 가중치 w ~ U(0.5,1.5) 후 목적별 합=1 정규화.
- **분리가능**: 이 항만 있으면 각 컬럼을 독립으로 최적화(각자 peak 로 맞춤)하면
  전역 최적이다. 상호작용·기만·ruggedness 가 전혀 없다.

### ② trade-off (F6)

```
c(u) = Σ_{common} c_w · u   (가중평균, c_w 합=1 → c ∈ [0,1])
latent_k += c_gain_k · c(u)
```

- c 가 커지면 6목적 latent 가 **전부** 커진다.
- 최대화 목적(y·1,y·2)엔 이득이지만 **최소화 목적(y13,y23)엔 c 가 클수록
  악화** → common 블록을 두고 max/min 목적이 충돌한다.
- `c_gain ~ U(0.25, 0.35)` — 5벤치 중 **가장 약한** 충돌. 기울기가 완만해
  타협점을 쉽게 찾는다("easy").

## 왜 쉬운가

- 없는 것: 블록 내/교차 상호작용(F2/F3), 기만(F4), ruggedness(F5).
- 있는 것: 분리가능한 단봉 + 완만한 trade-off.
- → 좌표를 하나씩 peak 로 맞추고 common 에서 trade-off 타협만 잡으면 된다.
  `blockwise_coord` 같은 좌표 계열이 잘 통해야 하며, 통하지 않으면
  구현이나 스케일러를 의심해야 한다(진단 기준선).

## raw_y 로의 변환

latent 6값 → 관측 장치(stage.md §0-④):
- latent[0],latent[1] → mask1 → y11(height), y12(width)
- latent[3],latent[4] → mask2 → y21(height), y22(width)
- latent[2],latent[5] → 스칼라 y13, y23 (은닉 스케일 + 가우시안 노이즈)

## 실측 (budget 780, 10 seed)

챔피언 xgb_tr 0.755 / random 0.440(11위) / 갭 0.315. 분리가능해도 trade-off
때문에 상한이 1보다 한참 아래(≈0.76)에 형성된다.
