# ChurnRadar Feature Improvement Notes

## 1. Baza static feature expansion

기존 Baza 정적 피처는 규모, 매출 구성, 활성도 중심이라 “현재 값”은 보지만 고객 행동이 이탈 방향으로 변하고 있는 맥락은 약했다.

이번 변경으로 다음 피처를 학습/튜닝/벤치마크/API 입력 경로에 추가했다.

| Feature | Meaning |
| --- | --- |
| `Revenue_per_Active_Sub` | 활성 구독자 1명당 부담 매출 |
| `Suspended_Ratio` | 정지 구독자 비율 |
| `Inactive_x_Revenue` | 비활성 비율과 매출 규모의 교호작용 |
| `Revenue_Balance` | 모바일/유선 매출 균형도 |

이 피처들은 “매출이 누구에게 집중되어 있는가”, “비활성도가 매출 규모와 결합될 때 위험도가 커지는가”, “특정 서비스 의존도가 높은가”를 모델에 명시적으로 제공한다.

## 2. KKBox source-domain behavior features

KKBox 시계열은 기존에는 `total_secs`에서 나온 `Activity_Energy`, `Momentum`, `Acceleration`만 사용했다. 이는 하나의 신호의 값, 1차 변화, 2차 변화라 독립 행동 신호가 부족했다.

`map_kkbox_to_energy` 출력에 다음 채널을 추가했다.

| Feature | Formula | Interpretation |
| --- | --- | --- |
| `Skip_Rate` | `num_25 / (num_unq + 1e-5)` | 짧게 듣고 넘긴 곡 비율 |
| `Completion_Rate` | `num_100 / (num_unq + 1e-5)` | 끝까지 들은 곡 비율 |
| `Diversity_Score` | `num_unq / (total_plays + 1e-5)` | 반복 재생 대비 탐색 다양성 |

`ChurnTimeSeriesDataset`은 이제 기존 3채널 Parquet과 개선된 6채널 Parquet을 모두 읽는다.

## 3. Baza target-domain time-series simulation

기존 `make_baza_ts.py`는 각 timestep을 같은 기준값 주변의 iid 노이즈로 생성했다. 이 경우 위치 정보와 값 사이의 관계가 없어 Transformer/TCN이 시간 방향성을 학습하기 어렵다.

이번 변경은 label-aware trend를 도입했다.

| Label | Energy | Momentum | Acceleration |
| --- | --- | --- | --- |
| churn | 점진 감소 | 점진 감소 | 비활성 비율 증가 |
| non-churn | 안정 또는 완만 상승 | 안정 또는 완만 상승 | 안정 또는 완만 감소 |

단, Baza 원본은 정적 스냅샷이므로 이 시계열은 실제 관측값이 아니라 가정 기반 시뮬레이션이다. 보고서에는 “이탈 고객에게 이탈 방향 추세가 존재한다고 가정했을 때의 실험”으로 명시해야 한다.

## 4. Model input change

기존:

```text
X = (Customer, 30 Days, 3 Features)
[Activity_Energy, Momentum, Acceleration]
```

개선된 KKBox 입력:

```text
X = (Customer, 30 Days, 6 Features)
[Activity_Energy, Momentum, Acceleration, Skip_Rate, Completion_Rate, Diversity_Score]
```

Baza는 실제 로그가 없으므로 기본 3채널을 유지하되, 생성 방식만 iid 노이즈에서 방향성 있는 추세 시뮬레이션으로 변경했다.
