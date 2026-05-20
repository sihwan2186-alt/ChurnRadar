# ChurnRadar 투트랙 연구 정리

작성일: 2026-05-20

## 사용 근거

- 선행연구: Makokha, Obote, Muchiri, Senagi (2026), "Predicting Customer Churn in the Telecommunications Industry using Machine Learning Techniques", American Journal of Networks and Communications, 15(1), 10-26. DOI: https://doi.org/10.11648/j.ajnc.20261501.12
- 로컬 PDF: `c:\Users\sihwa\OneDrive\바탕 화면\기계학습프로젝\j.ajnc.20261501.12.pdf`
- 원천 데이터: Baza Telecom / Bulgarian business customer churn 데이터

선행연구는 Baza 계열 데이터를 사용했고, 약 8,454개 기업 고객 계정, 14개 원천 속성, 약 6.5% 이탈 비율을 전제로 한다. 논문 초록/인용 정보에는 EasyEnsembleClassifier가 대표 모델로 제시되며 F1-score 0.129, recall 약 38.24% 수준으로 보고되어 있다. 이 수치 자체가 낮기 때문에, 우리 프로젝트에서도 같은 데이터만으로 F1 0.6 이상을 기대하기 어렵다는 점을 보고서에 명확히 적는다.

## Track 1. 기존 Baza 연구 재현 및 한계 검증

목표는 기존 연구의 문제 설정을 재현하고, 현재 데이터의 한계를 수치로 확인하는 것이다. 모델을 무리하게 바꾸는 대신, 같은 정적 CRM 피처에서 예측 신호가 얼마나 약한지를 실험으로 보인다.

### 1. 실제 불균형 분포 실험

실험 파일:

- `scripts/track1_baza_downsample_assumption.py`
- `results/baza_downsample_assumption_table.csv`
- `results/baza_downsample_assumption_summary.json`

데이터 상태:

| 항목 | 값 |
|---|---:|
| 전체 행 | 8,453 |
| 이탈 고객 | 549 |
| 비이탈 고객 | 7,904 |
| 이탈률 | 6.49% |

실제 분포를 유지한 테스트셋에서는 최고 결과가 다음 수준에 머문다.

| 실험 | 최고 모델 | Test F1 | Precision | Recall | ROC-AUC |
|---|---|---:|---:|---:|---:|
| 실제 분포 유지 | DecisionTree_balanced | 0.1218 | 0.0694 | 0.5000 | 0.5076 |
| 학습셋만 다운샘플링, 테스트셋 실제 분포 | DecisionTree_balanced | 0.1335 | 0.0719 | 0.9273 | 0.5503 |

해석: 학습셋을 균형에 맞춰도 실제 테스트셋의 이탈률이 6.5%이면 precision이 낮아지고 F1은 크게 오르지 않는다. 따라서 이 데이터에서 낮은 F1은 단순 모델 선택 문제가 아니라 데이터 분포와 피처 신호의 문제로 보는 것이 타당하다.

### 2. 균형 데이터 가정 실험

교수님 피드백에 맞춰 이탈 고객 549명 전체와 비이탈 고객 549명을 랜덤 추출해 총 1,098개 행의 균형 샘플을 만들었다. 이는 원본의 약 13.0%에 해당한다.

| 실험 | 최고 모델 | Test F1 | Precision | Recall | ROC-AUC |
|---|---|---:|---:|---:|---:|
| 테스트셋까지 50:50 균형 가정 | RandomForest_balanced | 0.6667 | 0.5023 | 0.9909 | 0.5453 |

보고 시 주의 문장:

> 이 결과는 실제 운영 분포의 성능이 아니라, 이탈/비이탈 비율이 균등하다고 가정한 샘플링 실험 결과이다. 실제 Baza 분포에서는 이탈률이 6.5%이므로 precision과 F1이 다시 낮아진다.

## Track 2. 차별화 피처 확장 및 제거 실험

목표는 모델 종류를 늘리는 것이 아니라, 우리 팀만의 피처 발상을 추가하고 그 피처가 실제로 기여하는지 검증하는 것이다.

추가한 피처 방향:

- GeoNames 우편번호 기반 지리 피처: 위도, 경도, 주요 도시까지의 거리, 행정구역
- Bulgaria NSI 공공통계 기반 인구 피처: district/municipality 인구, 성비, 도시/농촌 비율, 노동연령 비율
- 기존 CRM 피처의 파생 변수: 매출 비율, 가입자 상태 비율, revenue per subscriber, dormant ratio 등

관련 파일:

- `scripts/build_bulgaria_geo_features.py`
- `scripts/build_bulgaria_public_features.py`
- `scripts/track2_baza_public_feature_ablation.py`
- `results/baza_public_feature_importance.csv`
- `results/baza_public_feature_group_importance.csv`
- `results/baza_public_feature_ablation.csv`
- `results/baza_public_feature_ablation_summary.json`

GeoNames/NSI 매칭률은 약 79.59%였다. 최종 실험에는 86개 피처가 사용되었다.

### 1. 피처 중요도

Random Forest 기준 상위 피처는 다음과 같다.

| 순위 | Feature | Group | Importance |
|---:|---|---|---:|
| 1 | AvgMobileRevenue | revenue_size | 0.0656 |
| 2 | TotalRevenue | revenue_size | 0.0649 |
| 3 | ARPU | revenue_size | 0.0636 |
| 4 | Revenue_Per_Sub | revenue_size | 0.0576 |
| 5 | Revenue_Per_Active | revenue_size | 0.0526 |
| 6 | Total_SUBs | revenue_size | 0.0450 |
| 7 | Active_subscribers | subscriber_status | 0.0419 |
| 8 | Inactive_Ratio | subscriber_status | 0.0311 |
| 9 | Active_Ratio | subscriber_status | 0.0295 |
| 10 | Dormant_Ratio | subscriber_status | 0.0268 |
| 11 | bg_latitude | geo_billing_zone | 0.0233 |
| 12 | bg_place_name | geo_billing_zone | 0.0213 |

해석: 중요도 상위권은 대부분 매출 규모와 가입자 상태 피처다. 지리 피처도 일부 상위권에 등장하지만, 중요도만으로 실제 성능 기여를 단정할 수 없으므로 제거 실험을 함께 진행했다.

### 2. 피처 제거 실험

Random Forest 전체 피처 기준:

| 조건 | Test F1 | Precision | Recall | ROC-AUC | F1 변화 |
|---|---:|---:|---:|---:|---:|
| 전체 피처 | 0.1142 | 0.0744 | 0.2455 | 0.5041 | 기준 |
| customer_profile 제거 | 0.1027 | 0.0769 | 0.1545 | 0.5053 | -0.0114 |
| subscriber_status 제거 | 0.1079 | 0.0699 | 0.2364 | 0.5170 | -0.0063 |
| public_demographics 제거 | 0.1121 | 0.0744 | 0.2273 | 0.5182 | -0.0021 |
| revenue_size 제거 | 0.1139 | 0.0659 | 0.4182 | 0.5019 | -0.0003 |
| geo_billing_zone 제거 | 0.1189 | 0.0767 | 0.2636 | 0.5019 | +0.0047 |

해석:

- `customer_profile`과 `subscriber_status` 제거 시 F1이 가장 많이 감소했다. 즉, 기존 CRM 세그먼트와 가입자 상태 비율은 실제 예측에 가장 직접적인 근거가 된다.
- `public_demographics`는 제거 시 F1이 소폭 감소했지만, 효과가 크지는 않다.
- `geo_billing_zone`은 제거해도 F1이 오히려 소폭 상승했다. 현재 split에서는 지리 피처가 안정적인 예측 신호라기보다 잡음에 가까울 수 있다.
- 결론적으로, 차별화 피처를 만들고 검증했지만 현 데이터만으로 강한 예측력을 확보하지는 못했다.

## 최종 보고 논리

1. 기존 연구 트랙에서는 선행연구와 같은 Baza Telecom 계열 데이터를 재현하고, 파이프라인과 여러 모델을 구현했다. 그러나 이 데이터는 이탈률이 약 6.5%이고 피처가 14개 원천 컬럼 중심이라 예측 신호가 약했다.
2. 균형 데이터 가정에서는 F1이 0.6667까지 상승했다. 다만 이는 실제 운영 성능이 아니라, 이탈/비이탈이 균등하다는 가정 아래의 실험이다.
3. 모델을 더 복잡하게 바꾸는 대신 지리/공공통계/가입자상태 파생 피처를 추가했다. 또한 피처 중요도와 제거 실험으로 각 피처군의 역할을 검증했다.
4. 제거 실험 결과, customer profile과 subscriber status는 상대적으로 의미가 있었지만, 새로 붙인 지리/공공통계 피처만으로 큰 성능 향상은 없었다.
5. 따라서 최종 결론은 "학부 수준에서 모델 튜닝으로 억지 성능을 만든 것이 아니라, 데이터 한계를 검증하고 피처 중심 개선을 시도했으며, 실제로 어떤 피처군이 기여했는지 실험적으로 확인했다"로 잡는다.

## 발표용 한 문장

> 기존 Baza Telecom 데이터는 이탈률 6.5%의 강한 불균형과 정적 CRM 피처 부족으로 실제 분포 F1이 낮게 유지되었다. 다만 균형 샘플 가정에서는 F1이 크게 상승함을 확인했고, 추가로 지리/공공통계 피처를 설계한 뒤 중요도 및 제거 실험을 통해 피처군별 기여도를 검증했다.

## 재실행 명령

```bash
python -B scripts/track1_baza_downsample_assumption.py
python -B scripts/track2_baza_public_feature_ablation.py
```
