# ChurnRadar Baza F1 개선 실험 결과

작성일: 2026-05-08

## 1. 목표

- 주 목표 데이터셋: Baza Telecom 원본 데이터
- 보조 데이터셋: IBM, Cell2Cell, Iranian, BigML, India Monthly Usage, KKBox Activity 등은 학습 보조용으로만 사용
- 목표 지표: Baza 테스트셋 기준 F1 0.6 이상
- 현재 목적: 가능한 많은 모델과 데이터 조합을 실험하여, 어떤 접근이 Baza F1 개선에 가장 효과적인지 확인

## 2. 현재 결론

현재까지의 실험에서는 F1 0.6에 도달하지 못했다.

가장 좋은 결과는 외부 보조 데이터까지 함께 사용한 transfer 방식의 `all + LGBM_balanced` 조합이며, Baza 테스트 F1은 `0.1908`이다.

Baza 원본 데이터만 사용하고 피처 엔지니어링과 불균형 모델을 대량 도입한 실험에서는 `encoded_no_ka + EasyEnsemble_10` 조합이 가장 좋았고, Baza 테스트 F1은 `0.1664`이다.

즉, Baza 원본 14개 정적 컬럼만으로는 분류 신호가 약하며, F1 0.6 달성에는 PID 기준의 실제 거래/사용 로그 또는 월별 행동 이력 같은 추가 Baza 전용 데이터가 필요할 가능성이 높다.

## 3. Baza 데이터 상태

사용 파일:

- `raw/raw/baza_telecom_v2.csv`

데이터 크기:

| 항목 | 값 |
|---|---:|
| 전체 행 수 | 8,453 |
| 전체 컬럼 수 | 14 |
| 비이탈 고객 | 7,904 |
| 이탈 고객 | 549 |
| 이탈률 | 약 6.49% |

데이터 분할:

| 분할 | 행 수 |
|---|---:|
| Train | 5,071 |
| Validation | 1,691 |
| Test | 1,691 |

## 4. 이번에 추가로 진행한 작업

### 4.1 불균형 모델 추가

다음 모델 계열을 추가하여 Baza churn처럼 양성 클래스가 적은 문제에 대응했다.

- EasyEnsemble
- BalancedRandomForest
- BalancedBagging
- RUSBoost
- RandomOverSampler
- SMOTE
- ADASYN
- BorderlineSMOTE
- SMOTEENN
- SMOTETomek
- class_weight 기반 Logistic Regression, SVM, Tree, Forest, XGBoost, LightGBM

### 4.2 Baza 전용 피처 엔지니어링

Baza 원본 컬럼만 사용해서 다음 피처를 생성했다.

- 매출 비율: mobile/fixed revenue ratio
- 가입자당 매출: revenue per subscriber
- 활성 고객당 매출: revenue per active subscriber
- 고객 상태 비율: active, inactive, suspended, dormant 비율
- 로그/제곱근 변환: skew가 큰 수치 컬럼 완화
- 결측 여부 flag
- mobile only / fixed only / zero revenue / large account flag
- CRM, Segment, ZIP, KA 등 범주형 frequency encoding
- train fold 기반 target encoding
- WOE encoding
- CRM x Segment, ZIP x Segment 등 interaction encoding
- KA 관련 피처를 제외한 `encoded_no_ka` 피처셋도 별도 실험

### 4.3 외부 보조 데이터셋 확인 및 실험

외부 데이터는 Baza의 실제 PID와 직접 연결되지 않기 때문에 Baza 전용 feature로 join할 수는 없었다.

대신 보조 학습 데이터로 활용하여 Baza 테스트 F1이 올라가는지 확인했다.

사용 또는 확인한 데이터:

| 데이터셋 | 사용 방식 | 결론 |
|---|---|---|
| IBM Telco | 보조 학습 | transfer에서 일부 도움 |
| Cell2Cell | 보조 학습 | transfer에서 일부 도움 |
| Iranian Churn | 보조 학습 | transfer에서 일부 도움 |
| BigML churn | 보조 학습 | 최고점 갱신은 못함 |
| India Monthly Usage | 보조 학습 | 최고점 갱신은 못함 |
| KKBox transactions/user_logs/members | activity feature 생성 후 보조 학습 | Baza와 PID 연결 불가, 단독 효과 제한 |
| PowerBI churn challenge | 중복 IBM 계열로 판단 | 신규 정보로 사용하지 않음 |
| Kaggle feature engineering churn | 확인 | 인공/소규모 성격이라 우선순위 낮음 |

## 5. 전체 최고 결과 비교

| 구분 | 최고 조합 | Baza Test F1 | Precision | Recall | ROC-AUC | 해석 |
|---|---|---:|---:|---:|---:|---|
| 기존 Baza-only broad benchmark | `core + LogReg_L1_balanced` | 0.1584 | 0.1013 | 0.3636 | 0.5716 | 기존 Baza-only 최고 |
| 신규 Baza-only feature engineering | `encoded_no_ka + EasyEnsemble_10` | 0.1664 | 0.1006 | 0.4818 | 0.5883 | Baza-only 기준 개선 |
| 외부 데이터 transfer 포함 | `all + LGBM_balanced` | 0.1908 | 0.1398 | 0.3000 | 0.6137 | 현재 전체 최고 |
| 목표 | F1 0.6 이상 | 0.6000 | - | - | - | 현재 미달 |

개선 폭:

| 비교 | F1 변화 |
|---|---:|
| 기존 Baza-only 최고 0.1584 -> 신규 Baza-only FE 최고 0.1664 | +0.0080 |
| 기존 transfer baseline 0.1435 -> 현재 전체 최고 0.1908 | +0.0473 |

## 6. Baza-only Feature Engineering Top 10

| 순위 | Feature set | Model | Family | Test F1 | Precision | Recall | ROC-AUC |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | encoded_no_ka | EasyEnsemble_10 | imbalance_ensemble | 0.1664 | 0.1006 | 0.4818 | 0.5883 |
| 2 | encoded_compact | EasyEnsemble_10 | imbalance_ensemble | 0.1664 | 0.1006 | 0.4818 | 0.5883 |
| 3 | encoded | EasyEnsemble_30 | imbalance_ensemble | 0.1649 | 0.0994 | 0.4818 | 0.5895 |
| 4 | core | XGBoost_d1 | boosting | 0.1644 | 0.1098 | 0.3273 | 0.5665 |
| 5 | core | GradientBoosting_d3 | boosting | 0.1631 | 0.1013 | 0.4182 | 0.5788 |
| 6 | engineered | GradientBoosting_d3 | boosting | 0.1580 | 0.1128 | 0.2636 | 0.5758 |
| 7 | engineered | EasyEnsemble_10 | imbalance_ensemble | 0.1555 | 0.1011 | 0.3364 | 0.5899 |
| 8 | core | EasyEnsemble_10 | imbalance_ensemble | 0.1536 | 0.0956 | 0.3909 | 0.5869 |
| 9 | core | GradientBoosting_d2 | boosting | 0.1531 | 0.0937 | 0.4182 | 0.5755 |
| 10 | encoded_no_ka | EasyEnsemble_30 | imbalance_ensemble | 0.1515 | 0.0963 | 0.3545 | 0.5852 |

## 7. Feature Set별 최고 결과

| Feature set | 최고 모델 | Test F1 |
|---|---|---:|
| encoded_no_ka | EasyEnsemble_10 | 0.1664 |
| encoded_compact | EasyEnsemble_10 | 0.1664 |
| encoded | EasyEnsemble_30 | 0.1649 |
| core | XGBoost_d1 | 0.1644 |
| engineered | GradientBoosting_d3 | 0.1580 |
| ensemble blend | weighted_blend_diverse_family_top | 0.1429 |

해석:

- `encoded_no_ka`가 가장 좋았다.
- KA 관련 피처를 제외해도 성능이 유지되거나 약간 좋아졌기 때문에, KA 관련 컬럼은 현재 split에서는 강한 일반화 신호가 아닐 수 있다.
- EasyEnsemble 계열이 Baza의 클래스 불균형에 가장 안정적으로 반응했다.

## 8. 모델 계열별 최고 결과

| 계열 | 최고 조합 | Test F1 |
|---|---|---:|
| imbalance_ensemble | `encoded_no_ka + EasyEnsemble_10` | 0.1664 |
| boosting | `core + XGBoost_d1` | 0.1644 |
| linear | `encoded + Ridge_balanced_calibrated` | 0.1503 |
| forest | `encoded_no_ka + RandomForest_d5_balanced` | 0.1471 |
| linear_resampled | `encoded + LogReg_SMOTE` | 0.1465 |
| svm | `encoded + LinearSVC_calibrated` | 0.1461 |
| resampled_boosting | `engineered + XGB_RUS` | 0.1434 |
| discriminant | `core + LDA_shrinkage` | 0.1430 |
| score_blend | `weighted_blend_diverse_family_top` | 0.1429 |
| resampled_forest | `encoded_no_ka + RF_ROS` | 0.1400 |
| bayes | `core + GaussianNB` | 0.1293 |
| neural_net | `encoded_no_ka + MLP_64_32` | 0.1265 |
| neighbors | `engineered + KNN_15_distance` | 0.1259 |
| tree | `encoded_no_ka + DecisionTree_balanced` | 0.1245 |

## 9. 현재 전체 최고 모델 상세

현재 전체 최고 모델:

- 학습 방식: 외부 보조 데이터 transfer 포함
- Source group: `all`
- Source weight: `1.0`
- Model: `LGBM_balanced`
- Baza Test F1: `0.1908`
- Precision: `0.1398`
- Recall: `0.3000`
- ROC-AUC: `0.6137`
- Average Precision: `0.1091`
- Threshold: `0.1938`

Confusion matrix:

| 항목 | 값 |
|---|---:|
| TN | 1,378 |
| FP | 203 |
| FN | 77 |
| TP | 33 |

## 10. 왜 F1 0.6까지 올라가지 않는가

### 10.1 Baza 원본 컬럼의 정보량 한계

현재 Baza 원본은 8,453행, 14개 컬럼의 정적 요약 데이터다.

이탈 예측에서 강한 신호가 되는 월별 사용량 변화, 결제 실패, 서비스 중단 횟수, 최근 활동 감소, 요금 납부 패턴, 고객 문의 이력 같은 시간 기반 행동 정보가 부족하다.

### 10.2 클래스 불균형

이탈 고객은 약 6.49%뿐이다.

모델이 이탈자를 많이 잡도록 threshold를 낮추면 recall은 올라가지만 precision이 크게 떨어진다. 반대로 precision을 올리면 recall이 떨어진다.

현재 최고 Baza-only 모델도 recall은 0.4818까지 올렸지만 precision이 0.1006에 머물러 F1이 0.1664에서 제한된다.

### 10.3 외부 데이터의 domain mismatch

IBM, Cell2Cell, Iranian, India, KKBox는 모두 통신/구독 이탈 데이터지만 Baza 고객과 직접 연결되지 않는다.

그래서 보조 학습에는 도움이 되지만, Baza의 실제 고객 행동을 설명하는 데에는 한계가 있다.

특히 KKBox의 transactions, user_logs, members는 `msno` 기준 데이터라 Baza의 `PID`와 직접 join할 수 없다.

## 11. F1을 더 올리기 위한 다음 전략

### 11.1 가장 효과적인 방향

가장 필요한 것은 Baza 고객 `PID`와 연결되는 추가 데이터다.

우선순위가 높은 데이터:

| 우선순위 | 필요 데이터 | 이유 |
|---:|---|---|
| 1 | 월별 매출/사용량 이력 | churn 직전 감소 패턴을 잡을 수 있음 |
| 2 | 결제/청구/미납 이력 | 이탈 전 강한 신호일 가능성이 높음 |
| 3 | 서비스 중단/정지/재개 이력 | 현재 상태보다 변화 이력이 더 중요할 수 있음 |
| 4 | 고객 문의/불만/장애 접수 이력 | 이탈 전 dissatisfaction 신호 |
| 5 | 요금제 변경/해지 신청 이력 | churn 직전 행동 신호 |
| 6 | 지역/지점/상품군 상세 정보 | 현재 ZIP/segment보다 더 세밀한 신호 가능 |

### 11.2 PID 로그 확보가 불가능할 때 가능한 대안

PID 단위 거래/사용 로그 확보가 불가능하다면 다음 방향으로 진행할 수 있다.

1. 현재 Baza-only best 모델을 기준 모델로 확정
   - `encoded_no_ka + EasyEnsemble_10`
   - 장점: 외부 데이터 없이 Baza 원본만 사용
   - 단점: F1 0.1664 수준

2. 실제 운영 목적이 이탈자 탐지라면 threshold를 recall 중심으로 조정
   - F1만 최적화하지 않고, 영업팀이 감당 가능한 FP 수를 기준으로 threshold 선택
   - 예: 상위 5%, 10%, 20% 위험 고객 리스트 생성

3. transfer 최고 모델을 후보로 유지
   - `all + LGBM_balanced`
   - 현재 전체 최고 F1 0.1908
   - 단, 외부 데이터 사용 근거와 domain mismatch를 보고서에 명확히 적어야 함

4. 모델 자체 추가보다 검증 안정성 강화
   - repeated stratified split
   - nested cross-validation
   - seed 여러 개 평균
   - Baza 데이터가 작기 때문에 단일 split 점수만으로 모델 우열을 확정하기 어렵다.

5. 목적 지표 재검토
   - 현재 데이터 상태에서는 F1 0.6 목표가 매우 높다.
   - PR-AUC, Recall@TopK, Lift@TopK 같은 영업 활용 지표를 함께 보는 것이 현실적이다.

## 12. 산출 파일

이번 실험에서 생성 또는 사용한 주요 파일:

| 파일 | 설명 |
|---|---|
| `scripts/train_baza_feature_engineering.py` | Baza-only 피처 엔지니어링 및 275개 모델/조합 벤치마크 |
| `scripts/build_kkbox_activity_features.py` | KKBox transactions/user_logs/members 기반 activity feature 생성 |
| `scripts/train_baza_transfer.py` | 외부 보조 데이터 transfer 실험 |
| `results/baza_feature_engineering_benchmark.csv` | Baza-only feature engineering 전체 결과 275행 |
| `results/baza_feature_engineering_summary.json` | Baza-only feature engineering 요약 결과 |
| `models/baza_feature_engineering_best_model.joblib` | Baza-only 최고 모델 저장 파일 |
| `results/baza_transfer_benchmark_with_bigml.csv` | BigML 포함 transfer 실험 결과 |
| `results/baza_transfer_benchmark_with_india_monthly.csv` | India monthly 포함 transfer 실험 결과 |
| `results/baza_transfer_benchmark_with_kkbox_activity.csv` | KKBox activity 포함 transfer 실험 결과 |
| `data/processed/kkbox_activity_common_features.csv` | KKBox activity 보조 학습용 feature |

## 13. Notion에 올릴 최종 요약

현재까지 가능한 많은 모델과 보조 데이터셋을 도입해 Baza Telecom churn 예측 성능을 개선하려고 시도했다. EasyEnsemble, BalancedRandomForest, SMOTE 계열, LightGBM, XGBoost, CatBoost, SVM, neural network, score blending 등 총 275개 이상의 Baza-only 조합을 추가로 실험했다.

Baza 원본 데이터만 사용한 최고 성능은 `encoded_no_ka + EasyEnsemble_10`의 F1 `0.1664`였고, 외부 보조 데이터까지 포함한 전체 최고 성능은 `all + LGBM_balanced`의 F1 `0.1908`이었다.

F1 0.6에는 도달하지 못했으며, 가장 큰 원인은 Baza 원본 데이터가 정적 요약 컬럼 14개에 가깝고, churn 직전의 사용량 변화/결제 실패/서비스 중단/고객 문의 같은 시간 기반 행동 데이터가 없기 때문이다.

따라서 다음 단계에서 F1을 의미 있게 올리려면 모델을 더 추가하는 것보다 Baza 고객 PID와 연결되는 월별 거래/사용/결제/활동 로그를 확보하는 것이 가장 중요하다. PID 로그 확보가 불가능하다면 현재 최고 모델을 기준으로 위험 고객 Top-K 리스트를 만드는 방식으로 운영 목적에 맞춰 threshold를 조정하는 것이 현실적인 방향이다.
