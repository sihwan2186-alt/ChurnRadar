# Threshold 최적화

## 1. Threshold란?

이탈 예측 모델은 고객을 바로 이탈/비이탈로 분류하지 않고, 먼저 이탈 확률을 출력한다.

예를 들어 고객별 예측 확률이 다음과 같을 수 있다.

| 고객 | 이탈 확률 |
| --- | ---: |
| A | 0.82 |
| B | 0.63 |
| C | 0.47 |
| D | 0.31 |
| E | 0.12 |

이 확률을 실제 클래스로 바꾸려면 기준값이 필요하다. 이 기준값을 threshold, 즉 분류 임계값이라고 한다.

기본 threshold는 보통 0.5이다.

```text
이탈 확률 >= 0.5 -> 이탈
이탈 확률 < 0.5  -> 비이탈
```

하지만 고객 이탈 예측처럼 클래스 불균형이 큰 문제에서는 0.5가 최적이 아닐 수 있다.

## 2. 왜 최적화가 필요한가?

고객 이탈 데이터에서는 대부분의 고객이 이탈하지 않고, 실제 이탈 고객은 상대적으로 적다. 이런 경우 모델은 이탈 고객을 보수적으로 예측하는 경향이 생길 수 있다.

예를 들어 실제 이탈 고객의 예측 확률이 0.48, 0.44, 0.41처럼 0.5보다 조금 낮게 나온다면, 기본 threshold 0.5에서는 모두 비이탈로 분류되어 실제 이탈 고객을 놓치게 된다.

Threshold를 0.35로 낮추면 더 많은 고객을 이탈 위험 고객으로 분류할 수 있다. 이때 Recall은 올라갈 수 있지만, False Positive도 함께 증가할 수 있다.

## 3. Precision, Recall, F1 Score와의 관계

Threshold를 조정하면 TP, FP, FN, TN이 달라지고, 이에 따라 Precision, Recall, F1 Score도 바뀐다.

| 지표 | 의미 |
| --- | --- |
| Precision | 이탈이라고 예측한 고객 중 실제 이탈 고객의 비율 |
| Recall | 실제 이탈 고객 중 모델이 이탈이라고 잡아낸 비율 |
| F1 Score | Precision과 Recall의 균형 지표 |

고객 이탈 예측에서는 실제 이탈 고객을 놓치는 비용이 크기 때문에 Recall을 중요하게 보되, 알림 과다를 막기 위해 F1 Score로 균형점을 찾는다.

## 4. 최적화 방법

본 프로젝트에서는 여러 threshold 후보를 validation 데이터에서 실험하고, F1 Score가 가장 높은 threshold를 선택한다.

```text
0.20, 0.21, 0.22, ..., 0.70
```

각 threshold마다 Precision, Recall, F1 Score를 계산하고 가장 좋은 값을 `results/threshold_optimization_summary.json`에 저장한다.

실행 예시는 다음과 같다.

```bash
python scripts/optimize_threshold.py
```

## 5. Test 데이터 사용 금지

Threshold도 하나의 튜닝 대상이므로 test 데이터에서 고르면 안 된다.

올바른 방식은 다음과 같다.

```text
Train: 모델 학습
Validation: threshold 선택
Test: 최종 성능 확인
```

Test 데이터에서 threshold를 선택하면 test 데이터에 맞춰 성능을 억지로 올린 것이 되어, 새로운 고객 데이터에서 성능이 떨어질 수 있다.

## 6. 프로젝트 적용 방식

FastAPI는 `results/threshold_optimization_summary.json`의 `selected_threshold`를 읽어 최종 `churn_prediction` 판단에 사용한다.

파일이 없거나 값이 잘못된 경우에는 안전하게 기본값 0.5를 사용한다.

환경변수 `CHURN_THRESHOLD`를 설정하면 파일보다 우선 적용할 수 있다.

```bash
CHURN_THRESHOLD=0.35 uvicorn api.main:app --reload
```

## 7. 한 줄 요약

Threshold 최적화는 모델이 출력한 이탈 확률을 이탈/비이탈 클래스로 변환하는 기준값을 validation 데이터에서 조정하여, 모델 구조를 바꾸지 않고 F1 Score와 Recall을 개선하는 방법이다.
