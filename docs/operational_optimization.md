# 운영 최적화 및 의사결정 고도화 모듈

## Operational Optimization & Decision Intelligence Module

## 1. 도입 상태 확인

운영 최적화 모듈은 일부 기능이 이미 구현되어 있었고, 부족한 기능을 이번에 추가했다.

| 기능 | 구현 상태 | 위치 |
| --- | --- | --- |
| FastAPI Batch Prediction | 추가 완료 | `POST /predict/batch`, `api/main.py` |
| Threshold Optimization | 기존 구현 확인 | `src/utils/threshold_optimizer.py`, `scripts/optimize_threshold.py`, `api/threshold_policy.py` |
| Alert Fatigue Control | 기존 구현 확인 | `api/alert_fatigue.py`, `POST /alert-control` |
| Retention ROI Tracker | 추가 완료 | `api/retention_roi.py`, `POST /retention/roi` |

n8n 예시 워크플로우는 `workflows/churn_operational_optimization.json`에 추가했다.

이 모듈은 모델 자체를 새로 만드는 작업이 아니라, 학습된 모델이 실제 운영 환경에서 빠르고 안정적으로 활용되도록 만드는 최적화 기능이다.

## 2. 모듈 추가 배경

ChurnRadar는 고객의 이탈 징후를 사전에 탐지하고, FastAPI와 n8n을 통해 Slack/Gmail 알림까지 자동으로 연결하는 End-to-End 파이프라인을 목표로 한다.

하지만 실제 운영 환경에서는 단순히 모델이 이탈 확률을 예측하는 것만으로는 충분하지 않다.

운영 단계에서는 다음 문제가 발생할 수 있다.

- 고객 수가 많아질수록 API 호출 시간이 증가함
- 기본 threshold 0.5 기준이 B2B 이탈 문제에 적합하지 않을 수 있음
- Recall을 높이면 같은 고객에게 반복 알림이 발생할 수 있음
- 알림이 실제로 고객 이탈 방어에 도움이 되었는지 측정하기 어려움

이를 해결하기 위해 운영 최적화 및 의사결정 고도화 모듈을 구성한다.

```text
1. FastAPI Batch Prediction
2. Threshold Optimization
3. Alert Fatigue Control
4. Retention ROI Tracker
```

## 3. FastAPI Batch Prediction

### 기능 개요

기존 방식에서는 고객 1명마다 FastAPI `/predict` 엔드포인트를 호출해야 했다.

```text
고객 100명 -> API 100번 호출
```

Batch Prediction API는 여러 고객 데이터를 한 번에 입력받아 한 번의 호출로 일괄 예측한다.

```text
고객 100명 -> API 1번 호출
```

### API

```text
POST /predict/batch
```

요청 예시:

```json
{
  "batch_id": "batch-2026-05-09-001",
  "customers": [
    {
      "customer_id": "B2B-1023",
      "total_subs": 3,
      "avg_mobile_revenue": 50.0,
      "avg_fix_revenue": 20.0,
      "total_revenue": 70.0,
      "arpu": 23.3,
      "active_subscribers": 2,
      "not_active_subscribers": 1,
      "crm_segment": "VIP",
      "effective_segment": "Business"
    }
  ]
}
```

응답 예시:

```json
{
  "batch_id": "batch-2026-05-09-001",
  "total_customers": 1,
  "alert_required_count": 1,
  "elapsed_ms": 12.4,
  "predictions": [
    {
      "customer_id": "B2B-1023",
      "churn_probability": 0.82,
      "risk_level": "Critical",
      "alert_required": true,
      "alert_channel": "Slack,Gmail"
    }
  ]
}
```

### 기대 효과

- API 호출 횟수 감소
- n8n 실행 시간 단축
- 대량 고객 일괄 스코어링 가능
- 5초 이내 알림 발송 KPI 달성 가능성 향상
- FastAPI 서버의 운영 효율성 개선

## 4. Threshold Optimization

### 기능 개요

일반적인 분류 모델은 이탈 확률이 0.5 이상이면 이탈 위험 고객으로 판단한다.

```text
Churn Probability >= 0.5 -> 이탈 위험 고객
```

하지만 Baza Telecom v2 데이터는 이탈률이 약 6.5%에 불과한 불균형 데이터이다. 따라서 기본 threshold 0.5는 B2B 이탈 문제에 최적화된 기준이 아닐 수 있다.

본 프로젝트는 validation 데이터에서 여러 threshold를 비교하고, 선택된 threshold를 `results/threshold_optimization_summary.json`에 저장한다. FastAPI는 `api/threshold_policy.py`를 통해 이 값을 읽어 최종 이탈 판단에 사용한다.

### 탐색 기준

```text
Threshold 후보:
0.10, 0.15, 0.20, 0.25, ... , 0.90
```

비교 지표:

- Recall
- Precision
- F1-score
- Confusion Matrix
- 예상 알림 발송 수

### 관련 파일

- `src/utils/threshold_optimizer.py`
- `scripts/optimize_threshold.py`
- `api/threshold_policy.py`
- `docs/threshold_optimization.md`
- `tests/test_threshold_optimizer.py`

## 5. Alert Fatigue Control

### 기능 개요

Recall을 높이면 정상 고객에게도 알림이 발송되는 오탐이 증가할 수 있다. 또한 같은 고객에게 반복적으로 Slack/Gmail 알림이 발송되면 영업 담당자가 알림을 무시하게 되는 Alert Fatigue 문제가 발생할 수 있다.

이를 해결하기 위해 고객별 최근 알림 이력과 Risk Level을 기준으로 중복 알림을 제어한다.

### 알림 제어 정책

| Risk Level | 알림 정책 | 설명 |
| --- | --- | --- |
| Critical | 24시간 후 재알림 가능 | 매우 위험한 고객이므로 반복 확인 필요 |
| High | 3일 내 중복 알림 차단 | 단기 중복 발송 방지 |
| Medium | Slack/Gmail 알림 제외, Sheets 기록만 수행 | 모니터링 대상으로 관리 |
| Low | 알림 제외, 로그만 저장 | 운영 채널 노이즈 최소화 |

### API

```text
POST /alert-control
```

### 관련 파일

- `api/alert_fatigue.py`
- `api/main.py`
- `tests/test_alert_fatigue.py`
- `workflows/churn_alert_fatigue_control.json`

## 6. Retention ROI Tracker

### 기능 개요

ChurnRadar는 이탈 위험 고객을 탐지하고 알림을 발송한다. 하지만 실무적으로 중요한 것은 단순히 알림을 보냈는지가 아니라, 그 알림이 실제로 고객 이탈을 막았는지이다.

Retention ROI Tracker는 알림 발송 이후의 실제 대응 결과를 추적하여 방어 성공 여부와 매출 방어 효과를 계산한다.

### API

```text
POST /retention/roi
```

요청 예시:

```json
{
  "customer_id": "B2B-1023",
  "churn_probability": 0.82,
  "risk_level": "Critical",
  "alert_sent": true,
  "expected_revenue_loss": 1000.0,
  "action_type": "discount",
  "discount_cost": 120.0,
  "consulting_cost": 30.0,
  "response_status": "completed",
  "actual_churn": false
}
```

응답 예시:

```json
{
  "customer_id": "B2B-1023",
  "retention_success": true,
  "expected_revenue_loss": 1000.0,
  "saved_revenue": 1000.0,
  "retention_cost": 150.0,
  "net_benefit": 850.0,
  "roi": 5.6667
}
```

### 계산 지표

```text
Saved Revenue = 방어 성공 고객의 예상 매출 손실
Retention Cost = 쿠폰 비용 + 할인 비용 + 상담 비용
Net Benefit = Saved Revenue - Retention Cost
ROI = Net Benefit / Retention Cost
```

`retention_success`를 직접 입력하지 않은 경우, 다음 조건을 만족하면 방어 성공으로 추론한다.

```text
alert_sent == true
actual_churn == false
response_status가 완료 상태
```

### Google Sheets 로그 컬럼 예시

| 컬럼명 | 설명 |
| --- | --- |
| customer_id | 고객 식별자 |
| prediction_time | 예측 시각 |
| churn_probability | 이탈 확률 |
| risk_level | 위험 등급 |
| alert_sent | 알림 발송 여부 |
| action_type | 상담, 할인, 쿠폰, 재계약 제안 등 |
| action_cost | 대응 비용 |
| response_status | 미대응, 진행 중, 대응 완료 |
| actual_churn | 실제 이탈 여부 |
| retention_success | 방어 성공 여부 |
| saved_revenue | 방어한 예상 매출 |
| net_benefit | 순이익 |
| roi | 투자 대비 효과 |
| notes | 담당자 메모 |

## 7. n8n 워크플로우 연동 구조

```text
Schedule Trigger
-> 고객 데이터 수집
-> FastAPI Batch Prediction 호출
-> Threshold Optimization 기준으로 Risk Level 분류
-> Alert Fatigue Control로 중복 알림 여부 확인
-> Critical/High 고객만 Slack/Gmail 알림
-> 모든 예측 및 대응 결과를 Google Sheets에 저장
-> Retention ROI Tracker로 방어 성과 계산
```

## 8. 전체 기대 효과

- 대량 고객 예측 처리 속도 개선
- n8n 자동화 워크플로우 실행 시간 단축
- 5초 이내 알림 발송 KPI 달성 가능성 향상
- 불균형 데이터에 적합한 알림 기준 설정
- 반복 알림으로 인한 실무자 피로도 감소
- 알림 이후 실제 비즈니스 성과 측정 가능
- 단순 예측 모델을 실무형 AI 운영 시스템으로 확장

## 9. 발표용 요약 문장

> ChurnRadar가 단순 이탈 예측 모델에 머무르지 않고 실제 운영 환경에서 안정적으로 작동할 수 있도록 운영 최적화 및 의사결정 고도화 모듈을 설계했습니다. FastAPI Batch Prediction을 통해 대량 고객 예측 속도를 개선하고, Threshold Optimization으로 B2B 이탈 문제에 맞는 알림 기준을 설정했습니다. 또한 Alert Fatigue Control로 반복 알림을 방지하고, Retention ROI Tracker를 통해 알림 이후 실제 고객 방어 성과와 매출 효과를 측정할 수 있도록 확장했습니다.

## 10. 한 줄 요약

운영 최적화 및 의사결정 고도화 모듈은 ChurnRadar를 단순 예측 시스템에서 대량 고객 처리, 알림 기준 최적화, 중복 알림 제어, ROI 측정까지 가능한 실무형 AI 자동화 시스템으로 확장하는 기능이다.
