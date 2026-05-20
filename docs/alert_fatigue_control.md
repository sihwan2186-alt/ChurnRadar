# Alert Fatigue Control

## 1. Purpose

ChurnRadar는 Recall을 우선하기 때문에 정상 고객에게도 알림이 갈 수 있다. 같은 고객에게 반복 알림이 발생하면 영업 담당자가 알림을 무시하고, Critical 고객의 가시성이 낮아질 수 있다.

Alert Fatigue Control은 고객별 최근 알림 이력, Risk Level, 대응 상태, 이전 이탈 확률, VIP/고매출 여부를 기준으로 Slack/Gmail 알림 발송 여부를 제어한다.

## 2. Implemented Components

| Component | Status | Path |
| --- | --- | --- |
| Alert policy engine | Implemented | `api/alert_fatigue.py` |
| Predict response integration | Implemented | `api/main.py` |
| Standalone API endpoint | Implemented | `POST /alert-control` |
| Batch prediction integration | Implemented | `POST /predict/batch` |
| n8n workflow | Implemented | `workflows/churn_alert_fatigue_control.json` |
| Tests | Implemented | `tests/test_alert_fatigue.py` |

## 3. Policy

| Risk Level | Policy |
| --- | --- |
| Critical | 재알림은 24시간 후 허용 |
| High | 최근 3일 내 중복 알림 차단 |
| Medium | Slack/Gmail 알림 제외, Google Sheets에만 기록 |
| Low | 알림 제외, 로그만 저장 |

다음 조건은 중복 차단보다 우선한다.

- 이탈 확률이 이전보다 0.20 이상 상승
- Risk Level이 High 이상으로 상승

담당자 대응 상태가 `completed`, `resolved`, `대응완료`, `완료`, `해결`이면 중복 알림을 차단한다.

VIP 또는 고매출 고객은 High 이상에서 `Slack,Gmail` 채널을 사용한다.

## 4. FastAPI Response Example

```json
{
  "customer_id": "B2B-1023",
  "churn_probability": 0.72,
  "risk_level": "High",
  "alert_required": false,
  "alert_channel": "None",
  "suppress_reason": "High 등급 고객은 최근 3일 내 알림이 발송되어 중복 알림을 차단함",
  "log_required": true
}
```

Critical 재알림 예시는 다음과 같다.

```json
{
  "customer_id": "B2B-3302",
  "churn_probability": 0.91,
  "risk_level": "Critical",
  "alert_required": true,
  "alert_channel": "Slack",
  "suppress_reason": null,
  "log_required": true
}
```

## 5. n8n Integration

```text
FastAPI /predict 호출
-> Google Sheets 최근 알림 이력 조회
-> last_alert_time, previous_churn_probability, previous_risk_level, response_status 전달
-> Alert Fatigue Control 정책 적용
-> alert_required=true면 Slack/Gmail 발송
-> 모든 결과를 Google Sheets에 저장
```

Google Sheets 로그 컬럼:

| Column | Meaning |
| --- | --- |
| customer_id | 고객 식별자 |
| prediction_time | 예측 시각 |
| churn_probability | 이탈 확률 |
| risk_level | 위험 등급 |
| alert_sent | 알림 발송 여부 |
| alert_channel | Slack, Gmail, None |
| last_alert_time | 이전 알림 시각 |
| suppress_reason | 알림 차단 사유 |
| response_status | 담당자 대응 상태 |
| owner | 담당자 |
| notes | 대응 메모 |

## 6. One-Line Summary

Alert Fatigue Control은 Risk Level과 최근 알림 이력을 기준으로 같은 고객에게 반복 알림이 발송되는 문제를 막고, 중요한 고객 알림의 가시성을 유지하는 운영형 알림 최적화 기능이다.
