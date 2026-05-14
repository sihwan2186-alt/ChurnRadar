from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional


CRITICAL_RENOTIFY_WINDOW = timedelta(hours=24)
HIGH_SUPPRESS_WINDOW = timedelta(days=3)
PROBABILITY_SPIKE_THRESHOLD = 0.20

RISK_ORDER = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}

RESOLVED_RESPONSE_STATUSES = {
    "done",
    "closed",
    "complete",
    "completed",
    "resolved",
    "responded",
    "response_complete",
    "대응완료",
    "완료",
    "해결",
}


@dataclass(frozen=True)
class AlertDecision:
    alert_required: bool
    alert_channel: str
    suppress_reason: Optional[str]
    log_required: bool = True


def classify_risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "Critical"
    if probability >= 0.65:
        return "High"
    if probability >= 0.5:
        return "Medium"
    return "Low"


def normalize_risk_level(risk_level: str) -> str:
    compact = (risk_level or "").strip().replace("_", " ").upper()
    if compact in {"CRITICAL", "URGENT"}:
        return "CRITICAL"
    if compact in {"HIGH", "HIGH RISK"}:
        return "HIGH"
    if compact in {"MEDIUM", "MID", "MEDIUM RISK"}:
        return "MEDIUM"
    return "LOW"


def parse_alert_time(value: Optional[object]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def is_response_resolved(response_status: Optional[str]) -> bool:
    if response_status is None:
        return False
    return response_status.strip().lower() in RESOLVED_RESPONSE_STATUSES


def has_probability_spike(
    churn_probability: float,
    previous_churn_probability: Optional[float],
) -> bool:
    if previous_churn_probability is None:
        return False
    return churn_probability - previous_churn_probability >= PROBABILITY_SPIKE_THRESHOLD


def has_risk_escalated(
    risk_level: str,
    previous_risk_level: Optional[str],
) -> bool:
    if not previous_risk_level:
        return False
    current_rank = RISK_ORDER[normalize_risk_level(risk_level)]
    previous_rank = RISK_ORDER[normalize_risk_level(previous_risk_level)]
    return current_rank > previous_rank and current_rank >= RISK_ORDER["HIGH"]


def choose_alert_channel(risk_level: str, is_priority_customer: bool) -> str:
    normalized = normalize_risk_level(risk_level)
    if normalized == "CRITICAL":
        return "Slack,Gmail" if is_priority_customer else "Slack"
    if normalized == "HIGH":
        return "Slack"
    return "None"


def evaluate_alert_fatigue(
    *,
    risk_level: str,
    churn_probability: float,
    last_alert_time: Optional[object] = None,
    response_status: Optional[str] = None,
    previous_churn_probability: Optional[float] = None,
    previous_risk_level: Optional[str] = None,
    is_vip_customer: bool = False,
    is_high_revenue_customer: bool = False,
    now: Optional[datetime] = None,
) -> AlertDecision:
    normalized = normalize_risk_level(risk_level)
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    last_alert = parse_alert_time(last_alert_time)
    responded = is_response_resolved(response_status)
    probability_spiked = has_probability_spike(
        churn_probability,
        previous_churn_probability,
    )
    risk_escalated = has_risk_escalated(normalized, previous_risk_level)
    priority_customer = is_vip_customer or is_high_revenue_customer

    if normalized == "LOW":
        return AlertDecision(
            alert_required=False,
            alert_channel="None",
            suppress_reason="Low 등급 고객은 알림 대상이 아니므로 로그만 저장함",
        )

    if normalized == "MEDIUM":
        return AlertDecision(
            alert_required=False,
            alert_channel="None",
            suppress_reason="Medium 등급 고객은 실시간 알림 없이 Google Sheets에만 기록함",
        )

    if probability_spiked:
        return AlertDecision(
            alert_required=True,
            alert_channel=choose_alert_channel(normalized, priority_customer),
            suppress_reason=None,
        )

    if risk_escalated:
        return AlertDecision(
            alert_required=True,
            alert_channel=choose_alert_channel(normalized, priority_customer),
            suppress_reason=None,
        )

    if responded and last_alert is not None:
        return AlertDecision(
            alert_required=False,
            alert_channel="None",
            suppress_reason="담당자 대응이 완료된 고객으로 중복 알림을 차단함",
        )

    if normalized == "CRITICAL":
        if last_alert is None:
            return AlertDecision(
                alert_required=True,
                alert_channel=choose_alert_channel(normalized, priority_customer),
                suppress_reason=None,
            )

        if current_time - last_alert >= CRITICAL_RENOTIFY_WINDOW:
            return AlertDecision(
                alert_required=True,
                alert_channel=choose_alert_channel(normalized, priority_customer),
                suppress_reason=None,
            )

        return AlertDecision(
            alert_required=False,
            alert_channel="None",
            suppress_reason="Critical 등급 고객은 최근 24시간 내 알림이 발송되어 재알림을 차단함",
        )

    if normalized == "HIGH":
        if last_alert is None:
            return AlertDecision(
                alert_required=True,
                alert_channel=choose_alert_channel(normalized, priority_customer),
                suppress_reason=None,
            )

        if current_time - last_alert >= HIGH_SUPPRESS_WINDOW:
            return AlertDecision(
                alert_required=True,
                alert_channel=choose_alert_channel(normalized, priority_customer),
                suppress_reason=None,
            )

        return AlertDecision(
            alert_required=False,
            alert_channel="None",
            suppress_reason="High 등급 고객은 최근 3일 내 알림이 발송되어 중복 알림을 차단함",
        )

    return AlertDecision(
        alert_required=False,
        alert_channel="None",
        suppress_reason="알림 정책에 따라 발송 대상에서 제외됨",
    )
