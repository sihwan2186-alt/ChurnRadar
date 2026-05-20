from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

from api.alert_fatigue import is_response_resolved


class RetentionROIDict(TypedDict):
    customer_id: str
    churn_probability: float
    risk_level: str
    alert_sent: bool
    action_type: str
    response_status: Optional[str]
    actual_churn: Optional[bool]
    retention_success: bool
    expected_revenue_loss: float
    saved_revenue: float
    retention_cost: float
    net_benefit: float
    roi: Optional[float]
    notes: Optional[str]


@dataclass(frozen=True)
class RetentionROI:
    customer_id: str
    churn_probability: float
    risk_level: str
    alert_sent: bool
    action_type: str
    response_status: Optional[str]
    actual_churn: Optional[bool]
    retention_success: bool
    expected_revenue_loss: float
    saved_revenue: float
    retention_cost: float
    net_benefit: float
    roi: Optional[float]
    notes: Optional[str] = None

    def to_dict(self) -> RetentionROIDict:
        return {
            "customer_id": self.customer_id,
            "churn_probability": self.churn_probability,
            "risk_level": self.risk_level,
            "alert_sent": self.alert_sent,
            "action_type": self.action_type,
            "response_status": self.response_status,
            "actual_churn": self.actual_churn,
            "retention_success": self.retention_success,
            "expected_revenue_loss": self.expected_revenue_loss,
            "saved_revenue": self.saved_revenue,
            "retention_cost": self.retention_cost,
            "net_benefit": self.net_benefit,
            "roi": self.roi,
            "notes": self.notes,
        }


def infer_retention_success(
    *,
    alert_sent: bool,
    actual_churn: Optional[bool],
    retention_success: Optional[bool],
    response_status: Optional[str],
) -> bool:
    if retention_success is not None:
        return bool(retention_success)
    if actual_churn is None:
        return False
    return bool(alert_sent and not actual_churn and is_response_resolved(response_status))


def calculate_retention_roi(
    *,
    customer_id: str,
    churn_probability: float,
    risk_level: str,
    alert_sent: bool,
    expected_revenue_loss: float,
    action_type: str = "none",
    action_cost: Optional[float] = None,
    coupon_cost: float = 0.0,
    discount_cost: float = 0.0,
    consulting_cost: float = 0.0,
    response_status: Optional[str] = None,
    actual_churn: Optional[bool] = None,
    retention_success: Optional[bool] = None,
    notes: Optional[str] = None,
) -> RetentionROI:
    cost = (
        float(action_cost)
        if action_cost is not None
        else float(coupon_cost) + float(discount_cost) + float(consulting_cost)
    )
    success = infer_retention_success(
        alert_sent=alert_sent,
        actual_churn=actual_churn,
        retention_success=retention_success,
        response_status=response_status,
    )
    saved_revenue = float(expected_revenue_loss) if success else 0.0
    net_benefit = saved_revenue - cost
    roi = None if cost == 0 else net_benefit / cost
    return RetentionROI(
        customer_id=customer_id,
        churn_probability=float(churn_probability),
        risk_level=risk_level,
        alert_sent=bool(alert_sent),
        action_type=action_type,
        response_status=response_status,
        actual_churn=actual_churn,
        retention_success=success,
        expected_revenue_loss=float(expected_revenue_loss),
        saved_revenue=saved_revenue,
        retention_cost=cost,
        net_benefit=net_benefit,
        roi=roi,
        notes=notes,
    )
