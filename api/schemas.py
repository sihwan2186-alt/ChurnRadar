from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class CustomerData(BaseModel):
    customer_id: str
    total_subs: int
    avg_mobile_revenue: float
    avg_fix_revenue: float
    total_revenue: float
    arpu: float
    active_subscribers: int
    not_active_subscribers: float
    crm_segment: str = "Unknown"
    effective_segment: str = "Unknown"
    history_arpu: Optional[List[float]] = None
    last_alert_time: Optional[str] = None
    previous_churn_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    previous_risk_level: Optional[str] = None
    response_status: Optional[str] = None
    is_vip_customer: bool = False
    is_high_revenue_customer: bool = False

    @field_validator("previous_churn_probability", mode="before")
    @classmethod
    def empty_previous_probability_to_none(cls, value):
        if value in ("", "None", "null"):
            return None
        return value


class ChurnPrediction(BaseModel):
    customer_id: str
    xgb_probability: float
    tcn_probability: Optional[float] = None
    ts_probability: float
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    expected_revenue_loss: float
    alert_required: bool
    alert_channel: str
    suppress_reason: Optional[str] = None
    log_required: bool = True


class AlertControlRequest(BaseModel):
    customer_id: str
    churn_probability: float = Field(ge=0.0, le=1.0)
    risk_level: str
    last_alert_time: Optional[str] = None
    previous_churn_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    previous_risk_level: Optional[str] = None
    response_status: Optional[str] = None
    is_vip_customer: bool = False
    is_high_revenue_customer: bool = False

    @field_validator("previous_churn_probability", mode="before")
    @classmethod
    def empty_previous_probability_to_none(cls, value):
        if value in ("", "None", "null"):
            return None
        return value


class AlertControlResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str
    alert_required: bool
    alert_channel: str
    suppress_reason: Optional[str] = None
    log_required: bool = True
