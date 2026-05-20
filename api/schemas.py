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
    suspended_subscribers: float = 0.0
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
    prediction_threshold: float = 0.5
    churn_prediction: bool
    risk_level: str
    expected_revenue_loss: float
    alert_required: bool
    alert_channel: str
    suppress_reason: Optional[str] = None
    log_required: bool = True


class BatchPredictionRequest(BaseModel):
    batch_id: Optional[str] = None
    customers: List[CustomerData] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    batch_id: str
    total_customers: int
    alert_required_count: int
    elapsed_ms: float
    predictions: List[ChurnPrediction]


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


class RetentionROIRequest(BaseModel):
    customer_id: str
    churn_probability: float = Field(ge=0.0, le=1.0)
    risk_level: str
    alert_sent: bool = False
    expected_revenue_loss: float = Field(default=0.0, ge=0.0)
    action_type: str = "none"
    action_cost: Optional[float] = Field(default=None, ge=0.0)
    coupon_cost: float = Field(default=0.0, ge=0.0)
    discount_cost: float = Field(default=0.0, ge=0.0)
    consulting_cost: float = Field(default=0.0, ge=0.0)
    response_status: Optional[str] = None
    actual_churn: Optional[bool] = None
    retention_success: Optional[bool] = None
    notes: Optional[str] = None


class RetentionROIResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str
    alert_sent: bool
    action_type: str
    response_status: Optional[str] = None
    actual_churn: Optional[bool] = None
    retention_success: bool
    expected_revenue_loss: float
    saved_revenue: float
    retention_cost: float
    net_benefit: float
    roi: Optional[float]
    notes: Optional[str] = None
