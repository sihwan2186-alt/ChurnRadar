from pydantic import BaseModel
from typing import Optional, List

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
    history_arpu: Optional[List[float]] = None  # What-if 시뮬레이터용 30일 배열

class ChurnPrediction(BaseModel):
    customer_id: str
    xgb_probability: float
    ts_probability: float
    churn_probability: float  # 앙상블 평균
    churn_prediction: bool
    risk_level: str  # HIGH / MEDIUM / LOW
    expected_revenue_loss: float  # 이탈 시 ARPU 기준 손실액
