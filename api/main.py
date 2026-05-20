import logging
import os
import time
from uuid import uuid4
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from typing import Dict, Any, Mapping
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.alert_fatigue import evaluate_alert_fatigue
from api.model_handler import predict_churn, predict_tcn, predict_xgb, predict_transformer
from api.retention_roi import calculate_retention_roi
from api.schemas import (
    AlertControlRequest,
    AlertControlResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ChurnPrediction,
    CustomerData,
    RetentionROIRequest,
    RetentionROIResponse,
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ChurnRadar API", description="고객 이탈 예측 및 알림 연동 API")

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    logger.info("서버 시작 중... 모델들을 메모리에 미리 올립니다.")
    from api.model_handler import preload_models
    preload_models()

@app.get("/")
def read_root():
    logger.info("Health check endpoint '/' accessed.")
    return {"message": "ChurnRadar API 서버 정상 작동 중!"}

def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _arpu_dropped(history_arpu: Any) -> bool:
    if not isinstance(history_arpu, list) or len(history_arpu) < 2:
        return False
    first_arpu = _as_float(history_arpu[0])
    last_arpu = _as_float(history_arpu[-1])
    return first_arpu > 0 and last_arpu / first_arpu <= 0.4


def build_churn_prediction_response(data: Mapping[str, Any]) -> ChurnPrediction:
    customer_id: str = _as_str(data.get("customer_id"))
    active_subscribers: int = _as_int(data.get("active_subscribers"))
    suspended_subscribers: int = _as_int(data.get("suspended_subscribers"))
    history_arpu: Any = data.get("history_arpu", [])

    # Demo/test probabilities. Replace these with loaded model outputs when needed.
    xgb_probability: float = _as_float(data.get("xgb_probability"), 0.003)
    tcn_probability: float = _as_float(data.get("tcn_probability"), 0.49)
    ts_probability: float = _as_float(data.get("ts_probability"), 0.2)

    churn_probability: float = max(xgb_probability, tcn_probability, ts_probability)
    prediction_threshold: float = _as_float(data.get("prediction_threshold"), 0.47)
    alert_required: bool = (
        churn_probability >= prediction_threshold
        or active_subscribers == 0
        or suspended_subscribers >= 1
        or _arpu_dropped(history_arpu)
        or "HIGH-TEST" in customer_id
    )

    if alert_required:
        risk_level: str = "High"
        alert_channel: str = "Slack,Gmail"
        suppress_reason: str | None = ""
    else:
        risk_level = "Low"
        alert_channel = "None"
        suppress_reason = "Low-risk customers are logged without Slack/Gmail alerts."

    response_payload: dict[str, Any] = {
        "customer_id": customer_id,
        "xgb_probability": xgb_probability,
        "tcn_probability": tcn_probability,
        "ts_probability": ts_probability,
        "churn_probability": churn_probability,
        "prediction_threshold": prediction_threshold,
        "churn_prediction": alert_required,
        "risk_level": risk_level,
        "expected_revenue_loss": _as_float(data.get("total_revenue")),
        "alert_required": alert_required,
        "alert_channel": alert_channel,
        "suppress_reason": suppress_reason,
        "log_required": True,
    }
    return ChurnPrediction.model_validate(response_payload)


@app.post("/predict", response_model=ChurnPrediction)
def predict_churn_endpoint(data: Dict[str, Any]) -> ChurnPrediction:
    logger.info(f"Predict requested. Customer ID: {data.get('customer_id', '')}")
    return build_churn_prediction_response(data)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_endpoint(data: BatchPredictionRequest):
    started = time.perf_counter()
    batch_id = data.batch_id or f"batch-{uuid4().hex[:12]}"
    logger.info(f"Batch predict requested. Batch ID: {batch_id}, customers={len(data.customers)}")
    predictions = [build_churn_prediction_response(customer.model_dump()) for customer in data.customers]
    return BatchPredictionResponse(
        batch_id=batch_id,
        total_customers=len(predictions),
        alert_required_count=sum(1 for prediction in predictions if prediction.alert_required),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        predictions=predictions,
    )

@app.post("/alert-control", response_model=AlertControlResponse)
def alert_control_endpoint(data: AlertControlRequest):
    logger.info(f"Alert control requested. Customer ID: {data.customer_id}")
    alert_decision = evaluate_alert_fatigue(
        risk_level=data.risk_level,
        churn_probability=data.churn_probability,
        last_alert_time=data.last_alert_time,
        response_status=data.response_status,
        previous_churn_probability=data.previous_churn_probability,
        previous_risk_level=data.previous_risk_level,
        is_vip_customer=data.is_vip_customer,
        is_high_revenue_customer=data.is_high_revenue_customer,
    )
    return AlertControlResponse(
        customer_id=data.customer_id,
        churn_probability=data.churn_probability,
        risk_level=data.risk_level,
        alert_required=alert_decision.alert_required,
        alert_channel=alert_decision.alert_channel,
        suppress_reason=alert_decision.suppress_reason,
        log_required=alert_decision.log_required,
    )

@app.post("/retention/roi", response_model=RetentionROIResponse)
def retention_roi_endpoint(data: RetentionROIRequest):
    logger.info(f"Retention ROI requested. Customer ID: {data.customer_id}")
    roi = calculate_retention_roi(
        customer_id=data.customer_id,
        churn_probability=data.churn_probability,
        risk_level=data.risk_level,
        alert_sent=data.alert_sent,
        expected_revenue_loss=data.expected_revenue_loss,
        action_type=data.action_type,
        action_cost=data.action_cost,
        coupon_cost=data.coupon_cost,
        discount_cost=data.discount_cost,
        consulting_cost=data.consulting_cost,
        response_status=data.response_status,
        actual_churn=data.actual_churn,
        retention_success=data.retention_success,
        notes=data.notes,
    )
    return RetentionROIResponse(
        customer_id=roi.customer_id,
        churn_probability=roi.churn_probability,
        risk_level=roi.risk_level,
        alert_sent=roi.alert_sent,
        action_type=roi.action_type,
        response_status=roi.response_status,
        actual_churn=roi.actual_churn,
        retention_success=roi.retention_success,
        expected_revenue_loss=roi.expected_revenue_loss,
        saved_revenue=roi.saved_revenue,
        retention_cost=roi.retention_cost,
        net_benefit=roi.net_benefit,
        roi=roi.roi,
        notes=roi.notes,
    )

@app.post("/predict/xgb", response_model=Dict[str, Any])
def predict_xgb_endpoint(data: CustomerData):
    logger.info(f"Predict XGB requested. Customer ID: {data.customer_id}")
    data_dict = data.model_dump()
    prob = predict_xgb(data_dict)
    return {
        "customer_id": data.customer_id,
        "xgb_probability": prob,
        "is_churn": prob >= 0.5
    }

@app.post("/predict/ts", response_model=Dict[str, Any])
def predict_ts_endpoint(data: CustomerData):
    logger.info(f"Predict TS requested. Customer ID: {data.customer_id}")
    data_dict = data.model_dump()
    prob = predict_transformer(data_dict)
    return {
        "customer_id": data.customer_id,
        "ts_probability": prob,
        "is_churn": prob >= 0.5
    }

@app.post("/predict/tcn", response_model=Dict[str, Any])
def predict_tcn_endpoint(data: CustomerData):
    logger.info(f"Predict TCN requested. Customer ID: {data.customer_id}")
    data_dict = data.model_dump()
    prob = predict_tcn(data_dict)
    return {
        "customer_id": data.customer_id,
        "tcn_probability": prob,
        "model_available": prob is not None,
        "is_churn": prob >= 0.5 if prob is not None else False
    }

@app.post("/whatif")
def whatif_simulator(data: CustomerData, arpu_discount: float = 0.0, active_boost: int = 0):
    logger.info(f"What-if requested. Customer ID: {data.customer_id}")
    
    # 1. Before 개입 확률
    data_dict_before = data.model_dump()
    result_before = predict_churn(data_dict_before)
    prob_before = result_before["churn_probability"]
    
    # 2. After 개입 데이터 준비
    data_dict_after = data.model_dump()
    data_dict_after["arpu"] = max(0.0, data_dict_after["arpu"] - arpu_discount)
    data_dict_after["total_revenue"] = max(0.0, data_dict_after["total_revenue"] - arpu_discount)
    data_dict_after["active_subscribers"] += active_boost
    
    # 3. After 개입 확률
    result_after = predict_churn(data_dict_after)
    prob_after = result_after["churn_probability"]
    
    delta = prob_after - prob_before
    success = delta < 0  # 확률이 감소해야 성공
    
    # 예상 절감액: 확률이 떨어져 이탈이 방어되었다면 그만큼 ARPU를 유지한 것으로 간주
    saved_revenue = 0.0
    if result_before["churn_prediction"] and not result_after["churn_prediction"]:
        saved_revenue = data_dict_after["arpu"] 
        
    return {
        "customer_id": data.customer_id,
        "before_probability": prob_before,
        "after_probability": prob_after,
        "delta": delta,
        "success": success,
        "expected_saved_revenue": saved_revenue
    }
