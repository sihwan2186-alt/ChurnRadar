import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.alert_fatigue import evaluate_alert_fatigue
from api.model_handler import predict_churn, predict_tcn, predict_xgb, predict_transformer
from api.schemas import (
    AlertControlRequest,
    AlertControlResponse,
    ChurnPrediction,
    CustomerData,
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

@app.post("/predict", response_model=ChurnPrediction)
def predict_churn_endpoint(data: CustomerData):
    logger.info(f"Predict requested. Customer ID: {data.customer_id}")
    data_dict = data.model_dump()
    result = predict_churn(data_dict)
    alert_decision = evaluate_alert_fatigue(
        risk_level=result["risk_level"],
        churn_probability=result["churn_probability"],
        last_alert_time=data.last_alert_time,
        response_status=data.response_status,
        previous_churn_probability=data.previous_churn_probability,
        previous_risk_level=data.previous_risk_level,
        is_vip_customer=data.is_vip_customer or data.crm_segment.upper() == "VIP",
        is_high_revenue_customer=data.is_high_revenue_customer,
    )
    
    return ChurnPrediction(
        customer_id=data.customer_id,
        **result,
        alert_required=alert_decision.alert_required,
        alert_channel=alert_decision.alert_channel,
        suppress_reason=alert_decision.suppress_reason,
        log_required=alert_decision.log_required,
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
