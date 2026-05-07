from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import torch
from src.utils.helpers import first_existing_path, model_path

# 로거 설정
logger = logging.getLogger(__name__)

_XGB_MODEL_PATH = model_path("model.joblib")
_TS_MODEL_PATH = first_existing_path(
    model_path("transformer_churn_v1.pth"),
    model_path("churn_pro_engine.pth"),
) or model_path("transformer_churn_v1.pth")

_xgb_model: Optional[Any] = None
_ts_model: Optional[Any] = None
_load_attempted_xgb = False
_load_attempted_ts = False

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def _get_xgb_model() -> Optional[Any]:
    global _xgb_model, _load_attempted_xgb
    if _load_attempted_xgb:
        return _xgb_model
    _load_attempted_xgb = True
    if _XGB_MODEL_PATH.is_file():
        import joblib
        _xgb_model = joblib.load(_XGB_MODEL_PATH)
        logger.info("XGBoost 모델을 성공적으로 로드했습니다.")
    else:
        logger.warning(f"XGBoost 모델 파일이 없습니다: {_XGB_MODEL_PATH}")
    return _xgb_model

def _get_ts_model() -> Optional[Any]:
    global _ts_model, _load_attempted_ts
    if _load_attempted_ts:
        return _ts_model
    _load_attempted_ts = True
    if _TS_MODEL_PATH.is_file():
        try:
            # src.models.ts_transformer 에서 ChurnTransformer 로드
            import sys
            repo_root = str(Path(__file__).resolve().parent.parent)
            if repo_root not in sys.path:
                sys.path.append(repo_root)
            from src.models.ts_transformer import ChurnTransformer
            
            # 모델 인스턴스화 (입력 차원 3)
            model = ChurnTransformer(input_size=3, d_model=64, nhead=4, num_layers=2)
            model.load_state_dict(torch.load(_TS_MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()
            _ts_model = model
            logger.info("TS-Transformer 모델을 성공적으로 로드했습니다.")
        except Exception as e:
            logger.error(f"TS-Transformer 로드 실패: {e}")
    else:
        logger.warning(f"TS-Transformer 모델 파일이 없습니다: {_TS_MODEL_PATH}")
    return _ts_model

def get_risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "HIGH"
    elif probability >= 0.5:
        return "MEDIUM"
    return "LOW"

def _prepare_xgb_input(data: Dict[str, Any]) -> pd.DataFrame:
    # 파생 변수 계산
    total_subs = max(data["total_subs"], 1) # 0으로 나누기 방지
    active_ratio = data["active_subscribers"] / total_subs
    inactive_ratio = data["not_active_subscribers"] / total_subs
    total_rev = data["total_revenue"] if data["total_revenue"] != 0 else 1.0
    mobile_revenue_ratio = data["avg_mobile_revenue"] / total_rev
    
    row = {
        "Total_SUBs": data["total_subs"],
        "AvgMobileRevenue": data["avg_mobile_revenue"],
        "AvgFIXRevenue": data["avg_fix_revenue"],
        "TotalRevenue": data["total_revenue"],
        "ARPU": data["arpu"],
        "Active_subscribers": data["active_subscribers"],
        "Not_Active_subscribers": data["not_active_subscribers"],
        "CRM_PID_Value_Segment": data["crm_segment"],
        "EffectiveSegment": data["effective_segment"],
        "Active_Ratio": active_ratio,
        "Inactive_Ratio": inactive_ratio,
        "Mobile_Revenue_Ratio": mobile_revenue_ratio
    }
    return pd.DataFrame([row])

def predict_xgb(data: Dict[str, Any]) -> float:
    model = _get_xgb_model()
    if model is None:
        return 0.82 # 더미 점수
    
    df = _prepare_xgb_input(data)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        return float(proba[1] if proba.shape[-1] > 1 else proba[0])
    return float(model.predict(df)[0])

def _prepare_ts_input(data: Dict[str, Any]) -> torch.Tensor:
    # 30일 시계열 데이터 시뮬레이션
    history = data.get("history_arpu")
    if history and len(history) > 0:
        # 클라이언트가 제공한 배열 사용 (최대 30일)
        energy_arr = np.array(history[-30:])
        if len(energy_arr) < 30:
            energy_arr = np.pad(energy_arr, (30 - len(energy_arr), 0), 'edge')
    else:
        # 값이 없을 경우 ARPU와 Active Ratio 기반으로 단순 시뮬레이션
        # 30일간 ARPU가 일정 비율로 감소/유지되는 궤적 역산
        current_arpu = data["arpu"]
        active_ratio = data["active_subscribers"] / max(data["total_subs"], 1)
        # Active Ratio가 낮으면 과거보다 ARPU가 떨어졌을 것이라고 가정
        decay_factor = 1.0 if active_ratio > 0.8 else 0.95
        energy_arr = [current_arpu / (decay_factor ** i) for i in range(29, -1, -1)]
        energy_arr = np.array(energy_arr)
    
    energy = energy_arr
    avg_past = np.mean(energy) if np.mean(energy) != 0 else 1.0
    momentum = energy / avg_past
    acceleration = np.diff(momentum, prepend=momentum[0])
    
    tensor_3d = np.stack([energy, momentum, acceleration], axis=1) # (30, 3)
    return torch.FloatTensor(tensor_3d).unsqueeze(0).to(device)

def predict_transformer(data: Dict[str, Any]) -> float:
    model = _get_ts_model()
    if model is None:
        return 0.88 # 더미 점수
    
    tensor_input = _prepare_ts_input(data)
    with torch.no_grad():
        logits = model(tensor_input)
        prob = torch.sigmoid(logits).item()
    return prob

def predict_churn(data: Dict[str, Any]) -> Dict[str, Any]:
    xgb_prob = predict_xgb(data)
    ts_prob = predict_transformer(data)
    
    churn_prob = (xgb_prob + ts_prob) / 2.0
    is_churn = churn_prob >= 0.5
    
    risk_level = get_risk_level(churn_prob)
    expected_revenue_loss = data["arpu"] if is_churn else 0.0
    
    return {
        "xgb_probability": xgb_prob,
        "ts_probability": ts_prob,
        "churn_probability": churn_prob,
        "churn_prediction": is_churn,
        "risk_level": risk_level,
        "expected_revenue_loss": expected_revenue_loss
    }
