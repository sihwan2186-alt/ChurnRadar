from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from api.alert_fatigue import classify_risk_level
from src.utils.helpers import first_existing_path, model_path

# 로거 설정
logger = logging.getLogger(__name__)

_XGB_MODEL_PATH = model_path("model.joblib")
_TS_MODEL_PATH = first_existing_path(
    model_path("churn_pro_engine.pth"),
    model_path("transformer_churn_v1.pth"),
) or model_path("churn_pro_engine.pth")
_TCN_MODEL_PATH = model_path("churn_tcn.pth")

_xgb_model: Optional[Any] = None
_ts_model: Optional[Any] = None
_tcn_model: Optional[Any] = None
_tcn_checkpoint: Optional[Dict[str, Any]] = None
_load_attempted_xgb = False
_load_attempted_ts = False
_load_attempted_tcn = False

def preload_models():
    """FastAPI startup 이벤트 시 메인 스레드에서 미리 로드하여 데드락 방지"""
    logger.info("모델 Pre-load 시작...")
    # 반드시 PyTorch(TS)를 먼저 로드해야 함! (libomp 충돌 방지)
    # _get_ts_model() # Mac 로컬 세그폴트 이슈로 주석 처리
    _get_xgb_model()
    logger.info("모델 Pre-load 완료")

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
            import torch
            import sys
            repo_root = str(Path(__file__).resolve().parent.parent)
            if repo_root not in sys.path:
                sys.path.append(repo_root)
            from src.models.ts_transformer import ChurnTransformer

            # API 추론은 배치 사이즈가 1이므로 CPU가 안전하고 빠름 (MPS 충돌 원천 차단)
            device = torch.device("cpu")
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

def _get_tcn_model() -> Optional[Any]:
    global _tcn_model, _tcn_checkpoint, _load_attempted_tcn
    if _load_attempted_tcn:
        return _tcn_model
    _load_attempted_tcn = True
    if _TCN_MODEL_PATH.is_file():
        try:
            import torch
            from src.models.tcn_model import ChurnTCN

            device = torch.device("cpu")
            checkpoint = torch.load(_TCN_MODEL_PATH, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                config = checkpoint.get("config", {})
                model = ChurnTCN(
                    input_size=int(config.get("input_size", 3)),
                    channels=tuple(config.get("channels", [32, 64])),
                    kernel_size=int(config.get("kernel_size", 3)),
                    dropout=float(config.get("dropout", 0.2)),
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                _tcn_checkpoint = checkpoint
            else:
                model = ChurnTCN(input_size=3, channels=(32, 64))
                model.load_state_dict(checkpoint)
                _tcn_checkpoint = {}

            model.to(device)
            model.eval()
            _tcn_model = model
            logger.info("TCN 모델이 성공적으로 로드되었습니다.")
        except Exception as e:
            logger.error(f"TCN 로드 실패: {e}")
    else:
        logger.warning(f"TCN 모델 파일이 없습니다: {_TCN_MODEL_PATH}")
    return _tcn_model

def get_risk_level(probability: float) -> str:
    return classify_risk_level(probability)

def _prepare_xgb_input(data: Dict[str, Any]) -> pd.DataFrame:
    # 파생 변수 계산
    total_subs = max(data["total_subs"], 1)  # 0으로 나누기 방지
    active_ratio = data["active_subscribers"] / total_subs
    inactive_ratio = data["not_active_subscribers"] / total_subs
    total_rev = data["total_revenue"] if data["total_revenue"] != 0 else 1.0
    mobile_revenue_ratio = data["avg_mobile_revenue"] / total_rev

    # 학습 피처와 완전히 일치 (tune_xgboost.py NUMERIC_FEATURES + CAT_FEATURES 순서)
    row = {
        "Total_SUBs":             data["total_subs"],
        "AvgMobileRevenue":       data["avg_mobile_revenue"],
        "AvgFIXRevenue":          data["avg_fix_revenue"],
        "TotalRevenue":           data["total_revenue"],
        "ARPU":                   data["arpu"],
        "Active_Ratio":           active_ratio,
        "Not_Active_subscribers": data["not_active_subscribers"],
        "Mobile_Revenue_Ratio":   mobile_revenue_ratio,
        "Inactive_Ratio":         inactive_ratio,
        "CRM_PID_Value_Segment":  data["crm_segment"],
        "EffectiveSegment":       data["effective_segment"],
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

def _prepare_ts_input(data: Dict[str, Any]) -> Any:
    import torch

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
    
    device = torch.device("cpu")
    tensor_3d = np.stack([energy, momentum, acceleration], axis=1) # (30, 3)
    return torch.FloatTensor(tensor_3d).unsqueeze(0).to(device)

def _prepare_tcn_input(data: Dict[str, Any]) -> Any:
    import torch

    ts_tensor = _prepare_ts_input(data).cpu().numpy()
    checkpoint = _tcn_checkpoint or {}
    scaler_mean = checkpoint.get("scaler_mean")
    scaler_scale = checkpoint.get("scaler_scale")
    if scaler_mean is not None and scaler_scale is not None:
        mean = np.array(scaler_mean, dtype=np.float32).reshape(1, 1, -1)
        scale = np.array(scaler_scale, dtype=np.float32).reshape(1, 1, -1)
        scale = np.where(scale == 0, 1.0, scale)
        ts_tensor = (ts_tensor - mean) / scale

    device = torch.device("cpu")
    return torch.FloatTensor(ts_tensor).to(device)

def predict_transformer(data: Dict[str, Any]) -> float:
    # [Mac 환경 PyTorch/XGBoost C++ 라이브러리(libomp) 충돌 방지]
    # 두 라이브러리를 동시 임포트하면 Mac에서 세그폴트가 발생하는 알려진 버그로 인해,
    # 로컬 테스트 환경에서는 TS 모델 예측을 더미값(0.20)으로 우회합니다.
    # 실제 서버(Linux/Docker) 배포 시에는 원래 코드를 사용하면 됩니다.
    return 0.20

def predict_tcn(data: Dict[str, Any], default: Optional[float] = None) -> Optional[float]:
    model = _get_tcn_model()
    if model is None:
        return default

    try:
        import torch

        tensor = _prepare_tcn_input(data)
        length = torch.tensor([min(len(data.get("history_arpu") or []), 30) or 30], dtype=torch.long)
        with torch.no_grad():
            logits = model(tensor, lengths=length)
            probability = torch.sigmoid(logits).cpu().item()
        return float(probability)
    except Exception as e:
        logger.error(f"TCN 예측 실패: {e}")
        return default


def predict_churn(data: Dict[str, Any]) -> Dict[str, Any]:
    xgb_prob = predict_xgb(data)
    tcn_prob = predict_tcn(data)
    ts_prob = predict_transformer(data)
    
    has_real_history = data.get("history_arpu") and len(data["history_arpu"]) >= 7
    if tcn_prob is not None and has_real_history:
        churn_prob = xgb_prob * 0.3 + tcn_prob * 0.3 + ts_prob * 0.4
    elif tcn_prob is not None:
        churn_prob = xgb_prob * 0.7 + tcn_prob * 0.2 + ts_prob * 0.1
    elif has_real_history:
        churn_prob = xgb_prob * 0.4 + ts_prob * 0.6  # 실제 시계열 데이터 → Transformer 위주
    else:
        churn_prob = xgb_prob * 0.8 + ts_prob * 0.2  # 시뮬레이션 데이터 → XGBoost 위주
    is_churn = churn_prob >= 0.5
    
    risk_level = get_risk_level(churn_prob)
    expected_revenue_loss = data["arpu"] if is_churn else 0.0
    
    return {
        "xgb_probability": xgb_prob,
        "tcn_probability": tcn_prob,
        "ts_probability": ts_prob,
        "churn_probability": churn_prob,
        "churn_prediction": is_churn,
        "risk_level": risk_level,
        "expected_revenue_loss": expected_revenue_loss
    }
