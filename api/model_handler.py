"""학습된 모델 로드 및 이탈 확률 추론. 모델 파일이 없으면 발표·연동용 더미 점수를 씁니다."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
_model: Optional[Any] = None
_load_attempted = False


def _get_model() -> Optional[Any]:
    """`models/model.joblib`가 있으면 로드, 없으면 None."""
    global _model, _load_attempted
    if _load_attempted:
        return _model
    _load_attempted = True
    if _MODEL_PATH.is_file():
        import joblib

        _model = joblib.load(_MODEL_PATH)
    return _model


CHURN_THRESHOLD = 0.80  # 이탈로 판별할 확률 임계값

def predict_churn(tenure: int, monthly_charges: float) -> Tuple[float, bool]:
    """
    학습 시 사용한 순서와 동일: (Total_SUBs, ARPU) ↔ API (tenure, monthly_charges).
    `models/model.joblib`가 없으면 더미 점수를 반환합니다.
    """
    model = _get_model()
    if model is None:
        probability = 0.85
        return probability, probability >= CHURN_THRESHOLD

    X = pd.DataFrame(
        [[tenure, monthly_charges]],
        columns=["Total_SUBs", "ARPU"],
    )
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        probability = float(proba[1] if proba.shape[-1] > 1 else proba[0])
    else:
        probability = float(model.predict(X)[0])
    return probability, probability >= CHURN_THRESHOLD
