"""학습된 모델 로드 및 이탈 확률 추론. 모델 파일이 없으면 발표·연동용 더미 점수를 씁니다."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

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


def predict_churn(tenure: int, monthly_charges: float) -> Tuple[float, bool]:
    """
    tenure, monthly_charges 두 특성으로 이탈 확률과 이진 판정을 반환합니다.
    실제 학습 파이프라인과 특성 개수·순서가 맞아야 합니다.
    """
    model = _get_model()
    if model is None:
        probability = 0.85
        return probability, probability >= 0.80

    X = np.array([[tenure, monthly_charges]], dtype=float)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        probability = float(proba[1] if proba.shape[-1] > 1 else proba[0])
    else:
        probability = float(model.predict(X)[0])
    return probability, probability >= 0.80
