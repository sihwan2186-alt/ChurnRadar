# PPT·발표용 코드 스니펫 모음

슬라이드에 붙여 넣기 쉽게 **저장소 실제 파일과 동일한 내용**을 모아 두었습니다.  
원본은 각 경로를 기준으로 유지·갱신합니다.

---

## 1. 요청·응답 스키마 (`api/schemas.py`)

```python
from pydantic import BaseModel

# 1. 고객 데이터 스키마
class CustomerData(BaseModel):
    customer_id: str
    tenure: int                 # 고객 가입 기간 (개월)
    monthly_charges: float      # 월 요금

# 2. 이탈 예측 결과 스키마
class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float    # 이탈 확률 (0~1)
    churn_prediction: bool     # 이탈 예측 결과 (True/False)
```

---

## 2. FastAPI 엔드포인트 (`api/main.py`)

```python
from fastapi import FastAPI

from api.model_handler import predict_churn
from api.schemas import ChurnPrediction, CustomerData

app = FastAPI(title="ChurnRadar API", description="고객 이탈 예측 및 알림 연동 API")


@app.get("/")
def read_root():
    return {"message": "ChurnRadar API 서버 정상 작동 중!"}


@app.post("/predict", response_model=ChurnPrediction)
def predict_churn_endpoint(data: CustomerData):
    probability, is_churn = predict_churn(data.tenure, data.monthly_charges)
    return ChurnPrediction(
        customer_id=data.customer_id,
        churn_probability=probability,
        churn_prediction=is_churn,
    )
```

---

## 3. 모델 로드·추론 분리 (`api/model_handler.py`)

발표에서 **API ↔ 모델 계층 분리**를 설명할 때 사용합니다.

```python
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
```

---

## 4. 기술 스택·의존성 (`requirements.txt`)

```text
fastapi
uvicorn
scikit-learn
pandas
imbalanced-learn
jupyter
```

---

## 5. 팀 에디터 공통 설정 발췌 (`.vscode/settings.json`)

**“개발 환경 통일”** 슬라이드용입니다.

```json
{
  "python.terminal.activateEnvironment": true,
  "jupyter.defaultKernel": "Python 3",
  "files.eol": "\n",
  "files.insertFinalNewline": true,
  "files.trimTrailingWhitespace": true,
  "editor.formatOnSave": false,
  "[windows]": {
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe"
  },
  "[osx]": {
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python"
  },
  "[linux]": {
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python"
  }
}
```

---

## 6. 권장 확장 (`.vscode/extensions.json`)

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "eamodio.gitlens",
    "GitHub.vscode-pull-request-github"
  ]
}
```

---

## 서버 실행 (발표 데모 시)

저장소 루트에서 가상환경 활성화 후:

```bash
uvicorn api.main:app --reload
```

브라우저에서 `http://127.0.0.1:8000/docs` 로 Swagger UI를 열 수 있습니다.
