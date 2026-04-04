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
