import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.model_handler import predict_churn
from api.schemas import ChurnPrediction, CustomerData

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ChurnRadar API", description="고객 이탈 예측 및 알림 연동 API")

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시 특정 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    logger.info("Health check endpoint '/' accessed.")
    return {"message": "ChurnRadar API 서버 정상 작동 중!"}


@app.post("/predict", response_model=ChurnPrediction)
def predict_churn_endpoint(data: CustomerData):
    logger.info(f"Predict requested. Customer ID: {data.customer_id}")
    probability, is_churn = predict_churn(data.tenure, data.monthly_charges)
    logger.info(f"Predicted -> Probability: {probability:.4f}, Churn: {is_churn}")

    return ChurnPrediction(
        customer_id=data.customer_id,
        churn_probability=probability,
        churn_prediction=is_churn,
    )
