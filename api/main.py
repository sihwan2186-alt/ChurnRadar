from fastapi import FastAPI
from api.schemas import CustomerData, ChurnPrediction

# FastAPI 앱 생성
app = FastAPI(title="ChurnRadar API", description="고객 이탈 예측 및 알림 연동 API")

@app.get("/")
def read_root():
    return {"message": "ChurnRadar API 서버 정상 작동 중!"}

# 이탈 예측 엔드포인트
@app.post("/predict", response_model=ChurnPrediction)
def predict_churn(data: CustomerData):

    # TODO: 나중에 모델 완성되면 아래 가짜 로직 대신 실제 예측 모델 적용

    # 뼈대용 가짜(Dummy) 로직: 예시로 85% 확률 반환
    dummy_probability = 0.85
    is_danger = True if dummy_probability >= 0.80 else False

    return {
        "customer_id": data.customer_id,
        "churn_probability": dummy_probability,
        "is_churn_expected": is_danger
    }
