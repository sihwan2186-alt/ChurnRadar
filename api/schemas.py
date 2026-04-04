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
