from pydantic import BaseModel, Field

# 1. 고객 데이터 스키마
# 불가리아 B2B 텔레콤 원본 CSV와 맞춤: tenure=Total_SUBs, monthly_charges=ARPU
class CustomerData(BaseModel):
    customer_id: str
    tenure: int = Field(
        ...,
        description="원본 데이터 Total_SUBs(총 구독·회선 규모)와 동일 의미로 사용",
    )
    monthly_charges: float = Field(
        ...,
        description="원본 데이터 ARPU(가입자당 평균 매출)와 동일 의미로 사용",
    )

# 2. 이탈 예측 결과 스키마
class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float    # 이탈 확률 (0~1)
    churn_prediction: bool     # 이탈 예측 결과 (True/False)
