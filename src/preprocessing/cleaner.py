import pandas as pd
import numpy as np
from typing import Tuple

TARGET = "CHURN"
NUMERIC_FEATURES = [
    "Total_SUBs", "AvgMobileRevenue", "AvgFIXRevenue",
    "TotalRevenue", "ARPU", "Active_Ratio", "Not_Active_subscribers",
    "Mobile_Revenue_Ratio", "Inactive_Ratio",
    "Suspended_Ratio", "Revenue_per_Active_Sub",
    "Inactive_x_Revenue", "Revenue_Balance",
]
CAT_FEATURES = ["CRM_PID_Value_Segment", "EffectiveSegment"]
FEATURE_COLS = NUMERIC_FEATURES + CAT_FEATURES

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    원시 데이터(DataFrame)를 받아 결측치 보간, 파생변수 생성 및 타겟 인코딩을 수행합니다.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # ARPU 결측 보간: TotalRevenue / Total_SUBs
    mask = df["ARPU"].isna() & df["Total_SUBs"].gt(0)
    df.loc[mask, "ARPU"] = df.loc[mask, "TotalRevenue"] / df.loc[mask, "Total_SUBs"]

    # 파생변수: 활성 구독자 비율
    df["Active_Ratio"] = df["Active_subscribers"] / df["Total_SUBs"].replace(0, np.nan)
    df["Active_Ratio"] = df["Active_Ratio"].fillna(0.0).clip(0.0, 1.0)

    # Not_Active_subscribers: 결측은 0으로 (비활성 구독자 없음으로 간주)
    df["Not_Active_subscribers"] = df["Not_Active_subscribers"].fillna(0.0)
    if "Suspended_subscribers" not in df.columns:
        df["Suspended_subscribers"] = 0.0
    df["Suspended_subscribers"] = df["Suspended_subscribers"].fillna(0.0)

    # 파생변수: 모바일 매출 비중
    df["Mobile_Revenue_Ratio"] = df["AvgMobileRevenue"] / df["TotalRevenue"].replace(0, np.nan)
    df["Mobile_Revenue_Ratio"] = df["Mobile_Revenue_Ratio"].fillna(0.0).clip(0.0, 1.0)

    # 파생변수: 비활성 구독자 비율
    df["Inactive_Ratio"] = df["Not_Active_subscribers"] / df["Total_SUBs"].replace(0, np.nan)
    df["Inactive_Ratio"] = df["Inactive_Ratio"].fillna(0.0).clip(0.0, 1.0)

    # 파생변수: 정지 구독자 비율. 정지 상태는 이탈 직전 상태일 수 있어 별도 신호로 둔다.
    df["Suspended_Ratio"] = df["Suspended_subscribers"] / df["Total_SUBs"].replace(0, np.nan)
    df["Suspended_Ratio"] = df["Suspended_Ratio"].fillna(0.0).clip(0.0, 1.0)

    # 파생변수: 활성 구독자 1명당 매출. 소수 활성 고객에게 매출이 집중된 계정을 분리한다.
    df["Revenue_per_Active_Sub"] = df["TotalRevenue"] / df["Active_subscribers"].replace(0, np.nan)
    df["Revenue_per_Active_Sub"] = df["Revenue_per_Active_Sub"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 파생변수: 비활성도와 매출 규모의 교호작용.
    df["Inactive_x_Revenue"] = df["Inactive_Ratio"] * df["TotalRevenue"].fillna(0.0)

    # 파생변수: 모바일/유선 매출 균형도. 0에 가까울수록 한 서비스 의존도가 높다.
    revenue_pair = df[["AvgMobileRevenue", "AvgFIXRevenue"]].fillna(0.0)
    df["Revenue_Balance"] = revenue_pair.min(axis=1) / (revenue_pair.max(axis=1) + 1e-5)
    df["Revenue_Balance"] = df["Revenue_Balance"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    # 범주형 결측 → 'Unknown'
    for col in CAT_FEATURES:
        df[col] = df[col].fillna("Unknown")

    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    
    X = df.loc[valid, FEATURE_COLS].reset_index(drop=True)
    y = y[valid].astype(int).reset_index(drop=True)
    
    return X, y
