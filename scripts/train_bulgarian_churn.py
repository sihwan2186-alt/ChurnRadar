#!/usr/bin/env python3
"""
불가리아 B2B 텔레콤 CRM CSV(`data/raw/bulgarian_telco_churn.csv`)로 이탈 모델을 학습해
`models/model.joblib`에 저장합니다.

API `CustomerData`와 동일한 2개 특성 순서로 맞춥니다.
- tenure → CSV의 Total_SUBs (총 구독 회선·규모 지표)
- monthly_charges → ARPU (비어 있으면 TotalRevenue / Total_SUBs 등으로 보간)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "bulgarian_telco_churn.csv"
MODEL_OUT = REPO_ROOT / "models" / "model.joblib"


def load_and_clean(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "CHURN" not in df.columns:
        raise ValueError(f"CHURN 열이 없습니다. 열 목록: {list(df.columns)}")

    for col in ("Total_SUBs", "TotalRevenue", "AvgMobileRevenue"):
        if col not in df.columns:
            raise ValueError(f"필수 열 '{col}' 이(가) 없습니다.")

    subs = pd.to_numeric(df["Total_SUBs"], errors="coerce")
    revenue = pd.to_numeric(df["TotalRevenue"], errors="coerce")
    arpu = pd.to_numeric(df["ARPU"], errors="coerce") if "ARPU" in df.columns else pd.Series(np.nan, index=df.index)
    mobile = pd.to_numeric(df["AvgMobileRevenue"], errors="coerce")

    # ARPU 누락 시: TotalRevenue / Total_SUBs, 그다음 모바일 평균 요금
    implied = revenue / subs.replace(0, np.nan)
    arpu_filled = arpu.combine_first(implied).combine_first(mobile)

    y_raw = df["CHURN"].astype(str).str.strip().str.lower()
    y = y_raw.map({"yes": 1, "no": 0, "1": 1, "0": 0})
    mask = y.notna() & subs.notna() & arpu_filled.notna() & (subs >= 0)
    X = pd.DataFrame({"Total_SUBs": subs[mask], "ARPU": arpu_filled[mask]})
    y = y[mask].astype(int).to_numpy()
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="불가리아 텔레콤 CSV로 이탈 모델 학습")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"입력 CSV 경로 (기본: {DEFAULT_CSV})",
    )
    parser.add_argument("--out", type=Path, default=MODEL_OUT, help="저장할 .joblib 경로")
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV를 찾을 수 없습니다: {args.csv}")

    X, y = load_and_clean(args.csv)
    if len(X) < 50:
        raise SystemExit(f"학습 행이 너무 적습니다: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"], zero_division=0))
    print(f"F1 (Churn=1): {f1:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
