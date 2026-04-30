#!/usr/bin/env python3
"""
저장된 XGBoost 모델에 대해 SHAP 분석 실행.
피처 중요도를 출력하고 plots/shap_*.png로 저장합니다.

Usage:
    python scripts/explain_shap.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.explain import plot_bar, plot_summary

NUMERIC_FEATURES = [
    "Total_SUBs", "AvgMobileRevenue", "AvgFIXRevenue",
    "TotalRevenue", "ARPU", "Active_Ratio", "Not_Active_subscribers",
    "Mobile_Revenue_Ratio", "Inactive_Ratio",
]
CAT_FEATURES = ["CRM_PID_Value_Segment", "EffectiveSegment"]


def load_sample(path: Path, n: int = 500) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    mask = df["ARPU"].isna() & df["Total_SUBs"].gt(0)
    df.loc[mask, "ARPU"] = df.loc[mask, "TotalRevenue"] / df.loc[mask, "Total_SUBs"]

    df["Active_Ratio"] = (df["Active_subscribers"] / df["Total_SUBs"].replace(0, np.nan)).fillna(0.0).clip(0, 1)
    df["Not_Active_subscribers"] = df["Not_Active_subscribers"].fillna(0.0)
    df["Mobile_Revenue_Ratio"] = (df["AvgMobileRevenue"] / df["TotalRevenue"].replace(0, np.nan)).fillna(0.0).clip(0, 1)
    df["Inactive_Ratio"] = (df["Not_Active_subscribers"] / df["Total_SUBs"].replace(0, np.nan)).fillna(0.0).clip(0, 1)

    for col in CAT_FEATURES:
        df[col] = df[col].fillna("Unknown")

    feature_cols = NUMERIC_FEATURES + CAT_FEATURES
    return df[feature_cols].dropna().sample(min(n, len(df)), random_state=42)


def main() -> None:
    model_path = REPO_ROOT / "models" / "model.joblib"
    if not model_path.is_file():
        raise SystemExit(f"모델 없음: {model_path}\n먼저 python scripts/train_ensemble.py 실행하세요.")

    model = joblib.load(model_path)
    X_sample = load_sample(REPO_ROOT / "data" / "raw" / "baza_telecom_v2.csv")
    print(f"SHAP 분석 샘플: {len(X_sample)}행")

    plots_dir = REPO_ROOT / "plots"

    print("Summary plot 생성 중...")
    plot_summary(model, X_sample, NUMERIC_FEATURES, CAT_FEATURES,
                 save_path=plots_dir / "shap_summary.png")

    print("Bar plot 생성 중...")
    importance = plot_bar(model, X_sample, NUMERIC_FEATURES, CAT_FEATURES,
                          save_path=plots_dir / "shap_bar.png")

    print("\n=== 피처 중요도 (Mean |SHAP|) ===")
    for feat, val in importance.items():
        print(f"  {feat:<30} {val:.4f}")


if __name__ == "__main__":
    main()
