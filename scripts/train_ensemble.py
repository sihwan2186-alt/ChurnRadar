#!/usr/bin/env python3
"""
5개 모델(LR, RF, XGBoost, LightGBM, CatBoost) 학습 및 비교.
최고 F1 모델을 models/model.joblib에 저장합니다.

Usage:
    python scripts/train_ensemble.py
    python scripts/train_ensemble.py --csv data/raw/baza_telecom_v2.csv --out models/model.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models import build_catboost, build_lgbm, build_lr, build_rf, build_voting_ensemble, build_xgb

DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "baza_telecom_v2.csv"
MODEL_OUT = REPO_ROOT / "models" / "model.joblib"

NUMERIC_FEATURES = ["Total_SUBs", "AvgMobileRevenue", "AvgFIXRevenue", "TotalRevenue", "ARPU"]
CAT_FEATURES = ["CRM_PID_Value_Segment", "EffectiveSegment", "KA_name"]
FEATURE_COLS = NUMERIC_FEATURES + CAT_FEATURES
TARGET = "CHURN"


# ── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # ARPU 결측 보간: TotalRevenue / Total_SUBs
    mask = df["ARPU"].isna() & df["Total_SUBs"].gt(0)
    df.loc[mask, "ARPU"] = df.loc[mask, "TotalRevenue"] / df.loc[mask, "Total_SUBs"]

    # 범주형 결측 → 'Unknown'
    for col in CAT_FEATURES:
        df[col] = df[col].fillna("Unknown")

    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    return df.loc[valid, FEATURE_COLS].reset_index(drop=True), y[valid].astype(int).reset_index(drop=True)


# ── 전처리 파이프라인 ─────────────────────────────────────────────────────────

def make_standard_preprocessor() -> ColumnTransformer:
    """LR / RF / XGBoost / LightGBM 용 전처리기."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES),
    ])


def make_catboost_preprocessor() -> ColumnTransformer:
    """CatBoost 용: 수치형 결측만 처리, 범주형은 string 그대로."""
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown"))])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES),
    ], remainder="drop")


# ── 평가 ─────────────────────────────────────────────────────────────────────

def evaluate(name: str, model, X_test, y_test) -> float:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = float("nan")
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], zero_division=0))
    print(f"  F1 (Churn): {f1:.4f}  |  AUC: {auc:.4f}")
    return f1


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=MODEL_OUT)
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV 없음: {args.csv}")

    X, y = load_data(args.csv)
    print(f"데이터: {X.shape[0]}행 | Churn 비율: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    std_prep = make_standard_preprocessor()
    cb_prep = make_catboost_preprocessor()
    cat_feature_indices = list(range(len(NUMERIC_FEATURES), len(FEATURE_COLS)))

    models = {
        "LR": Pipeline([("prep", make_standard_preprocessor()), ("model", build_lr())]),
        "RF": Pipeline([("prep", make_standard_preprocessor()), ("model", build_rf())]),
        "XGBoost": Pipeline([("prep", make_standard_preprocessor()), ("model", build_xgb())]),
        "LightGBM": Pipeline([("prep", make_standard_preprocessor()), ("model", build_lgbm())]),
        "CatBoost": Pipeline([
            ("prep", make_catboost_preprocessor()),
            ("model", build_catboost(cat_features=cat_feature_indices)),
        ]),
    }

    scores: dict[str, float] = {}
    for name, pipeline in models.items():
        print(f"\n[학습 중] {name}...", flush=True)
        pipeline.fit(X_train, y_train)
        scores[name] = evaluate(name, pipeline, X_test, y_test)

    # Voting Ensemble: CatBoost는 sklearn clone 미호환 → 4개 모델로 구성
    fresh_ensemble_models = {
        "LR": Pipeline([("prep", make_standard_preprocessor()), ("model", build_lr())]),
        "RF": Pipeline([("prep", make_standard_preprocessor()), ("model", build_rf())]),
        "XGBoost": Pipeline([("prep", make_standard_preprocessor()), ("model", build_xgb())]),
        "LightGBM": Pipeline([("prep", make_standard_preprocessor()), ("model", build_lgbm())]),
    }
    print(f"\n[학습 중] Voting Ensemble (LR+RF+XGBoost+LightGBM)...", flush=True)
    ensemble = build_voting_ensemble(fresh_ensemble_models)
    ensemble.fit(X_train, y_train)
    scores["Ensemble"] = evaluate("Voting Ensemble", ensemble, X_test, y_test)

    # 결과 요약
    print(f"\n{'='*50}")
    print("  F1 Score 요약")
    print(f"{'='*50}")
    for name, f1 in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<15} {f1:.4f}")

    best_name = max(scores, key=lambda k: scores[k])
    best_model = ensemble if best_name == "Ensemble" else models[best_name]
    print(f"\n최고 모델: {best_name} (F1={scores[best_name]:.4f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.out)
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
