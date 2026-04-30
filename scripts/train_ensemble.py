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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models import build_catboost, build_lgbm, build_lr, build_rf, build_voting_ensemble, build_xgb

DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "baza_telecom_v2.csv"
MODEL_OUT = REPO_ROOT / "models" / "model.joblib"

NUMERIC_FEATURES = [
    "Total_SUBs", "AvgMobileRevenue", "AvgFIXRevenue",
    "TotalRevenue", "ARPU", "Active_Ratio", "Not_Active_subscribers",
    "Mobile_Revenue_Ratio", "Inactive_Ratio",
]
CAT_FEATURES = ["CRM_PID_Value_Segment", "EffectiveSegment"]  # KA_name 제거 (ID성 노이즈)
FEATURE_COLS = NUMERIC_FEATURES + CAT_FEATURES
TARGET = "CHURN"

# ColumnTransformer 출력 기준 범주형 컬럼 인덱스 (수치형 뒤에 위치)
CAT_FEATURE_INDICES = list(range(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES) + len(CAT_FEATURES)))


# ── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # ARPU 결측 보간: TotalRevenue / Total_SUBs
    mask = df["ARPU"].isna() & df["Total_SUBs"].gt(0)
    df.loc[mask, "ARPU"] = df.loc[mask, "TotalRevenue"] / df.loc[mask, "Total_SUBs"]

    # 파생변수: 활성 구독자 비율
    df["Active_Ratio"] = df["Active_subscribers"] / df["Total_SUBs"].replace(0, np.nan)
    df["Active_Ratio"] = df["Active_Ratio"].fillna(0.0).clip(0.0, 1.0)

    # Not_Active_subscribers: 결측은 0으로 (비활성 구독자 없음으로 간주)
    df["Not_Active_subscribers"] = df["Not_Active_subscribers"].fillna(0.0)

    # 파생변수: 모바일 매출 비중
    df["Mobile_Revenue_Ratio"] = df["AvgMobileRevenue"] / df["TotalRevenue"].replace(0, np.nan)
    df["Mobile_Revenue_Ratio"] = df["Mobile_Revenue_Ratio"].fillna(0.0).clip(0.0, 1.0)

    # 파생변수: 비활성 구독자 비율
    df["Inactive_Ratio"] = df["Not_Active_subscribers"] / df["Total_SUBs"].replace(0, np.nan)
    df["Inactive_Ratio"] = df["Inactive_Ratio"].fillna(0.0).clip(0.0, 1.0)

    # 범주형 결측 → 'Unknown'
    for col in CAT_FEATURES:
        df[col] = df[col].fillna("Unknown")

    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    return df.loc[valid, FEATURE_COLS].reset_index(drop=True), y[valid].astype(int).reset_index(drop=True)


# ── 전처리 파이프라인 ─────────────────────────────────────────────────────────

def make_preprocessor() -> ColumnTransformer:
    """수치형: median impute + StandardScale / 범주형: Unknown fill + OrdinalEncode."""
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


def make_smote_pipeline(model) -> ImbPipeline:
    """전처리 → SMOTE (train only) → 모델 순서의 imbalanced-learn Pipeline."""
    return ImbPipeline([
        ("prep", make_preprocessor()),
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("model", model),
    ])


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
    print(f"피처: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]}행 | Test: {X_test.shape[0]}행")
    print(f"Train Churn: {y_train.sum()}건 → SMOTE 후 균형 맞춤")

    models = {
        # LR: class_weight="balanced"로 불균형 처리
        "LR":       Pipeline([("prep", make_preprocessor()), ("model", build_lr())]),
        # RF: SMOTE 적용
        "RF":       make_smote_pipeline(build_rf(max_depth=6)),
        # 부스팅 모델: SMOTE 제거, scale_pos_weight 복원
        "XGBoost":  Pipeline([("prep", make_preprocessor()), ("model", build_xgb(n_estimators=300, scale_pos_weight=14.4))]),
        "LightGBM": Pipeline([("prep", make_preprocessor()), ("model", build_lgbm(n_estimators=300))]),
        "CatBoost": Pipeline([("prep", make_preprocessor()), ("model", build_catboost(cat_features=[], iterations=500))]),
    }

    scores: dict[str, float] = {}
    for name, pipeline in models.items():
        print(f"\n[학습 중] {name}...", flush=True)
        pipeline.fit(X_train, y_train)
        scores[name] = evaluate(name, pipeline, X_test, y_test)

    # Voting Ensemble: CatBoost sklearn clone 미호환 → 4개 모델로 구성
    print(f"\n[학습 중] Voting Ensemble (LR+RF+XGBoost+LightGBM)...", flush=True)
    fresh_models = {
        "LR":       Pipeline([("prep", make_preprocessor()), ("model", build_lr())]),
        "RF":       make_smote_pipeline(build_rf(max_depth=6)),
        "XGBoost":  Pipeline([("prep", make_preprocessor()), ("model", build_xgb(n_estimators=300, scale_pos_weight=14.4))]),
        "LightGBM": Pipeline([("prep", make_preprocessor()), ("model", build_lgbm(n_estimators=300))]),
    }
    ensemble = build_voting_ensemble(fresh_models)
    ensemble.fit(X_train, y_train)
    scores["Ensemble"] = evaluate("Voting Ensemble", ensemble, X_test, y_test)

    # 10-Fold Stratified CV (최고 단일 모델 기준)
    best_single = max((k for k in scores if k != "Ensemble"), key=lambda k: scores[k])
    print(f"\n[10-Fold CV] {best_single}...")
    if best_single == "RF":
        cv_pipe = make_smote_pipeline(build_rf(max_depth=6))
    elif best_single == "LR":
        cv_pipe = Pipeline([("prep", make_preprocessor()), ("model", build_lr())])
    elif best_single == "XGBoost":
        cv_pipe = Pipeline([("prep", make_preprocessor()), ("model", build_xgb(n_estimators=300, scale_pos_weight=14.4))])
    elif best_single == "LightGBM":
        cv_pipe = Pipeline([("prep", make_preprocessor()), ("model", build_lgbm(n_estimators=300))])
    else:
        cv_pipe = Pipeline([("prep", make_preprocessor()), ("model", build_catboost(cat_features=[], iterations=500))])
    cv_scores = cross_val_score(
        cv_pipe, X, y,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        scoring="f1", n_jobs=-1,
    )
    print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

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
