#!/usr/bin/env python3
"""
Train and compare churn classifiers.

The candidate set includes standard class-weighted models, SMOTE + RandomForest,
and imbalance-aware ensembles from imbalanced-learn:
BalancedRandomForestClassifier and EasyEnsembleClassifier.

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

from src.models import (
    build_balanced_random_forest,
    build_catboost,
    build_easy_ensemble,
    build_lgbm,
    build_lr,
    build_rf,
    build_voting_ensemble,
    build_xgb,
)
from src.utils.helpers import model_path, raw_data_path, resolve_input_path

DEFAULT_CSV = raw_data_path("baza_telecom_v2.csv")
MODEL_OUT = model_path("model.joblib")

NUMERIC_FEATURES = [
    "Total_SUBs",
    "AvgMobileRevenue",
    "AvgFIXRevenue",
    "TotalRevenue",
    "ARPU",
    "Active_Ratio",
    "Not_Active_subscribers",
    "Mobile_Revenue_Ratio",
    "Inactive_Ratio",
]
CAT_FEATURES = ["CRM_PID_Value_Segment", "EffectiveSegment"]
FEATURE_COLS = NUMERIC_FEATURES + CAT_FEATURES
TARGET = "CHURN"


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    mask = df["ARPU"].isna() & df["Total_SUBs"].gt(0)
    df.loc[mask, "ARPU"] = df.loc[mask, "TotalRevenue"] / df.loc[mask, "Total_SUBs"]

    df["Active_Ratio"] = df["Active_subscribers"] / df["Total_SUBs"].replace(0, np.nan)
    df["Active_Ratio"] = df["Active_Ratio"].fillna(0.0).clip(0.0, 1.0)

    df["Not_Active_subscribers"] = df["Not_Active_subscribers"].fillna(0.0)

    df["Mobile_Revenue_Ratio"] = df["AvgMobileRevenue"] / df["TotalRevenue"].replace(0, np.nan)
    df["Mobile_Revenue_Ratio"] = df["Mobile_Revenue_Ratio"].fillna(0.0).clip(0.0, 1.0)

    df["Inactive_Ratio"] = df["Not_Active_subscribers"] / df["Total_SUBs"].replace(0, np.nan)
    df["Inactive_Ratio"] = df["Inactive_Ratio"].fillna(0.0).clip(0.0, 1.0)

    for col in CAT_FEATURES:
        df[col] = df[col].fillna("Unknown")

    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    return df.loc[valid, FEATURE_COLS].reset_index(drop=True), y[valid].astype(int).reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
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


def make_pipeline(model) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", model),
    ])


def make_smote_pipeline(model) -> ImbPipeline:
    return ImbPipeline([
        ("prep", make_preprocessor()),
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("model", model),
    ])


def make_model_pipelines() -> dict[str, Pipeline | ImbPipeline]:
    return {
        "LR": make_pipeline(build_lr()),
        "RF_SMOTE": make_smote_pipeline(build_rf(max_depth=6)),
        "BalancedRF": make_pipeline(build_balanced_random_forest(max_depth=6)),
        "EasyEnsemble": make_pipeline(build_easy_ensemble(n_estimators=10)),
        "XGBoost": make_pipeline(build_xgb(n_estimators=300, scale_pos_weight=14.4)),
        "LightGBM": make_pipeline(build_lgbm(n_estimators=300)),
        "CatBoost": make_pipeline(build_catboost(cat_features=[], iterations=500)),
    }


def evaluate(name: str, model, X_test, y_test) -> float:
    y_pred = model.predict(X_test)
    auc = float("nan")

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], zero_division=0))
    print(f"  F1 (Churn): {f1:.4f}  |  AUC: {auc:.4f}")
    return f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=MODEL_OUT)
    args = parser.parse_args()
    args.csv = resolve_input_path(args.csv, DEFAULT_CSV)
    if not args.out.is_absolute():
        args.out = REPO_ROOT / args.out

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    X, y = load_data(args.csv)
    print(f"Rows: {X.shape[0]} | Churn rate: {y.mean():.2%}")
    print(f"Features: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Train churn count: {y_train.sum()}")

    models = make_model_pipelines()
    scores: dict[str, float] = {}
    for name, pipeline in models.items():
        print(f"\n[Training] {name}...", flush=True)
        pipeline.fit(X_train, y_train)
        scores[name] = evaluate(name, pipeline, X_test, y_test)

    print("\n[Training] Voting Ensemble (LR+RF_SMOTE+BalancedRF+EasyEnsemble+XGBoost+LightGBM)...", flush=True)
    voting_members = {
        name: make_model_pipelines()[name]
        for name in ["LR", "RF_SMOTE", "BalancedRF", "EasyEnsemble", "XGBoost", "LightGBM"]
    }
    ensemble = build_voting_ensemble(voting_members)
    ensemble.fit(X_train, y_train)
    scores["Ensemble"] = evaluate("Voting Ensemble", ensemble, X_test, y_test)

    best_single = max((k for k in scores if k != "Ensemble"), key=lambda k: scores[k])
    print(f"\n[10-Fold CV] {best_single}...")
    cv_pipe = make_model_pipelines()[best_single]
    cv_scores = cross_val_score(
        cv_pipe,
        X,
        y,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        scoring="f1",
        n_jobs=-1,
    )
    print(f"  CV F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    print(f"\n{'=' * 50}")
    print("  F1 Score Summary")
    print(f"{'=' * 50}")
    for name, f1 in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<15} {f1:.4f}")

    best_name = max(scores, key=lambda k: scores[k])
    best_model = ensemble if best_name == "Ensemble" else models[best_name]
    print(f"\nBest model: {best_name} (F1={scores[best_name]:.4f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
