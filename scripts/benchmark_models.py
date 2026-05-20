#!/usr/bin/env python3
"""
Benchmark a broad set of churn classifiers.

This script is intentionally wider than scripts/train_ensemble.py. It compares
many model families, imbalance strategies, feature sets, and F1 thresholds, then
saves a sortable results table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import (
    BalancedBaggingClassifier,
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, RandomOverSampler, SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

from src.utils.helpers import model_path, raw_data_path, resolve_input_path, result_path

RANDOM_STATE = 42
TARGET = "CHURN"

CORE_NUMERIC_FEATURES = [
    "Total_SUBs",
    "AvgMobileRevenue",
    "AvgFIXRevenue",
    "TotalRevenue",
    "ARPU",
    "Active_Ratio",
    "Not_Active_subscribers",
    "Mobile_Revenue_Ratio",
    "Inactive_Ratio",
    "Suspended_Ratio",
    "Revenue_per_Active_Sub",
    "Inactive_x_Revenue",
    "Revenue_Balance",
]
CORE_CATEGORICAL_FEATURES = ["CRM_PID_Value_Segment", "EffectiveSegment"]
FULL_EXTRA_NUMERIC_FEATURES = ["Suspended_subscribers"]
FULL_EXTRA_CATEGORICAL_FEATURES = ["Billing_ZIP", "KA_name"]
GEO_NUMERIC_FEATURES = [
    "bg_zip_found",
    "bg_place_count",
    "bg_latitude",
    "bg_longitude",
    "bg_geo_accuracy",
    "bg_distance_sofia_km",
    "bg_distance_plovdiv_km",
    "bg_distance_varna_km",
    "bg_distance_burgas_km",
    "bg_min_big_city_distance_km",
]
GEO_CATEGORICAL_FEATURES = [
    "Billing_ZIP_norm",
    "bg_place_name",
    "bg_admin1_name",
    "bg_admin1_code",
    "bg_admin2_name",
    "bg_admin2_code",
    "bg_admin3_name",
    "bg_admin3_code",
]
PUBLIC_NUMERIC_PREFIXES = ("district_nsi_", "municipality_nsi_")
PUBLIC_NUMERIC_FEATURES = [
    "nsi_district_found",
    "nsi_municipality_found",
]
PUBLIC_CATEGORICAL_FEATURES = [
    "nsi_district_norm",
    "nsi_municipality_norm",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: Any
    preprocessor: str = "ordinal"
    sampler: str | None = None
    family: str = "other"


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).fillna(0.0)


def load_data(path: Path, feature_set: str) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    for col in [
        "Total_SUBs",
        "AvgMobileRevenue",
        "AvgFIXRevenue",
        "TotalRevenue",
        "ARPU",
        "Active_subscribers",
        "Not_Active_subscribers",
        "Suspended_subscribers",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Suspended_subscribers" not in df.columns:
        df["Suspended_subscribers"] = 0.0

    mask = df["ARPU"].isna() & df["Total_SUBs"].gt(0)
    df.loc[mask, "ARPU"] = df.loc[mask, "TotalRevenue"] / df.loc[mask, "Total_SUBs"]

    df["Active_Ratio"] = _safe_divide(df["Active_subscribers"], df["Total_SUBs"]).clip(0.0, 1.0)
    df["Not_Active_subscribers"] = df["Not_Active_subscribers"].fillna(0.0)
    df["Mobile_Revenue_Ratio"] = _safe_divide(df["AvgMobileRevenue"], df["TotalRevenue"]).clip(0.0, 1.0)
    df["Inactive_Ratio"] = _safe_divide(df["Not_Active_subscribers"], df["Total_SUBs"]).clip(0.0, 1.0)
    df["Suspended_Ratio"] = _safe_divide(df["Suspended_subscribers"], df["Total_SUBs"]).clip(0.0, 1.0)
    df["Revenue_per_Active_Sub"] = _safe_divide(df["TotalRevenue"], df["Active_subscribers"])
    df["Inactive_x_Revenue"] = df["Inactive_Ratio"] * df["TotalRevenue"].fillna(0.0)
    revenue_pair = df[["AvgMobileRevenue", "AvgFIXRevenue"]].fillna(0.0)
    df["Revenue_Balance"] = (
        revenue_pair.min(axis=1) / (revenue_pair.max(axis=1) + 1e-5)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    numeric_features = list(CORE_NUMERIC_FEATURES)
    categorical_features = list(CORE_CATEGORICAL_FEATURES)
    if feature_set == "full":
        numeric_features += FULL_EXTRA_NUMERIC_FEATURES
        categorical_features += FULL_EXTRA_CATEGORICAL_FEATURES
        numeric_features += [col for col in GEO_NUMERIC_FEATURES if col in df.columns]
        categorical_features += [col for col in GEO_CATEGORICAL_FEATURES if col in df.columns]
        numeric_features += [col for col in PUBLIC_NUMERIC_FEATURES if col in df.columns]
        numeric_features += [
            col
            for col in df.columns
            if col.startswith(PUBLIC_NUMERIC_PREFIXES) and pd.api.types.is_numeric_dtype(df[col])
        ]
        categorical_features += [col for col in PUBLIC_CATEGORICAL_FEATURES if col in df.columns]

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_features:
        df[col] = df[col].fillna("Unknown").astype(str)

    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    features = numeric_features + categorical_features
    return (
        df.loc[valid, features].reset_index(drop=True),
        y[valid].astype(int).reset_index(drop=True),
        numeric_features,
        categorical_features,
    )


def make_preprocessor(kind: str, numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    if kind == "onehot":
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=5,
                    sparse_output=False,
                ),
            ),
        ])
    elif kind == "ordinal":
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
    else:
        raise ValueError(f"Unknown preprocessor: {kind}")

    return ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ])


def make_sampler(name: str | None, cat_indices: list[int]):
    if name is None:
        return None
    if name == "ros":
        return RandomOverSampler(random_state=RANDOM_STATE)
    if name == "rus":
        return RandomUnderSampler(random_state=RANDOM_STATE)
    if name == "smote":
        return SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    if name == "borderline_smote":
        return BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    if name == "adasyn":
        return ADASYN(random_state=RANDOM_STATE, n_neighbors=5)
    if name == "smoteenn":
        return SMOTEENN(random_state=RANDOM_STATE)
    if name == "smotetomek":
        return SMOTETomek(random_state=RANDOM_STATE)
    if name == "smotenc":
        if not cat_indices:
            return SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        return SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE, k_neighbors=5)
    raise ValueError(f"Unknown sampler: {name}")


def make_pipeline(candidate: Candidate, numeric_features: list[str], categorical_features: list[str]):
    preprocessor = make_preprocessor(candidate.preprocessor, numeric_features, categorical_features)
    cat_indices = list(range(len(numeric_features), len(numeric_features) + len(categorical_features)))
    sampler = make_sampler(candidate.sampler, cat_indices)

    steps = [("prep", preprocessor)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", candidate.estimator))
    pipeline_cls = ImbPipeline if sampler is not None else Pipeline
    return pipeline_cls(steps)


def build_candidates(scale_pos_weight: float) -> list[Candidate]:
    xgb_common = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    lgbm_common = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    cat_common = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "eval_metric": "F1",
        "auto_class_weights": "Balanced",
        "random_seed": RANDOM_STATE,
        "verbose": 0,
        "allow_writing_files": False,
    }

    return [
        Candidate("Dummy_prior", DummyClassifier(strategy="prior"), "onehot", family="baseline"),
        Candidate("Dummy_stratified", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE), "onehot", family="baseline"),
        Candidate("LogReg_L2_balanced", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE), "onehot", family="linear"),
        Candidate("LogReg_L1_balanced", LogisticRegression(max_iter=3000, penalty="l1", solver="saga", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), "onehot", family="linear"),
        Candidate("LogReg_ROS", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE), "onehot", "ros", "linear_resampled"),
        Candidate("LogReg_SMOTE", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE), "onehot", "smote", "linear_resampled"),
        Candidate("SGD_log_balanced", SGDClassifier(loss="log_loss", max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), "onehot", family="linear"),
        Candidate("Ridge_balanced", RidgeClassifier(class_weight="balanced", random_state=RANDOM_STATE), "onehot", family="linear"),
        Candidate("PassiveAggressive_balanced", PassiveAggressiveClassifier(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE), "onehot", family="linear"),
        Candidate("LinearSVC_calibrated", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, dual="auto", max_iter=5000), cv=3), "onehot", family="svm"),
        Candidate("SVC_rbf_balanced", SVC(class_weight="balanced", kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE), "onehot", family="svm"),
        Candidate("KNN_5", KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1), "onehot", family="neighbors"),
        Candidate("KNN_25", KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1), "onehot", family="neighbors"),
        Candidate("GaussianNB", GaussianNB(), "onehot", family="bayes"),
        Candidate("BernoulliNB", BernoulliNB(), "onehot", family="bayes"),
        Candidate("LDA_shrinkage", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "onehot", family="discriminant"),
        Candidate("QDA", QuadraticDiscriminantAnalysis(reg_param=0.2), "onehot", family="discriminant"),
        Candidate("MLP_64_32", MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300, random_state=RANDOM_STATE), "onehot", family="neural_net"),
        Candidate("DecisionTree_balanced", DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=10, random_state=RANDOM_STATE), "ordinal", family="tree"),
        Candidate("RandomForest_balanced_d6", RandomForestClassifier(n_estimators=400, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), "ordinal", family="forest"),
        Candidate("RandomForest_balanced_deep", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1), "ordinal", family="forest"),
        Candidate("ExtraTrees_balanced", ExtraTreesClassifier(n_estimators=500, class_weight="balanced", min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1), "ordinal", family="forest"),
        Candidate("GradientBoosting", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE), "ordinal", family="boosting"),
        Candidate("HistGradientBoosting_balanced", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=RANDOM_STATE), "ordinal", family="boosting"),
        Candidate("AdaBoost_tree", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE), n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE), "ordinal", family="boosting"),
        Candidate("XGBoost_d3_weighted", XGBClassifier(max_depth=3, **xgb_common), "ordinal", family="boosting"),
        Candidate("XGBoost_d6_weighted", XGBClassifier(max_depth=6, **xgb_common), "ordinal", family="boosting"),
        Candidate("LightGBM_balanced", LGBMClassifier(**lgbm_common), "ordinal", family="boosting"),
        Candidate("LightGBM_leaf15", LGBMClassifier(num_leaves=15, **lgbm_common), "ordinal", family="boosting"),
        Candidate("CatBoost_balanced", CatBoostClassifier(**cat_common), "ordinal", family="boosting"),
        Candidate("RF_SMOTE", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1), "ordinal", "smotenc", "forest_resampled"),
        Candidate("RF_BorderlineSMOTE", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1), "ordinal", "borderline_smote", "forest_resampled"),
        Candidate("RF_SMOTEENN", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1), "ordinal", "smoteenn", "forest_resampled"),
        Candidate("XGBoost_SMOTENC", XGBClassifier(max_depth=3, **{**xgb_common, "scale_pos_weight": 1.0}), "ordinal", "smotenc", "boosting_resampled"),
        Candidate("XGBoost_RUS", XGBClassifier(max_depth=3, **{**xgb_common, "scale_pos_weight": 1.0}), "ordinal", "rus", "boosting_resampled"),
        Candidate("LightGBM_SMOTENC", LGBMClassifier(**{**lgbm_common, "class_weight": None}), "ordinal", "smotenc", "boosting_resampled"),
        Candidate("BalancedRF_d6", BalancedRandomForestClassifier(n_estimators=400, max_depth=6, sampling_strategy="all", replacement=True, bootstrap=False, random_state=RANDOM_STATE, n_jobs=-1), "ordinal", family="imbalance_ensemble"),
        Candidate("BalancedRF_deep", BalancedRandomForestClassifier(n_estimators=400, sampling_strategy="all", replacement=True, bootstrap=False, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1), "ordinal", family="imbalance_ensemble"),
        Candidate("EasyEnsemble_10", EasyEnsembleClassifier(n_estimators=10, random_state=RANDOM_STATE, n_jobs=-1), "ordinal", family="imbalance_ensemble"),
        Candidate("EasyEnsemble_30", EasyEnsembleClassifier(n_estimators=30, random_state=RANDOM_STATE, n_jobs=-1), "ordinal", family="imbalance_ensemble"),
        Candidate("BalancedBagging_DT", BalancedBaggingClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5, random_state=RANDOM_STATE), n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), "ordinal", family="imbalance_ensemble"),
        Candidate("RUSBoost", RUSBoostClassifier(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE), "ordinal", family="imbalance_ensemble"),
        Candidate("Voting_LR_RF_XGB_LGBM", VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
                ("rf", RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
                ("xgb", XGBClassifier(max_depth=3, **xgb_common)),
                ("lgbm", LGBMClassifier(**lgbm_common)),
            ],
            voting="soft",
            n_jobs=-1,
        ), "ordinal", family="ensemble"),
        Candidate("Stacking_LR_RF_XGB", StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
                ("rf", RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
                ("xgb", XGBClassifier(max_depth=3, **xgb_common)),
            ],
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            stack_method="auto",
            cv=3,
            n_jobs=-1,
        ), "ordinal", family="ensemble"),
    ]


def score_model(model, X: pd.DataFrame) -> tuple[np.ndarray, str]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return np.asarray(proba[:, 1], dtype=float), "proba"
        return np.asarray(proba).ravel().astype(float), "proba"
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel().astype(float), "decision"
    return np.asarray(model.predict(X)).ravel().astype(float), "label"


def best_f1_threshold(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(scores)) <= 1:
        threshold = float(scores[0]) if len(scores) else 0.5
        return threshold, f1_score(y_true, scores >= threshold, zero_division=0)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_idx = int(np.nanargmax(f1_values))
    return float(thresholds[best_idx]), float(f1_values[best_idx])


def metrics_from_prediction(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pred_pos_rate": float(np.mean(y_pred)),
    }


def safe_auc(y_true: pd.Series | np.ndarray, scores: np.ndarray, metric: str) -> float:
    try:
        if metric == "roc_auc":
            return float(roc_auc_score(y_true, scores))
        if metric == "average_precision":
            return float(average_precision_score(y_true, scores))
    except ValueError:
        return float("nan")
    raise ValueError(metric)


def evaluate_candidate(
    candidate: Candidate,
    pipeline,
    feature_set: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, Any], Any | None]:
    started = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            pipeline.fit(X_train, y_train)

        val_scores, score_type = score_model(pipeline, X_val)
        test_scores, _ = score_model(pipeline, X_test)

        val_threshold, val_best_f1 = best_f1_threshold(y_val, val_scores)
        test_oracle_threshold, test_oracle_f1 = best_f1_threshold(y_test, test_scores)

        y_default = pipeline.predict(X_test)
        default_metrics = metrics_from_prediction(y_test, y_default)

        y_tuned = (test_scores >= val_threshold).astype(int)
        tuned_metrics = metrics_from_prediction(y_test, y_tuned)

        elapsed = time.perf_counter() - started
        row: dict[str, Any] = {
            "feature_set": feature_set,
            "model": candidate.name,
            "family": candidate.family,
            "preprocessor": candidate.preprocessor,
            "sampler": candidate.sampler or "none",
            "score_type": score_type,
            "train_seconds": round(elapsed, 3),
            "status": "ok",
            "error": "",
            "val_best_threshold": val_threshold,
            "val_best_f1": val_best_f1,
            "test_oracle_threshold": test_oracle_threshold,
            "test_oracle_f1": test_oracle_f1,
            "test_roc_auc": safe_auc(y_test, test_scores, "roc_auc"),
            "test_average_precision": safe_auc(y_test, test_scores, "average_precision"),
        }
        row.update({f"test_default_{k}": v for k, v in default_metrics.items()})
        row.update({f"test_tuned_{k}": v for k, v in tuned_metrics.items()})
        return row, pipeline
    except Exception as exc:  # Keep the benchmark moving.
        elapsed = time.perf_counter() - started
        return {
            "feature_set": feature_set,
            "model": candidate.name,
            "family": candidate.family,
            "preprocessor": candidate.preprocessor,
            "sampler": candidate.sampler or "none",
            "score_type": "",
            "train_seconds": round(elapsed, 3),
            "status": "failed",
            "error": repr(exc),
        }, None


def run_feature_set(
    csv_path: Path,
    feature_set: str,
    candidates: list[Candidate],
    target_f1: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], Any], dict[str, Any]]:
    X, y, numeric_features, categorical_features = load_data(csv_path, feature_set)
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
    )

    print(
        f"\n=== Feature set: {feature_set} | rows={len(X)} | features={X.shape[1]} | "
        f"churn_rate={y.mean():.2%} | scale_pos_weight={scale_pos_weight:.2f} ===",
        flush=True,
    )
    print(f"Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}", flush=True)

    fitted: dict[tuple[str, str], Any] = {}
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        pipeline = make_pipeline(candidate, numeric_features, categorical_features)
        print(f"[{feature_set}] {idx:02d}/{len(candidates)} {candidate.name}", flush=True)
        row, model = evaluate_candidate(candidate, pipeline, feature_set, X_train, X_val, X_test, y_train, y_val, y_test)
        rows.append(row)
        if model is not None:
            fitted[(feature_set, candidate.name)] = model

        if row.get("status") == "ok":
            print(
                f"    tuned_f1={row['test_tuned_f1']:.4f} default_f1={row['test_default_f1']:.4f} "
                f"oracle_f1={row['test_oracle_f1']:.4f} auc={row['test_roc_auc']:.4f}",
                flush=True,
            )
            if row["test_tuned_f1"] >= target_f1:
                print(f"    target reached: F1 >= {target_f1:.2f}", flush=True)
        else:
            print(f"    failed: {row['error']}", flush=True)

    metadata = {
        "feature_set": feature_set,
        "rows": int(len(X)),
        "features": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "churn_rate": float(y.mean()),
        "class_counts": {"no_churn": neg, "churn": pos},
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
    }
    return rows, fitted, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=raw_data_path("baza_telecom_v2.csv"))
    parser.add_argument("--feature-set", choices=["core", "full", "both"], default="both")
    parser.add_argument("--target-f1", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=0, help="Debug only: run the first N candidates.")
    parser.add_argument("--out", type=Path, default=model_path("best_benchmark_model.joblib"))
    parser.add_argument("--table-out", type=Path, default=result_path("model_benchmark_table.csv"))
    parser.add_argument("--json-out", type=Path, default=result_path("model_benchmark_summary.json"))
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    args.csv = resolve_input_path(args.csv, raw_data_path("baza_telecom_v2.csv"))
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    # Build with the actual class ratio from the core feature set.
    _, y_core, _, _ = load_data(args.csv, "core")
    scale_pos_weight = int((y_core == 0).sum()) / max(int((y_core == 1).sum()), 1)
    candidates = build_candidates(scale_pos_weight)
    if args.limit > 0:
        candidates = candidates[: args.limit]

    feature_sets = ["core", "full"] if args.feature_set == "both" else [args.feature_set]
    all_rows: list[dict[str, Any]] = []
    all_fitted: dict[tuple[str, str], Any] = {}
    all_metadata: dict[str, Any] = {
        "csv": str(args.csv),
        "target": TARGET,
        "target_f1": args.target_f1,
        "candidate_count_per_feature_set": len(candidates),
        "feature_sets": {},
    }

    started = time.perf_counter()
    for feature_set in feature_sets:
        rows, fitted, metadata = run_feature_set(args.csv, feature_set, candidates, args.target_f1)
        all_rows.extend(rows)
        all_fitted.update(fitted)
        all_metadata["feature_sets"][feature_set] = metadata

    table = pd.DataFrame(all_rows)
    ok_table = table[table["status"] == "ok"].copy()
    sort_cols = ["test_tuned_f1", "test_roc_auc", "test_average_precision"]
    ok_table = ok_table.sort_values(sort_cols, ascending=[False, False, False])
    failed_table = table[table["status"] != "ok"].copy()
    final_table = pd.concat([ok_table, failed_table], ignore_index=True)

    args.table_out = args.table_out if args.table_out.is_absolute() else REPO_ROOT / args.table_out
    args.json_out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    args.out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    final_table.to_csv(args.table_out, index=False, encoding="utf-8-sig")

    best_row = ok_table.iloc[0].to_dict() if not ok_table.empty else {}
    best_key = (best_row.get("feature_set"), best_row.get("model"))
    if best_key in all_fitted:
        joblib.dump(all_fitted[best_key], args.out)

    reached = ok_table[ok_table["test_tuned_f1"] >= args.target_f1]
    all_metadata.update({
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "table_path": str(args.table_out),
        "best_model_path": str(args.out) if best_key in all_fitted else "",
        "best": best_row,
        "target_reached": bool(len(reached) > 0),
        "target_reached_count": int(len(reached)),
        "failed_count": int(len(failed_table)),
        "top_models": ok_table.head(args.top).to_dict(orient="records"),
    })
    args.json_out.write_text(json.dumps(all_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Top Models by validation-threshold test F1 ===", flush=True)
    display_cols = [
        "feature_set",
        "model",
        "family",
        "sampler",
        "test_tuned_f1",
        "test_tuned_precision",
        "test_tuned_recall",
        "test_default_f1",
        "test_oracle_f1",
        "test_roc_auc",
        "test_average_precision",
    ]
    print(ok_table[display_cols].head(args.top).to_string(index=False), flush=True)
    print(f"\nSaved table: {args.table_out}", flush=True)
    print(f"Saved summary: {args.json_out}", flush=True)
    if best_key in all_fitted:
        print(f"Saved best benchmark model: {args.out}", flush=True)
    print(
        f"Target F1 {args.target_f1:.2f}: "
        f"{'REACHED' if all_metadata['target_reached'] else 'not reached'} "
        f"({all_metadata['target_reached_count']} models)",
        flush=True,
    )


if __name__ == "__main__":
    main()
