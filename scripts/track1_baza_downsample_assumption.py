#!/usr/bin/env python3
"""Baza churn downsampling assumption experiment.

This script separates two claims that are easy to mix up in a report:

1. Real Baza distribution: churn is only about 6.5%, so F1 remains low.
2. Balanced-data assumption: if we keep churn rows and randomly sample the
   same number of non-churn rows, the evaluation distribution becomes 50/50
   and F1 can rise substantially.

The balanced result is not a deployment estimate. It is an explicit
what-if experiment for a balanced population assumption.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.helpers import raw_data_path, resolve_input_path, result_path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TARGET = "CHURN"


@dataclass(frozen=True)
class SplitBundle:
    experiment: str
    assumption_note: str
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    sample_rows: int
    sample_churn_rate: float


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_baza_features(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)

    numeric_base = [
        "Active_subscribers",
        "Not_Active_subscribers",
        "Suspended_subscribers",
        "Total_SUBs",
        "AvgMobileRevenue",
        "AvgFIXRevenue",
        "TotalRevenue",
        "ARPU",
    ]
    for col in numeric_base:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Not_Active_subscribers"] = df["Not_Active_subscribers"].fillna(0.0)
    df["Suspended_subscribers"] = df["Suspended_subscribers"].fillna(0.0)
    arpu_from_total = safe_divide(df["TotalRevenue"], df["Total_SUBs"])
    df["ARPU"] = df["ARPU"].fillna(arpu_from_total)

    df["Active_Ratio"] = safe_divide(df["Active_subscribers"], df["Total_SUBs"]).clip(0.0, 1.0)
    df["Inactive_Ratio"] = safe_divide(df["Not_Active_subscribers"], df["Total_SUBs"]).clip(0.0, 1.0)
    df["Suspended_Ratio"] = safe_divide(df["Suspended_subscribers"], df["Total_SUBs"]).clip(0.0, 1.0)
    df["Dormant_Ratio"] = safe_divide(
        df["Not_Active_subscribers"] + df["Suspended_subscribers"], df["Total_SUBs"]
    ).clip(0.0, 1.0)
    df["Mobile_Revenue_Ratio"] = safe_divide(df["AvgMobileRevenue"], df["TotalRevenue"]).clip(0.0, 1.0)
    df["Fixed_Revenue_Ratio"] = safe_divide(df["AvgFIXRevenue"], df["TotalRevenue"]).clip(0.0, 1.0)
    df["Revenue_Per_Sub"] = safe_divide(df["TotalRevenue"], df["Total_SUBs"])
    df["Revenue_Per_Active"] = safe_divide(df["TotalRevenue"], df["Active_subscribers"])
    df["Mobile_To_Fixed_Ratio"] = safe_divide(df["AvgMobileRevenue"], df["AvgFIXRevenue"])
    df["Fixed_To_Mobile_Ratio"] = safe_divide(df["AvgFIXRevenue"], df["AvgMobileRevenue"])
    df["Has_Inactive"] = df["Not_Active_subscribers"].gt(0).astype(float)
    df["Has_Suspended"] = df["Suspended_subscribers"].gt(0).astype(float)
    df["Mobile_Only"] = (df["AvgMobileRevenue"].gt(0) & df["AvgFIXRevenue"].eq(0)).astype(float)
    df["Fixed_Only"] = (df["AvgFIXRevenue"].gt(0) & df["AvgMobileRevenue"].eq(0)).astype(float)

    numeric_features = numeric_base + [
        "Active_Ratio",
        "Inactive_Ratio",
        "Suspended_Ratio",
        "Dormant_Ratio",
        "Mobile_Revenue_Ratio",
        "Fixed_Revenue_Ratio",
        "Revenue_Per_Sub",
        "Revenue_Per_Active",
        "Mobile_To_Fixed_Ratio",
        "Fixed_To_Mobile_Ratio",
        "Has_Inactive",
        "Has_Suspended",
        "Mobile_Only",
        "Fixed_Only",
    ]
    categorical_features = ["CRM_PID_Value_Segment", "EffectiveSegment", "Billing_ZIP", "KA_name"]

    for col in categorical_features:
        df[col] = df[col].fillna("Unknown").astype(str)

    X = df[numeric_features + categorical_features].copy()
    return X, y, numeric_features, categorical_features


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False),
        ),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])


def build_models() -> dict[str, Any]:
    return {
        "LogReg_balanced": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        "DecisionTree_balanced": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest_balanced": RandomForestClassifier(
            n_estimators=400,
            max_depth=7,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "ExtraTrees_balanced": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "HistGB_balanced": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=15,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(model: Any, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(numeric_features, categorical_features)),
        ("model", model),
    ])


def balanced_sample(X: pd.DataFrame, y: pd.Series, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].sample(n=len(pos_idx), random_state=random_state).index
    sample_idx = pd.Index(pos_idx.tolist() + neg_idx.tolist())
    sample_idx = sample_idx.to_series().sample(frac=1.0, random_state=random_state).index
    return X.loc[sample_idx].reset_index(drop=True), y.loc[sample_idx].reset_index(drop=True)


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    experiment: str,
    assumption_note: str,
) -> SplitBundle:
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
    )
    return SplitBundle(
        experiment=experiment,
        assumption_note=assumption_note,
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        sample_rows=int(len(X)),
        sample_churn_rate=float(y.mean()),
    )


def build_experiments(X: pd.DataFrame, y: pd.Series) -> list[SplitBundle]:
    original = split_train_val_test(
        X,
        y,
        "original_distribution",
        "Real Baza distribution. Churn remains about 6.5%.",
    )

    X_balanced, y_balanced = balanced_sample(X, y, RANDOM_STATE)
    balanced = split_train_val_test(
        X_balanced,
        y_balanced,
        "balanced_population_assumption",
        "What-if experiment: all churn rows plus the same number of random non-churn rows. Test set is also balanced.",
    )

    X_train_down, y_train_down = balanced_sample(original.X_train, original.y_train, RANDOM_STATE)
    train_downsampled = SplitBundle(
        experiment="downsampled_train_original_holdout",
        assumption_note="Only the training set is downsampled. Validation and test keep the real imbalanced distribution.",
        X_train=X_train_down,
        X_val=original.X_val,
        X_test=original.X_test,
        y_train=y_train_down,
        y_val=original.y_val,
        y_test=original.y_test,
        sample_rows=int(len(X_train_down) + len(original.X_val) + len(original.X_test)),
        sample_churn_rate=float(pd.concat([y_train_down, original.y_val, original.y_test]).mean()),
    )
    return [original, balanced, train_downsampled]


def best_f1_threshold(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1_values))
    return float(thresholds[idx]), float(f1_values[idx])


def metrics_from_prediction(y_true: pd.Series | np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pred_pos_rate": float(np.mean(pred)),
    }


def model_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    return np.asarray(proba[:, 1] if proba.ndim == 2 else proba).ravel().astype(float)


def evaluate_experiment(
    bundle: SplitBundle,
    model_name: str,
    pipeline: Pipeline,
) -> dict[str, Any]:
    started = time.perf_counter()
    pipeline.fit(bundle.X_train, bundle.y_train)
    val_scores = model_scores(pipeline, bundle.X_val)
    test_scores = model_scores(pipeline, bundle.X_test)

    threshold, val_best_f1 = best_f1_threshold(bundle.y_val, val_scores)
    oracle_threshold, oracle_f1 = best_f1_threshold(bundle.y_test, test_scores)
    tuned_pred = (test_scores >= threshold).astype(int)
    default_pred = pipeline.predict(bundle.X_test)

    row: dict[str, Any] = {
        "experiment": bundle.experiment,
        "assumption_note": bundle.assumption_note,
        "model": model_name,
        "sample_rows": bundle.sample_rows,
        "sample_churn_rate": bundle.sample_churn_rate,
        "train_rows": int(len(bundle.X_train)),
        "train_churn_rate": float(bundle.y_train.mean()),
        "val_rows": int(len(bundle.X_val)),
        "val_churn_rate": float(bundle.y_val.mean()),
        "test_rows": int(len(bundle.X_test)),
        "test_churn_rate": float(bundle.y_test.mean()),
        "threshold": threshold,
        "val_best_f1": val_best_f1,
        "test_oracle_threshold": oracle_threshold,
        "test_oracle_f1": oracle_f1,
        "test_roc_auc": float(roc_auc_score(bundle.y_test, test_scores)),
        "test_average_precision": float(average_precision_score(bundle.y_test, test_scores)),
        "train_seconds": round(time.perf_counter() - started, 3),
    }
    row.update({f"test_tuned_{key}": value for key, value in metrics_from_prediction(bundle.y_test, tuned_pred).items()})
    row.update({f"test_default_{key}": value for key, value in metrics_from_prediction(bundle.y_test, default_pred).items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=raw_data_path("baza_telecom_v2.csv"))
    parser.add_argument("--table-out", type=Path, default=result_path("baza_downsample_assumption_table.csv"))
    parser.add_argument("--json-out", type=Path, default=result_path("baza_downsample_assumption_summary.json"))
    args = parser.parse_args()

    args.csv = resolve_input_path(args.csv, raw_data_path("baza_telecom_v2.csv"))
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    X, y, numeric_features, categorical_features = load_baza_features(args.csv)
    models = build_models()
    experiments = build_experiments(X, y)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    print(
        f"Baza rows={len(X)} churn_rate={y.mean():.2%} "
        f"churn={int((y == 1).sum())} non_churn={int((y == 0).sum())}",
        flush=True,
    )
    for bundle in experiments:
        print(
            f"\n=== {bundle.experiment} | train_churn={bundle.y_train.mean():.2%} "
            f"test_churn={bundle.y_test.mean():.2%} ===",
            flush=True,
        )
        for model_name, model in models.items():
            pipeline = make_pipeline(model, numeric_features, categorical_features)
            row = evaluate_experiment(bundle, model_name, pipeline)
            rows.append(row)
            print(
                f"{model_name}: tuned_f1={row['test_tuned_f1']:.4f} "
                f"precision={row['test_tuned_precision']:.4f} recall={row['test_tuned_recall']:.4f} "
                f"auc={row['test_roc_auc']:.4f}",
                flush=True,
            )

    table = pd.DataFrame(rows).sort_values(
        ["experiment", "test_tuned_f1", "test_roc_auc"], ascending=[True, False, False]
    )
    args.table_out = args.table_out if args.table_out.is_absolute() else REPO_ROOT / args.table_out
    args.json_out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table_out, index=False, encoding="utf-8-sig")

    best_by_experiment = table.groupby("experiment", sort=False).head(1).to_dict(orient="records")
    summary = {
        "csv": str(args.csv),
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "churn_rate": float(y.mean()),
        "class_counts": {"no_churn": int((y == 0).sum()), "churn": int((y == 1).sum())},
        "balanced_sample_rows": int(2 * (y == 1).sum()),
        "balanced_sample_fraction_of_original": float((2 * (y == 1).sum()) / len(y)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "table_path": str(args.table_out),
        "best_by_experiment": best_by_experiment,
        "report_caveat": (
            "The balanced_population_assumption rows are valid only under an artificial "
            "50/50 churn/non-churn evaluation assumption. They should not be reported as "
            "real Baza deployment performance."
        ),
    }
    args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    display_cols = [
        "experiment",
        "model",
        "test_churn_rate",
        "test_tuned_f1",
        "test_tuned_precision",
        "test_tuned_recall",
        "test_roc_auc",
    ]
    print("\n=== Best by experiment ===", flush=True)
    print(pd.DataFrame(best_by_experiment)[display_cols].to_string(index=False), flush=True)
    print(f"\nSaved table: {args.table_out}", flush=True)
    print(f"Saved summary: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
