#!/usr/bin/env python3
"""
Benchmark processed churn datasets and save the best F1 model.

The Baza CRM table has weak churn signal with the current static features.
This script checks whether feature-richer processed datasets can meet the
project F1 target, and stores the best threshold-tuned classifier separately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from lightgbm import LGBMClassifier
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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.models.threshold_model import ThresholdClassifier
from src.utils.helpers import model_path, processed_data_path, result_path

RANDOM_STATE = 42
TARGET = "CHURN"

DATASETS = {
    "baza": processed_data_path("baza_telecom_v2_processed.csv"),
    "ibm": processed_data_path("ibm_telco_churn_processed.csv"),
    "cell2cell": processed_data_path("cell2cell_train_processed.csv"),
    "iranian": processed_data_path("iranian_churn_processed.csv"),
}


def build_models(scale_pos_weight: float) -> dict[str, Any]:
    return {
        "LR_balanced": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        "RF_balanced": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "ExtraTrees_balanced": ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced",
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGB_balanced": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "XGBoost_weighted": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "LightGBM_balanced": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced",
            verbose=-1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "BalancedRF": BalancedRandomForestClassifier(
            n_estimators=300,
            sampling_strategy="all",
            replacement=True,
            bootstrap=False,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "EasyEnsemble": EasyEnsembleClassifier(
            n_estimators=20,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def best_f1_threshold(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1_values))
    return float(thresholds[idx]), float(f1_values[idx])


def load_processed_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} column not found: {path}")
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def make_pipeline(model: Any) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def evaluate_model(dataset: str, model_name: str, pipeline: Pipeline, X, y) -> tuple[dict[str, Any], ThresholdClassifier]:
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
    )

    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    val_scores = pipeline.predict_proba(X_val)[:, 1]
    test_scores = pipeline.predict_proba(X_test)[:, 1]
    threshold, val_best_f1 = best_f1_threshold(y_val, val_scores)
    tuned = ThresholdClassifier(pipeline, threshold)

    tuned_pred = tuned.predict(X_test)
    default_pred = pipeline.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, tuned_pred, labels=[0, 1]).ravel()

    row = {
        "dataset": dataset,
        "model": model_name,
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "churn_rate": float(y.mean()),
        "threshold": threshold,
        "val_best_f1": val_best_f1,
        "test_tuned_f1": float(f1_score(y_test, tuned_pred, zero_division=0)),
        "test_default_f1": float(f1_score(y_test, default_pred, zero_division=0)),
        "test_precision": float(precision_score(y_test, tuned_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, tuned_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "test_average_precision": float(average_precision_score(y_test, test_scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "train_seconds": round(time.perf_counter() - started, 3),
    }
    return row, tuned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-f1", type=float, default=0.6)
    parser.add_argument("--table-out", type=Path, default=result_path("processed_dataset_benchmark.csv"))
    parser.add_argument("--json-out", type=Path, default=result_path("processed_dataset_benchmark_summary.json"))
    parser.add_argument("--model-out", type=Path, default=model_path("best_processed_f1_model.joblib"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str], ThresholdClassifier] = {}
    for dataset, path in DATASETS.items():
        if not path.exists():
            print(f"[skip] {dataset}: missing {path}", flush=True)
            continue
        X, y = load_processed_dataset(path)
        scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
        print(
            f"\n=== {dataset}: rows={len(X)} features={X.shape[1]} churn_rate={y.mean():.2%} ===",
            flush=True,
        )
        for model_name, model in build_models(scale_pos_weight).items():
            print(f"[{dataset}] {model_name}", flush=True)
            row, tuned = evaluate_model(dataset, model_name, make_pipeline(model), X, y)
            rows.append(row)
            fitted[(dataset, model_name)] = tuned
            print(
                f"    tuned_f1={row['test_tuned_f1']:.4f} "
                f"default_f1={row['test_default_f1']:.4f} "
                f"precision={row['test_precision']:.4f} recall={row['test_recall']:.4f}",
                flush=True,
            )

    table = pd.DataFrame(rows).sort_values(["test_tuned_f1", "test_roc_auc"], ascending=[False, False])
    args.table_out = args.table_out if args.table_out.is_absolute() else REPO_ROOT / args.table_out
    args.json_out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    args.model_out = args.model_out if args.model_out.is_absolute() else REPO_ROOT / args.model_out
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(args.table_out, index=False, encoding="utf-8-sig")
    best = table.iloc[0].to_dict()
    best_model = fitted[(best["dataset"], best["model"])]
    joblib.dump(best_model, args.model_out)

    reached = table[table["test_tuned_f1"] >= args.target_f1]
    summary = {
        "target_f1": args.target_f1,
        "target_reached": bool(len(reached) > 0),
        "target_reached_count": int(len(reached)),
        "best_model_path": str(args.model_out),
        "table_path": str(args.table_out),
        "best": best,
        "best_by_dataset": table.groupby("dataset", sort=False).head(1).to_dict(orient="records"),
    }
    args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Top 12 ===", flush=True)
    cols = [
        "dataset",
        "model",
        "test_tuned_f1",
        "test_default_f1",
        "test_precision",
        "test_recall",
        "test_roc_auc",
        "threshold",
    ]
    print(table[cols].head(12).to_string(index=False), flush=True)
    print(f"\nSaved table: {args.table_out}", flush=True)
    print(f"Saved summary: {args.json_out}", flush=True)
    print(f"Saved best model: {args.model_out}", flush=True)
    print(
        f"Target F1 {args.target_f1:.2f}: "
        f"{'REACHED' if summary['target_reached'] else 'not reached'} "
        f"({summary['target_reached_count']} models)",
        flush=True,
    )


if __name__ == "__main__":
    main()
