#!/usr/bin/env python3
"""Train churn models on the rebuilt KKBox customer-month dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.helpers import model_path, processed_data_path, result_path

RANDOM_STATE = 42
TARGET = "churn_next_30d"
DROP_COLUMNS = {"msno", "snapshot_month", TARGET}


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: Any


def best_f1_threshold(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1_values))
    return float(thresholds[idx]), float(f1_values[idx])


def metrics(y_true: pd.Series | np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pred_pos_rate": float(np.mean(pred)),
    }


def load_dataset(path: Path, max_rows: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if max_rows and len(df) > max_rows:
        df = (
            df.groupby(TARGET, group_keys=False)
            .sample(frac=max_rows / len(df), random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
    y = df[TARGET].astype(int)
    feature_cols = [col for col in df.columns if col not in DROP_COLUMNS]
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y


def build_candidates(scale_pos_weight: float) -> list[Candidate]:
    return [
        Candidate(
            "LightGBM_balanced",
            LGBMClassifier(
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=40,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        Candidate(
            "LightGBM_regularized",
            LGBMClassifier(
                n_estimators=900,
                learning_rate=0.025,
                num_leaves=15,
                min_child_samples=80,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=4.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        Candidate(
            "XGBoost_weighted",
            XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.04,
                min_child_weight=5,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=4.0,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=processed_data_path("kkbox_customer_month_churn.csv"))
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=result_path("kkbox_customer_month_training_summary.json"))
    parser.add_argument("--model-out", type=Path, default=model_path("kkbox_customer_month_best_model.joblib"))
    args = parser.parse_args()

    args.csv = args.csv if args.csv.is_absolute() else REPO_ROOT / args.csv
    X, y = load_dataset(args.csv, args.max_rows)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
    )

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for candidate in build_candidates(scale_pos_weight):
        started = time.perf_counter()
        print(f"[train] {candidate.name}", flush=True)
        candidate.estimator.fit(X_train, y_train)
        val_scores = candidate.estimator.predict_proba(X_val)[:, 1]
        test_scores = candidate.estimator.predict_proba(X_test)[:, 1]
        threshold, val_f1 = best_f1_threshold(y_val, val_scores)
        row = {
            "model": candidate.name,
            "train_seconds": round(time.perf_counter() - started, 3),
            "val_best_f1": float(val_f1),
        }
        row.update(metrics(y_test, test_scores, threshold))
        row["default_f1"] = float(f1_score(y_test, test_scores >= 0.5, zero_division=0))
        rows.append(row)
        fitted[candidate.name] = candidate.estimator
        print(
            f"    f1={row['f1']:.4f} precision={row['precision']:.4f} "
            f"recall={row['recall']:.4f} auc={row['roc_auc']:.4f}",
            flush=True,
        )

    table = pd.DataFrame(rows).sort_values(["f1", "roc_auc"], ascending=[False, False])
    best = table.iloc[0].to_dict()
    args.json_out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    args.model_out = args.model_out if args.model_out.is_absolute() else REPO_ROOT / args.model_out
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted[best["model"]], "threshold": best["threshold"], "features": list(X.columns)}, args.model_out)

    summary = {
        "csv": str(args.csv),
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "churn_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "best": best,
        "models": table.to_dict(orient="records"),
        "model_path": str(args.model_out),
    }
    args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== KKBox Customer-Month Result ===", flush=True)
    print(table.to_string(index=False), flush=True)
    print(f"Saved summary: {args.json_out}", flush=True)
    print(f"Saved model: {args.model_out}", flush=True)


if __name__ == "__main__":
    main()
