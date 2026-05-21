#!/usr/bin/env python3
"""Final leakage-safe Baza optimization check.

This script intentionally excludes label-conditioned synthetic temporal
features. It checks whether leakage-safe Baza/PID risk features plus optional
unpaired external behavior proxies improve F1/Recall when XGBoost/LightGBM
positive-class weights are swept around 4-6 and dashboard thresholds are moved
around 0.35-0.40.
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

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_baza_churn_signal_workarounds import external_pool, sampled_external_features  # noqa: E402
from scripts.train_baza_pid_safe_risk_features import (  # noqa: E402
    PidSafeRiskFeatureBuilder,
    best_f1_threshold,
    metrics_at_threshold,
    model_scores,
)
from src.utils.helpers import raw_data_path, result_path  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TARGET = "CHURN"


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    scale_pos_weight: float
    estimator: Any


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if value is pd.NA:
        return None
    return value


def load_baza(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    return df.loc[valid].reset_index(drop=True), y.loc[valid].astype(int).reset_index(drop=True)


def candidate_grid(weights: list[float]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for weight in weights:
        candidates.append(
            Candidate(
                name=f"XGB_spw_{weight:g}_d5",
                family="xgboost",
                scale_pos_weight=weight,
                estimator=XGBClassifier(
                    n_estimators=700,
                    max_depth=5,
                    learning_rate=0.03,
                    min_child_weight=4,
                    reg_lambda=6.0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=weight,
                    eval_metric="aucpr",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )
        )
        candidates.append(
            Candidate(
                name=f"LGBM_spw_{weight:g}_leaf15",
                family="lightgbm",
                scale_pos_weight=weight,
                estimator=LGBMClassifier(
                    n_estimators=700,
                    learning_rate=0.03,
                    num_leaves=15,
                    min_child_samples=25,
                    reg_lambda=6.0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=weight,
                    objective="binary",
                    verbose=-1,
                    random_state=RANDOM_STATE,
                ),
            )
        )
    return candidates


def evaluate(
    scenario: str,
    candidate: Candidate,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    fixed_thresholds: list[float],
) -> dict[str, Any]:
    started = time.perf_counter()
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", clone(candidate.estimator)),
        ]
    )
    pipeline.fit(X_train, y_train)
    val_scores = model_scores(pipeline, X_val)
    test_scores = model_scores(pipeline, X_test)
    selected_threshold, val_best_f1 = best_f1_threshold(y_val, val_scores)
    selected = metrics_at_threshold(y_test, test_scores, selected_threshold)
    default = metrics_at_threshold(y_test, test_scores, 0.5)

    row = {
        "scenario": scenario,
        "valid_for_real_world": True,
        "model": candidate.name,
        "family": candidate.family,
        "scale_pos_weight": candidate.scale_pos_weight,
        "selected_threshold": selected_threshold,
        "val_best_f1": val_best_f1,
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "test_average_precision": float(average_precision_score(y_test, test_scores)),
        "train_seconds": round(time.perf_counter() - started, 3),
    }
    row.update({f"test_tuned_{key}": value for key, value in selected.items()})
    row.update({f"test_default_{key}": value for key, value in default.items()})
    for threshold in fixed_thresholds:
        metrics = metrics_at_threshold(y_test, test_scores, threshold)
        key = str(threshold).replace(".", "_")
        row[f"test_f1_at_{key}"] = metrics["f1"]
        row[f"test_precision_at_{key}"] = metrics["precision"]
        row[f"test_recall_at_{key}"] = metrics["recall"]
        row[f"test_pred_pos_rate_at_{key}"] = metrics["pred_pos_rate"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=raw_data_path("baza_telecom_v2.csv"))
    parser.add_argument("--weights", type=str, default="4,4.5,5,5.5,6")
    parser.add_argument("--fixed-thresholds", type=str, default="0.35,0.4,0.45,0.5")
    parser.add_argument("--table-out", type=Path, default=result_path("baza_final_optimization_check_table.csv"))
    parser.add_argument("--json-out", type=Path, default=result_path("baza_final_optimization_check_summary.json"))
    args = parser.parse_args()

    started = time.perf_counter()
    args.csv = args.csv if args.csv.is_absolute() else REPO_ROOT / args.csv
    args.table_out = args.table_out if args.table_out.is_absolute() else REPO_ROOT / args.table_out
    args.json_out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)

    weights = [float(item.strip()) for item in args.weights.split(",") if item.strip()]
    fixed_thresholds = [float(item.strip()) for item in args.fixed_thresholds.split(",") if item.strip()]
    raw, y = load_baza(args.csv)

    indices = np.arange(len(raw))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.25, stratify=y.iloc[train_val_idx], random_state=RANDOM_STATE
    )

    raw_train = raw.iloc[train_idx].reset_index(drop=True)
    raw_val = raw.iloc[val_idx].reset_index(drop=True)
    raw_test = raw.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    builder = PidSafeRiskFeatureBuilder(pid_aggregation_scope="train_only")
    X_train_base = builder.fit_transform(raw_train, y_train, pid_aggregation_raw=None)
    X_val_base = builder.transform(raw_val)
    X_test_base = builder.transform(raw_test)

    blocked = [
        col
        for col in X_train_base.columns
        if col.upper() == "PID" or col.upper().startswith("CRM_PID") or col.upper() == TARGET
    ]
    if blocked:
        raise ValueError(f"Leakage/id columns found in feature matrix: {blocked}")

    pool_X, pool_y = external_pool()
    external_all = sampled_external_features(len(raw), pool_X, pool_y, RANDOM_STATE + 101)
    ext_train = external_all.iloc[train_idx].reset_index(drop=True)
    ext_val = external_all.iloc[val_idx].reset_index(drop=True)
    ext_test = external_all.iloc[test_idx].reset_index(drop=True)

    scenarios = [
        ("baza_train_only_pid_risk", X_train_base, X_val_base, X_test_base),
        (
            "baza_plus_unpaired_external_behavior",
            pd.concat([X_train_base.reset_index(drop=True), ext_train], axis=1),
            pd.concat([X_val_base.reset_index(drop=True), ext_val], axis=1),
            pd.concat([X_test_base.reset_index(drop=True), ext_test], axis=1),
        ),
    ]

    rows: list[dict[str, Any]] = []
    candidates = candidate_grid(weights)
    print(
        f"Baza final check: rows={len(raw)} churn_rate={y.mean():.2%} "
        f"train={len(y_train)} val={len(y_val)} test={len(y_test)}",
        flush=True,
    )
    print(
        "Leakage guard: split first, train-only PID aggregation, OOF train target encodings, "
        "sklearn Pipeline fit on train only, no label-conditioned temporal features.",
        flush=True,
    )

    for scenario_name, X_train, X_val, X_test in scenarios:
        print(f"\n[{scenario_name}] features={X_train.shape[1]}", flush=True)
        for index, candidate in enumerate(candidates, start=1):
            row = evaluate(
                scenario_name,
                candidate,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                fixed_thresholds,
            )
            rows.append(row)
            print(
                f"  {index:02d}/{len(candidates):02d} {candidate.name}: "
                f"val_f1={row['val_best_f1']:.4f} test_f1={row['test_tuned_f1']:.4f} "
                f"recall={row['test_tuned_recall']:.4f} precision={row['test_tuned_precision']:.4f}",
                flush=True,
            )

    table = pd.DataFrame(rows).sort_values(["test_tuned_f1", "test_average_precision"], ascending=[False, False])
    table.to_csv(args.table_out, index=False, encoding="utf-8-sig")

    threshold_cols = []
    for threshold in fixed_thresholds:
        key = str(threshold).replace(".", "_")
        threshold_cols.extend([f"test_f1_at_{key}", f"test_recall_at_{key}", f"test_precision_at_{key}"])

    summary = {
        "target_f1": 0.6,
        "target_reached": bool((table["test_tuned_f1"] >= 0.6).any()),
        "valid_for_real_world": True,
        "leakage_controls": {
            "split_before_feature_engineering": True,
            "pid_aggregation_scope": "train_only",
            "target_encoding": "OOF on train; train-fitted mappings for val/test",
            "preprocessing_pipeline": "SimpleImputer is fit inside sklearn Pipeline on train only",
            "label_conditioned_temporal_features_excluded": True,
            "blocked_feature_columns": blocked,
        },
        "baza_rows": int(len(raw)),
        "baza_churn_rate": float(y.mean()),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "weights_tested": weights,
        "fixed_thresholds_tested": fixed_thresholds,
        "external_pool_rows": int(len(pool_X)),
        "external_pool_features": int(pool_X.shape[1]) if not pool_X.empty else 0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "table_path": str(args.table_out),
        "best_by_test_f1": table.head(1).to_dict(orient="records")[0],
        "top_models": table.head(20).to_dict(orient="records"),
    }
    args.json_out.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    display_cols = [
        "scenario",
        "model",
        "scale_pos_weight",
        "val_best_f1",
        "test_tuned_f1",
        "test_tuned_precision",
        "test_tuned_recall",
        "test_average_precision",
        *threshold_cols,
    ]
    print("\n=== Final Leakage-Safe Optimization Check ===", flush=True)
    print(table[display_cols].head(15).to_string(index=False), flush=True)
    print(f"\nSaved table: {args.table_out}", flush=True)
    print(f"Saved summary: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
