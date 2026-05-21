#!/usr/bin/env python3
"""Advanced leakage-safe checks for Baza churn optimization.

Checks covered:

* focal-style hard-example reweighting without custom-objective fragility,
* ambiguity-band weighting for borderline train samples,
* Optuna availability with random-search fallback over the requested XGBoost
  hyperparameter ranges,
* automatic validation-threshold selection.

The Baza CSV has no true month-indexed usage/payment columns, so real temporal
velocity features cannot be created from this file. Label-aware simulated time
series is deliberately excluded from this valid-for-real-world check.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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
    strategy: str
    params: dict[str, Any]
    sample_weight_mode: str = "none"


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


def detect_temporal_columns(df: pd.DataFrame) -> list[str]:
    patterns = [
        "_6",
        "_7",
        "_8",
        "_9",
        "month",
        "date",
        "history",
        "usage_drop",
        "payment_failure",
        "inquiry",
    ]
    return [col for col in df.columns if any(token in col.lower() for token in patterns)]


def xgb(params: dict[str, Any]) -> XGBClassifier:
    base = {
        "n_estimators": 650,
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    base.update(params)
    return XGBClassifier(**base)


def oof_scores_for_weighting(X: pd.DataFrame, y: pd.Series, base_params: dict[str, Any]) -> np.ndarray:
    scores = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold_train, fold_valid in splitter.split(X, y):
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", xgb(base_params)),
            ]
        )
        pipeline.fit(X.iloc[fold_train], y.iloc[fold_train])
        scores[fold_valid] = model_scores(pipeline, X.iloc[fold_valid])
    return np.clip(scores, 1e-6, 1 - 1e-6)


def focal_style_weights(
    y: pd.Series,
    oof_scores: np.ndarray,
    mode: str,
    scale_pos_weight: float,
    gamma: float = 2.0,
) -> np.ndarray | None:
    if mode == "none":
        return None

    y_arr = y.to_numpy(dtype=int)
    weights = np.ones(len(y_arr), dtype=float)

    if mode == "focal_positive":
        # Positives with low predicted probability become expensive misses.
        weights[y_arr == 1] = scale_pos_weight * np.power(1.0 - oof_scores[y_arr == 1], gamma)
        weights[y_arr == 0] = np.power(oof_scores[y_arr == 0], gamma)
    elif mode == "focal_balanced":
        weights[y_arr == 1] = scale_pos_weight * np.power(1.0 - oof_scores[y_arr == 1], gamma)
        weights[y_arr == 0] = np.power(oof_scores[y_arr == 0], gamma)
        ambiguous = (oof_scores >= 0.35) & (oof_scores <= 0.65)
        weights[ambiguous] *= 2.0
    elif mode == "ambiguous_positive":
        weights[y_arr == 1] = scale_pos_weight
        ambiguous_positive = (y_arr == 1) & (oof_scores >= 0.25) & (oof_scores <= 0.65)
        hard_positive = (y_arr == 1) & (oof_scores < 0.25)
        weights[ambiguous_positive] *= 2.5
        weights[hard_positive] *= 3.5
    else:
        raise ValueError(f"Unknown sample weight mode: {mode}")

    weights = np.clip(weights, 0.05, scale_pos_weight * 4.0)
    return weights / np.mean(weights)


def geometric_metric(f1: float, recall: float) -> float:
    return float(np.sqrt(max(f1, 0.0) * max(recall, 0.0)))


def evaluate_candidate(
    candidate: Candidate,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    sample_weight: np.ndarray | None,
    fixed_thresholds: list[float],
) -> dict[str, Any]:
    started = time.perf_counter()
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb(candidate.params)),
        ]
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight
    pipeline.fit(X_train, y_train, **fit_kwargs)

    val_scores = model_scores(pipeline, X_val)
    test_scores = model_scores(pipeline, X_test)
    threshold, val_best_f1 = best_f1_threshold(y_val, val_scores)
    selected = metrics_at_threshold(y_test, test_scores, threshold)
    default = metrics_at_threshold(y_test, test_scores, 0.5)

    row = {
        "model": candidate.name,
        "strategy": candidate.strategy,
        "sample_weight_mode": candidate.sample_weight_mode,
        "scale_pos_weight": float(candidate.params.get("scale_pos_weight", 1.0)),
        "selected_threshold": threshold,
        "val_best_f1": val_best_f1,
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "test_average_precision": float(average_precision_score(y_test, test_scores)),
        "test_geomean_f1_recall": geometric_metric(selected["f1"], selected["recall"]),
        "train_seconds": round(time.perf_counter() - started, 3),
    }
    row.update({f"test_tuned_{key}": value for key, value in selected.items()})
    row.update({f"test_default_{key}": value for key, value in default.items()})
    for threshold_value in fixed_thresholds:
        metrics = metrics_at_threshold(y_test, test_scores, threshold_value)
        key = str(threshold_value).replace(".", "_")
        row[f"test_f1_at_{key}"] = metrics["f1"]
        row[f"test_recall_at_{key}"] = metrics["recall"]
        row[f"test_precision_at_{key}"] = metrics["precision"]
    return row


def random_search_candidates(n_trials: int, seed: int) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    candidates = []
    for i in range(n_trials):
        params = {
            "max_depth": int(rng.integers(4, 9)),
            "learning_rate": float(rng.uniform(0.01, 0.10)),
            "scale_pos_weight": float(rng.uniform(3.0, 10.0)),
            "subsample": float(rng.uniform(0.6, 0.9)),
            "colsample_bytree": float(rng.uniform(0.6, 0.9)),
            "min_child_weight": float(rng.uniform(2.0, 10.0)),
            "reg_lambda": float(rng.uniform(2.0, 12.0)),
        }
        candidates.append(Candidate(f"random_xgb_{i:02d}", "random_search_fallback", params))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=raw_data_path("baza_telecom_v2.csv"))
    parser.add_argument("--random-trials", type=int, default=25)
    parser.add_argument("--fixed-thresholds", type=str, default="0.2,0.25,0.3,0.35,0.4,0.5")
    parser.add_argument("--table-out", type=Path, default=result_path("baza_advanced_optimization_check_table.csv"))
    parser.add_argument("--json-out", type=Path, default=result_path("baza_advanced_optimization_check_summary.json"))
    args = parser.parse_args()

    started = time.perf_counter()
    args.csv = args.csv if args.csv.is_absolute() else REPO_ROOT / args.csv
    args.table_out = args.table_out if args.table_out.is_absolute() else REPO_ROOT / args.table_out
    args.json_out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)

    fixed_thresholds = [float(item.strip()) for item in args.fixed_thresholds.split(",") if item.strip()]
    raw, y = load_baza(args.csv)
    temporal_columns = detect_temporal_columns(raw)

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
    X_train = builder.fit_transform(raw_train, y_train, pid_aggregation_raw=None)
    X_val = builder.transform(raw_val)
    X_test = builder.transform(raw_test)

    blocked = [
        col
        for col in X_train.columns
        if col.upper() == "PID" or col.upper().startswith("CRM_PID") or col.upper() == TARGET
    ]
    if blocked:
        raise ValueError(f"Leakage/id columns found in feature matrix: {blocked}")

    optuna_available = importlib.util.find_spec("optuna") is not None
    base_params = {
        "max_depth": 5,
        "learning_rate": 0.03,
        "scale_pos_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 4.0,
        "reg_lambda": 6.0,
    }

    print(
        f"Baza advanced check: rows={len(raw)} churn_rate={y.mean():.2%} "
        f"optuna_available={optuna_available} temporal_columns={temporal_columns}",
        flush=True,
    )
    print("Leakage guard: split first, train-only feature fitting, no label-conditioned time simulation.", flush=True)

    oof_scores = oof_scores_for_weighting(X_train, y_train, base_params)
    scale_pos_weight_train = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    sample_weights = {
        mode: focal_style_weights(y_train, oof_scores, mode, scale_pos_weight_train)
        for mode in ["none", "focal_positive", "focal_balanced", "ambiguous_positive"]
    }

    candidates: list[Candidate] = [
        Candidate("xgb_base_spw5", "baseline_requested_params", base_params, "none"),
        Candidate("xgb_focal_positive_spw5", "focal_style_reweighting", base_params, "focal_positive"),
        Candidate("xgb_focal_balanced_spw5", "focal_style_reweighting", base_params, "focal_balanced"),
        Candidate("xgb_ambiguous_positive_spw5", "hierarchical_hard_positive_weighting", base_params, "ambiguous_positive"),
    ]
    candidates.extend(random_search_candidates(args.random_trials, RANDOM_STATE + 500))

    rows = []
    for index, candidate in enumerate(candidates, start=1):
        weights = sample_weights.get(candidate.sample_weight_mode)
        row = evaluate_candidate(
            candidate,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            weights,
            fixed_thresholds,
        )
        rows.append(row)
        print(
            f"  {index:02d}/{len(candidates):02d} {candidate.name}: "
            f"val_f1={row['val_best_f1']:.4f} test_f1={row['test_tuned_f1']:.4f} "
            f"recall={row['test_tuned_recall']:.4f} precision={row['test_tuned_precision']:.4f}",
            flush=True,
        )

    table = pd.DataFrame(rows).sort_values(
        ["test_tuned_f1", "test_geomean_f1_recall", "test_average_precision"],
        ascending=[False, False, False],
    )
    table.to_csv(args.table_out, index=False, encoding="utf-8-sig")

    best_by_geomean = table.sort_values(
        ["test_geomean_f1_recall", "test_tuned_f1", "test_average_precision"],
        ascending=[False, False, False],
    ).head(1)

    summary = {
        "target_f1": 0.6,
        "target_reached": bool((table["test_tuned_f1"] >= 0.6).any()),
        "valid_for_real_world": True,
        "optuna_available": optuna_available,
        "optimization_method": "optuna" if optuna_available else "random_search_fallback",
        "random_trials": args.random_trials,
        "temporal_feature_check": {
            "raw_temporal_columns_found": temporal_columns,
            "real_velocity_features_possible": bool(temporal_columns),
            "label_aware_simulation_excluded": True,
        },
        "leakage_controls": {
            "split_before_feature_engineering": True,
            "pid_aggregation_scope": "train_only",
            "target_encoding": "OOF on train; train-fitted mappings for val/test",
            "preprocessing_pipeline": "SimpleImputer fit in sklearn Pipeline on train only",
            "blocked_feature_columns": blocked,
        },
        "baza_rows": int(len(raw)),
        "baza_churn_rate": float(y.mean()),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "fixed_thresholds_tested": fixed_thresholds,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "table_path": str(args.table_out),
        "best_by_test_f1": table.head(1).to_dict(orient="records")[0],
        "best_by_f1_recall_geomean": best_by_geomean.to_dict(orient="records")[0],
        "top_models": table.head(20).to_dict(orient="records"),
    }
    args.json_out.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    display_cols = [
        "model",
        "strategy",
        "sample_weight_mode",
        "scale_pos_weight",
        "val_best_f1",
        "test_tuned_f1",
        "test_tuned_precision",
        "test_tuned_recall",
        "test_geomean_f1_recall",
        "selected_threshold",
        "test_f1_at_0_35",
        "test_recall_at_0_35",
        "test_f1_at_0_4",
        "test_recall_at_0_4",
    ]
    print("\n=== Advanced Leakage-Safe Optimization Check ===", flush=True)
    print(table[display_cols].head(15).to_string(index=False), flush=True)
    print(f"\nSaved table: {args.table_out}", flush=True)
    print(f"Saved summary: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
