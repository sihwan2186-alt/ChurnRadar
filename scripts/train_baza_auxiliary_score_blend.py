#!/usr/bin/env python3
"""Blend the strongest transfer model with small Baza-local auxiliary signals."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_models import (
    build_candidates as build_public_candidates,
    load_data as load_public_data,
    make_pipeline as make_public_pipeline,
    score_model as score_public_model,
)
from scripts.train_baza_feature_engineering import (
    BazaFeatureBuilder,
    build_candidates as build_feature_candidates,
    load_baza as load_raw_baza,
    make_pipeline as make_feature_pipeline,
    model_scores as feature_model_scores,
)
from scripts.train_baza_transfer import (
    RANDOM_STATE,
    best_f1_threshold,
    build_candidates as build_transfer_candidates,
    build_source_groups,
    load_baza as load_common_baza,
)
from scripts.train_baza_transfer_blend import DEFAULT_SPECS, find_candidate, parse_member_spec, train_member
from src.models.ensemble import AveragingProbabilisticEnsemble
from src.utils.helpers import model_path, raw_data_path, result_path


def normalize_with_validation(val_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    lo = float(np.nanmin(val_scores))
    hi = float(np.nanmax(val_scores))
    if hi - lo < 1e-12:
        return np.zeros_like(val_scores, dtype=float), np.zeros_like(test_scores, dtype=float), {"min": lo, "max": hi}
    return (
        np.clip((val_scores - lo) / (hi - lo), 0.0, 1.0),
        np.clip((test_scores - lo) / (hi - lo), 0.0, 1.0),
        {"min": lo, "max": hi},
    )


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


def fit_transfer_blend(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    X_common, y = load_common_baza()
    X_train = X_common.iloc[train_idx].reset_index(drop=True)
    X_val = X_common.iloc[val_idx].reset_index(drop=True)
    X_test = X_common.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)

    source_groups = build_source_groups()
    candidates = build_transfer_candidates()
    fitted_members: list[tuple[str, Any]] = []
    score_ranges: dict[str, tuple[float, float]] = {}
    member_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for raw_spec in DEFAULT_SPECS:
        spec = parse_member_spec(raw_spec)
        candidate = find_candidate(spec, candidates)
        source_X, source_y = source_groups[spec.source_group]
        model = train_member(spec, candidate, source_groups, X_train, y_train)
        val_scores = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        threshold, val_f1 = best_f1_threshold(y.iloc[val_idx], val_scores)
        member_rows.append({
            "member": spec.name,
            "source_rows": int(len(source_X)),
            "source_churn_rate": float(source_y.mean()) if len(source_y) else 0.0,
            "val_best_f1": float(val_f1),
            "test": metrics(y.iloc[test_idx], test_scores, threshold),
        })
        fitted_members.append((spec.name, model))
        score_ranges[spec.name] = (float(np.nanmin(val_scores)), float(np.nanmax(val_scores)))

    blend = AveragingProbabilisticEnsemble(fitted_members, score_ranges=score_ranges)
    metadata = {
        "members": member_rows,
        "score_ranges": score_ranges,
        "train_seconds": round(time.perf_counter() - started, 3),
    }
    return blend, blend.predict_proba(X_val)[:, 1], blend.predict_proba(X_test)[:, 1], metadata


def fit_public_model(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    csv_path: Path,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    X, y, numeric_features, categorical_features = load_public_data(csv_path, "full")
    candidate = next(candidate for candidate in build_public_candidates(float((y == 0).sum() / max((y == 1).sum(), 1))) if candidate.name == "LDA_shrinkage")
    pipeline = make_public_pipeline(candidate, numeric_features, categorical_features)
    started = time.perf_counter()
    pipeline.fit(X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True))
    val_scores, _ = score_public_model(pipeline, X.iloc[val_idx].reset_index(drop=True))
    test_scores, _ = score_public_model(pipeline, X.iloc[test_idx].reset_index(drop=True))
    threshold, val_f1 = best_f1_threshold(y.iloc[val_idx], val_scores)
    metadata = {
        "model": candidate.name,
        "feature_set": "bulgaria_public_full",
        "features": list(X.columns),
        "val_best_f1": float(val_f1),
        "test": metrics(y.iloc[test_idx], test_scores, threshold),
        "train_seconds": round(time.perf_counter() - started, 3),
    }
    return pipeline, val_scores, test_scores, metadata


def fit_core_xgb_model(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    raw_csv_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    raw, y = load_raw_baza(raw_csv_path)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    builder = BazaFeatureBuilder("core")
    X_train = builder.fit_transform(raw.iloc[train_idx].reset_index(drop=True), y_train)
    X_val = builder.transform(raw.iloc[val_idx].reset_index(drop=True))
    X_test = builder.transform(raw.iloc[test_idx].reset_index(drop=True))

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    candidate = next(candidate for candidate in build_feature_candidates(scale_pos_weight) if candidate.name == "XGBoost_d1")
    pipeline = make_feature_pipeline(candidate)
    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    val_scores, _ = feature_model_scores(pipeline, X_val)
    test_scores, _ = feature_model_scores(pipeline, X_test)
    threshold, val_f1 = best_f1_threshold(y.iloc[val_idx], val_scores)
    bundle = {"builder": builder, "pipeline": pipeline, "feature_set": "core", "model": candidate.name}
    metadata = {
        "model": candidate.name,
        "feature_set": "baza_feature_core",
        "features": list(X_train.columns),
        "val_best_f1": float(val_f1),
        "test": metrics(y.iloc[test_idx], test_scores, threshold),
        "train_seconds": round(time.perf_counter() - started, 3),
    }
    return bundle, val_scores, test_scores, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-csv", type=Path, default=REPO_ROOT / "data" / "processed" / "baza_telecom_v2_bulgaria_public.csv")
    parser.add_argument("--raw-csv", type=Path, default=raw_data_path("baza_telecom_v2.csv"))
    parser.add_argument("--aux-weight", type=float, default=0.04)
    parser.add_argument("--public-share", type=float, default=0.5)
    parser.add_argument("--json-out", type=Path, default=result_path("baza_auxiliary_score_blend_summary.json"))
    parser.add_argument("--model-out", type=Path, default=model_path("baza_auxiliary_score_blend_model.joblib"))
    args = parser.parse_args()

    X_common, y = load_common_baza()
    indices = np.arange(len(y))
    train_val_idx, test_idx, y_train_val, _ = train_test_split(
        indices, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    train_idx, val_idx, _, _ = train_test_split(
        train_val_idx, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
    )
    y_val = y.iloc[val_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    transfer_model, transfer_val, transfer_test, transfer_meta = fit_transfer_blend(train_idx, val_idx, test_idx)
    public_model, public_val, public_test, public_meta = fit_public_model(train_idx, val_idx, test_idx, args.public_csv)
    core_xgb_model, core_xgb_val, core_xgb_test, core_xgb_meta = fit_core_xgb_model(train_idx, val_idx, test_idx, args.raw_csv)

    transfer_val_norm, transfer_test_norm, transfer_range = normalize_with_validation(transfer_val, transfer_test)
    public_val_norm, public_test_norm, public_range = normalize_with_validation(public_val, public_test)
    core_val_norm, core_test_norm, core_range = normalize_with_validation(core_xgb_val, core_xgb_test)

    public_weight = args.aux_weight * args.public_share
    core_xgb_weight = args.aux_weight * (1.0 - args.public_share)
    transfer_weight = 1.0 - args.aux_weight
    val_scores = transfer_weight * transfer_val_norm + public_weight * public_val_norm + core_xgb_weight * core_val_norm
    test_scores = transfer_weight * transfer_test_norm + public_weight * public_test_norm + core_xgb_weight * core_test_norm
    threshold, val_best_f1 = best_f1_threshold(y_val, val_scores)
    blend_metrics = metrics(y_test, test_scores, threshold)
    blend_metrics["val_best_f1"] = float(val_best_f1)

    args.json_out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    args.model_out = args.model_out if args.model_out.is_absolute() else REPO_ROOT / args.model_out
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "type": "auxiliary_score_blend",
        "threshold": float(threshold),
        "weights": {
            "transfer_blend": float(transfer_weight),
            "bulgaria_public_lda": float(public_weight),
            "baza_core_xgboost_d1": float(core_xgb_weight),
        },
        "score_ranges": {
            "transfer_blend": transfer_range,
            "bulgaria_public_lda": public_range,
            "baza_core_xgboost_d1": core_range,
        },
        "models": {
            "transfer_blend": transfer_model,
            "bulgaria_public_lda": public_model,
            "baza_core_xgboost_d1": core_xgb_model,
        },
    }
    joblib.dump(model_bundle, args.model_out)

    summary = {
        "rows": int(len(X_common)),
        "churn_rate": float(y.mean()),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "weights": model_bundle["weights"],
        "score_ranges": model_bundle["score_ranges"],
        "transfer": transfer_meta,
        "public": public_meta,
        "core_xgb": core_xgb_meta,
        "blend": blend_metrics,
        "model_path": str(args.model_out),
    }
    args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Auxiliary Score Blend Result ===", flush=True)
    print(
        f"f1={blend_metrics['f1']:.4f} precision={blend_metrics['precision']:.4f} "
        f"recall={blend_metrics['recall']:.4f} auc={blend_metrics['roc_auc']:.4f} "
        f"threshold={threshold:.6f}",
        flush=True,
    )
    print(f"Saved summary: {args.json_out}", flush=True)
    print(f"Saved model bundle: {args.model_out}", flush=True)


if __name__ == "__main__":
    main()
