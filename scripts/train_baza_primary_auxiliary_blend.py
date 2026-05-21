#!/usr/bin/env python3
"""Baza-primary churn experiment with auxiliary-only support models.

This script keeps Baza as the main dataset:

* Baza train/validation/test rows are split once and never mixed with
  auxiliary validation/test rows.
* Baza-only feature models use the richer Baza engineered features.
* Auxiliary datasets are used only in transfer models, with capped sample
  weights so they cannot dominate Baza.
* Final score blends are selected only on the Baza validation split.
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
from sklearn.exceptions import ConvergenceWarning
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

from scripts.train_baza_feature_engineering import (  # noqa: E402
    BazaFeatureBuilder,
    build_candidates as build_baza_candidates,
    evaluate_scores,
    load_baza as load_raw_baza,
    make_pipeline as make_baza_pipeline,
    model_scores,
    normalize_with_val,
)
from scripts.train_baza_transfer import (  # noqa: E402
    COMMON_FEATURES,
    build_candidates as build_transfer_candidates,
    build_source_groups,
    evaluate_candidate,
    load_baza as load_common_baza,
)
from src.models.threshold_model import ThresholdClassifier  # noqa: E402
from src.utils.helpers import model_path, raw_data_path, result_path  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

DEFAULT_BAZA_FEATURE_SETS = "core,engineered,encoded_no_ka,encoded_compact"
DEFAULT_BAZA_MODELS = ",".join(
    [
        "LogReg_L1_balanced",
        "LogReg_L2_C03_balanced",
        "Ridge_balanced_calibrated",
        "GradientBoosting_d2",
        "GradientBoosting_d3",
        "LightGBM_leaf7",
        "XGBoost_d1",
        "XGBoost_d2",
        "BalancedRF_d5",
        "EasyEnsemble_10",
        "EasyEnsemble_30",
    ]
)
DEFAULT_AUX_GROUPS = "baza_only,cell,all,ibm_cell_kdd,all_kdd,hf_all,hf_all_orange,cell_orange"
DEFAULT_AUX_MODELS = "LR_balanced,LGBM_balanced,LGBM_regularized,XGB_weighted"
DEFAULT_AUX_WEIGHTS = "0.05,0.1,0.25,0.5"


@dataclass
class ScoreRecord:
    name: str
    branch: str
    family: str
    val_f1: float
    val_scores: np.ndarray
    test_scores: np.ndarray
    row: dict[str, Any]
    model: Any | None = None


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_csv(value: str | None) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
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


def metric_row(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "test_tuned_f1": float(f1_score(y_true, pred, zero_division=0)),
        "test_tuned_precision": float(precision_score(y_true, pred, zero_division=0)),
        "test_tuned_recall": float(recall_score(y_true, pred, zero_division=0)),
        "test_tuned_tn": int(tn),
        "test_tuned_fp": int(fp),
        "test_tuned_fn": int(fn),
        "test_tuned_tp": int(tp),
        "test_tuned_pred_pos_rate": float(np.mean(pred)),
    }


def run_baza_feature_branch(
    raw_train: pd.DataFrame,
    raw_val: pd.DataFrame,
    raw_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    feature_sets: list[str],
    model_names: set[str],
) -> tuple[list[dict[str, Any]], list[ScoreRecord], dict[tuple[str, str], Any]]:
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    candidates = [
        candidate
        for candidate in build_baza_candidates(scale_pos_weight)
        if candidate.name in model_names
    ]
    if not candidates:
        raise ValueError("No Baza feature candidates matched --baza-models.")

    rows: list[dict[str, Any]] = []
    records: list[ScoreRecord] = []
    fitted: dict[tuple[str, str], Any] = {}

    for feature_set in feature_sets:
        builder = BazaFeatureBuilder(feature_set)
        X_train = builder.fit_transform(raw_train, y_train)
        X_val = builder.transform(raw_val)
        X_test = builder.transform(raw_test)
        print(
            f"\n[Baza features] {feature_set}: features={X_train.shape[1]} "
            f"models={len(candidates)}",
            flush=True,
        )
        for index, candidate in enumerate(candidates, start=1):
            started = time.perf_counter()
            pipeline = make_baza_pipeline(candidate)
            row: dict[str, Any]
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    pipeline.fit(X_train, y_train)
                val_scores, score_type = model_scores(pipeline, X_val)
                test_scores, _ = model_scores(pipeline, X_test)
                row = evaluate_scores(y_val, y_test, val_scores, test_scores)
                row.update(
                    {
                        "branch": "baza_feature",
                        "feature_set": feature_set,
                        "source_group": "baza_only",
                        "source_weight": 0.0,
                        "model": candidate.name,
                        "family": candidate.family,
                        "score_type": score_type,
                        "status": "ok",
                        "error": "",
                        "train_seconds": round(time.perf_counter() - started, 3),
                    }
                )
                fitted[(feature_set, candidate.name)] = {
                    "builder": builder,
                    "model": ThresholdClassifier(pipeline, row["threshold"]),
                    "feature_columns": list(X_train.columns),
                }
                records.append(
                    ScoreRecord(
                        name=f"baza_feature::{feature_set}::{candidate.name}",
                        branch="baza_feature",
                        family=candidate.family,
                        val_f1=float(row["val_best_f1"]),
                        val_scores=val_scores,
                        test_scores=test_scores,
                        row=row,
                        model=fitted[(feature_set, candidate.name)],
                    )
                )
                print(
                    f"  [{index:02d}/{len(candidates):02d}] {candidate.name}: "
                    f"val_f1={row['val_best_f1']:.4f} "
                    f"test_f1={row['test_tuned_f1']:.4f} "
                    f"recall={row['test_tuned_recall']:.4f}",
                    flush=True,
                )
            except Exception as exc:
                row = {
                    "branch": "baza_feature",
                    "feature_set": feature_set,
                    "source_group": "baza_only",
                    "source_weight": 0.0,
                    "model": candidate.name,
                    "family": candidate.family,
                    "score_type": "",
                    "status": "failed",
                    "error": repr(exc),
                    "train_seconds": round(time.perf_counter() - started, 3),
                }
                print(f"  [{index:02d}/{len(candidates):02d}] {candidate.name}: failed {exc!r}", flush=True)
            rows.append(row)
    return rows, records, fitted


def run_auxiliary_branch(
    common_train: pd.DataFrame,
    common_val: pd.DataFrame,
    common_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    groups: set[str],
    models: set[str],
    weights: set[float],
) -> tuple[list[dict[str, Any]], list[ScoreRecord], dict[tuple[str, float, str], Any]]:
    source_groups = build_source_groups()
    candidates = [
        candidate
        for candidate in build_transfer_candidates()
        if candidate.source_group in groups
        and candidate.model_name in models
        and candidate.source_weight in weights
        and candidate.source_group in source_groups
    ]
    if not candidates:
        raise ValueError("No auxiliary transfer candidates matched the requested filters.")

    print("\n[Auxiliary transfer] available groups:", flush=True)
    for name in sorted(groups):
        if name not in source_groups:
            print(f"  {name}: missing", flush=True)
            continue
        source_X, source_y = source_groups[name]
        churn_rate = float(source_y.mean()) if len(source_y) else 0.0
        print(f"  {name}: rows={len(source_X)} churn_rate={churn_rate:.2%}", flush=True)

    rows: list[dict[str, Any]] = []
    records: list[ScoreRecord] = []
    fitted: dict[tuple[str, float, str], Any] = {}

    for index, candidate in enumerate(candidates, start=1):
        source_X, source_y = source_groups[candidate.source_group]
        row, tuned = evaluate_candidate(
            candidate,
            source_X,
            source_y,
            common_train,
            common_val,
            common_test,
            y_train,
            y_val,
            y_test,
        )
        val_scores = tuned.estimator.predict_proba(common_val)[:, 1]
        test_scores = tuned.estimator.predict_proba(common_test)[:, 1]
        row.update(
            {
                "branch": "auxiliary_transfer",
                "feature_set": "common_schema",
                "family": candidate.model_name,
                "score_type": "proba",
                "status": "ok",
                "error": "",
                "test_tuned_precision": row["test_precision"],
                "test_tuned_recall": row["test_recall"],
                "test_tuned_tn": row["tn"],
                "test_tuned_fp": row["fp"],
                "test_tuned_fn": row["fn"],
                "test_tuned_tp": row["tp"],
            }
        )
        key = (candidate.source_group, candidate.source_weight, candidate.model_name)
        fitted[key] = tuned
        records.append(
            ScoreRecord(
                name=f"aux::{candidate.source_group}::w{candidate.source_weight}::{candidate.model_name}",
                branch="auxiliary_transfer",
                family=candidate.model_name,
                val_f1=float(row["val_best_f1"]),
                val_scores=val_scores,
                test_scores=test_scores,
                row=row,
                model=tuned,
            )
        )
        rows.append(row)
        print(
            f"  [{index:03d}/{len(candidates):03d}] {candidate.source_group} "
            f"w={candidate.source_weight:g} {candidate.model_name}: "
            f"val_f1={row['val_best_f1']:.4f} test_f1={row['test_tuned_f1']:.4f} "
            f"recall={row['test_recall']:.4f}",
            flush=True,
        )
    return rows, records, fitted


def blend_scores(records: list[ScoreRecord], weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for record in records:
        val_norm, test_norm = normalize_with_val(record.val_scores, record.test_scores)
        val_parts.append(val_norm)
        test_parts.append(test_norm)
    val_matrix = np.vstack(val_parts)
    test_matrix = np.vstack(test_parts)
    weight_arr = weights / weights.sum()
    return np.average(val_matrix, axis=0, weights=weight_arr), np.average(test_matrix, axis=0, weights=weight_arr)


def build_blends(
    records: list[ScoreRecord],
    y_val: pd.Series,
    y_test: pd.Series,
    max_baza_members: int,
    max_aux_members: int,
    aux_blend_weights: list[float],
) -> tuple[list[dict[str, Any]], list[ScoreRecord]]:
    baza_records = sorted(
        [record for record in records if record.branch == "baza_feature"],
        key=lambda item: item.val_f1,
        reverse=True,
    )[:max_baza_members]
    aux_records = sorted(
        [record for record in records if record.branch == "auxiliary_transfer"],
        key=lambda item: item.val_f1,
        reverse=True,
    )[:max_aux_members]
    if not baza_records:
        return [], []

    blend_rows: list[dict[str, Any]] = []
    blend_records: list[ScoreRecord] = []

    def append_blend(name: str, members: list[ScoreRecord], weights: np.ndarray, aux_weight: float) -> None:
        val_scores, test_scores = blend_scores(members, weights)
        row = evaluate_scores(y_val, y_test, val_scores, test_scores)
        row.update(
            {
                "branch": "validation_blend",
                "feature_set": "blend",
                "source_group": "baza_primary_auxiliary",
                "source_weight": aux_weight,
                "model": name,
                "family": "score_blend",
                "score_type": "normalized_validation_blend",
                "status": "ok",
                "error": "",
                "train_seconds": 0.0,
                "blend_members": ";".join(member.name for member in members),
            }
        )
        blend_rows.append(row)
        blend_records.append(
            ScoreRecord(
                name=f"blend::{name}",
                branch="validation_blend",
                family="score_blend",
                val_f1=float(row["val_best_f1"]),
                val_scores=val_scores,
                test_scores=test_scores,
                row=row,
            )
        )

    for k in range(1, min(5, len(baza_records)) + 1):
        members = baza_records[:k]
        val_weights = np.asarray([max(member.val_f1, 1e-6) for member in members], dtype=float)
        append_blend(f"baza_only_top{k}_by_val", members, val_weights, aux_weight=0.0)

    if aux_records:
        aux_pool = aux_records[:max_aux_members]
        aux_weights_base = np.asarray([max(member.val_f1, 1e-6) for member in aux_pool], dtype=float)
        baza_pool = baza_records[:max_baza_members]
        baza_weights_base = np.asarray([max(member.val_f1, 1e-6) for member in baza_pool], dtype=float)

        for aux_weight in aux_blend_weights:
            baza_weight = 1.0 - aux_weight
            members = baza_pool + aux_pool
            weights = np.concatenate(
                [
                    baza_weights_base / baza_weights_base.sum() * baza_weight,
                    aux_weights_base / aux_weights_base.sum() * aux_weight,
                ]
            )
            append_blend(
                f"baza_primary_aux_weight_{aux_weight:.2f}",
                members,
                weights,
                aux_weight=aux_weight,
            )

        # Single best Baza model plus best auxiliary model is deliberately simple
        # and often more stable than large score blends on this small Baza test set.
        for aux_weight in aux_blend_weights:
            members = [baza_records[0], aux_records[0]]
            weights = np.asarray([1.0 - aux_weight, aux_weight], dtype=float)
            append_blend(
                f"best_baza_plus_best_aux_weight_{aux_weight:.2f}",
                members,
                weights,
                aux_weight=aux_weight,
            )

    return blend_rows, blend_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=raw_data_path("baza_telecom_v2.csv"))
    parser.add_argument("--target-f1", type=float, default=0.6)
    parser.add_argument("--baza-feature-sets", type=str, default=DEFAULT_BAZA_FEATURE_SETS)
    parser.add_argument("--baza-models", type=str, default=DEFAULT_BAZA_MODELS)
    parser.add_argument("--aux-groups", type=str, default=DEFAULT_AUX_GROUPS)
    parser.add_argument("--aux-models", type=str, default=DEFAULT_AUX_MODELS)
    parser.add_argument("--aux-weights", type=str, default=DEFAULT_AUX_WEIGHTS)
    parser.add_argument("--blend-aux-weights", type=str, default="0.05,0.1,0.2,0.3,0.35")
    parser.add_argument("--max-baza-blend-members", type=int, default=4)
    parser.add_argument("--max-aux-blend-members", type=int, default=4)
    parser.add_argument("--table-out", type=Path, default=result_path("baza_primary_auxiliary_blend_table.csv"))
    parser.add_argument("--json-out", type=Path, default=result_path("baza_primary_auxiliary_blend_summary.json"))
    parser.add_argument("--model-out", type=Path, default=model_path("baza_primary_auxiliary_best_individual.joblib"))
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    args.csv = args.csv if args.csv.is_absolute() else REPO_ROOT / args.csv
    raw, y = load_raw_baza(args.csv)
    common_X, common_y = load_common_baza()
    if len(common_X) != len(raw) or not np.array_equal(common_y.to_numpy(), y.to_numpy()):
        raise ValueError("Baza raw and common-schema loaders are not aligned.")

    indices = np.arange(len(raw))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y.iloc[train_val_idx],
    )

    raw_train = raw.iloc[train_idx].reset_index(drop=True)
    raw_val = raw.iloc[val_idx].reset_index(drop=True)
    raw_test = raw.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    common_train = common_X.iloc[train_idx].reset_index(drop=True)
    common_val = common_X.iloc[val_idx].reset_index(drop=True)
    common_test = common_X.iloc[test_idx].reset_index(drop=True)

    started = time.perf_counter()
    print(
        f"Baza-primary split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"churn_rate={y.mean():.2%}",
        flush=True,
    )
    print(
        "Overfit controls: validation-only threshold/blend selection, "
        "OOF target encoding, no KA target encoding by default, auxiliary weights capped.",
        flush=True,
    )

    all_rows: list[dict[str, Any]] = []
    all_records: list[ScoreRecord] = []

    baza_rows, baza_records, baza_fitted = run_baza_feature_branch(
        raw_train,
        raw_val,
        raw_test,
        y_train,
        y_val,
        y_test,
        parse_csv(args.baza_feature_sets),
        set(parse_csv(args.baza_models)),
    )
    all_rows.extend(baza_rows)
    all_records.extend(baza_records)

    aux_rows, aux_records, aux_fitted = run_auxiliary_branch(
        common_train,
        common_val,
        common_test,
        y_train,
        y_val,
        y_test,
        set(parse_csv(args.aux_groups)),
        set(parse_csv(args.aux_models)),
        set(parse_float_csv(args.aux_weights)),
    )
    all_rows.extend(aux_rows)
    all_records.extend(aux_records)

    blend_rows, blend_records = build_blends(
        all_records,
        y_val,
        y_test,
        max_baza_members=args.max_baza_blend_members,
        max_aux_members=args.max_aux_blend_members,
        aux_blend_weights=parse_float_csv(args.blend_aux_weights),
    )
    all_rows.extend(blend_rows)
    all_records.extend(blend_records)

    table = pd.DataFrame(all_rows)
    ok_table = table[table["status"] == "ok"].copy()
    ok_table["selection_rank"] = ok_table["val_best_f1"].rank(method="first", ascending=False)
    final_table = pd.concat(
        [
            ok_table.sort_values(["val_best_f1", "test_roc_auc"], ascending=[False, False]),
            table[table["status"] != "ok"],
        ],
        ignore_index=True,
    )

    # Final recommendation is validation-only, but it also follows the
    # experiment policy: Baza remains primary, auxiliary rows are only support,
    # and tiny pairwise blends are treated as diagnostics because they are too
    # easy to tune to one validation split.
    primary_mask = (
        (ok_table["branch"] == "validation_blend")
        & ok_table["model"].astype(str).str.startswith("baza_primary_aux_weight_")
    )
    primary_candidates = ok_table[primary_mask].copy()
    if primary_candidates.empty:
        primary_candidates = ok_table[ok_table["branch"].isin(["baza_feature", "validation_blend"])].copy()
    recommended_by_policy = primary_candidates.sort_values(
        ["val_best_f1", "source_weight"], ascending=[False, True]
    ).iloc[0].to_dict()

    # Diagnostics below are reported after the recommendation is chosen; test
    # columns are not used by the recommendation rule.
    best_by_validation = ok_table.sort_values(["val_best_f1", "source_weight"], ascending=[False, True]).iloc[0].to_dict()
    best_test_observed = ok_table.sort_values(["test_tuned_f1", "test_roc_auc"], ascending=[False, False]).iloc[0].to_dict()
    best_baza_only = (
        ok_table[ok_table["branch"] == "baza_feature"]
        .sort_values(["val_best_f1", "test_roc_auc"], ascending=[False, False])
        .iloc[0]
        .to_dict()
    )
    best_auxiliary = (
        ok_table[ok_table["branch"] == "auxiliary_transfer"]
        .sort_values(["val_best_f1", "test_roc_auc"], ascending=[False, False])
        .iloc[0]
        .to_dict()
    )

    args.table_out = args.table_out if args.table_out.is_absolute() else REPO_ROOT / args.table_out
    args.json_out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    args.model_out = args.model_out if args.model_out.is_absolute() else REPO_ROOT / args.model_out
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    final_table.to_csv(args.table_out, index=False, encoding="utf-8-sig")

    saved_model_path = ""
    if recommended_by_policy["branch"] == "baza_feature":
        key = (recommended_by_policy["feature_set"], recommended_by_policy["model"])
        joblib.dump(baza_fitted[key], args.model_out)
        saved_model_path = str(args.model_out)
    elif recommended_by_policy["branch"] == "auxiliary_transfer":
        key = (
            recommended_by_policy["source_group"],
            float(recommended_by_policy["source_weight"]),
            recommended_by_policy["model"],
        )
        joblib.dump(aux_fitted[key], args.model_out)
        saved_model_path = str(args.model_out)

    summary = {
        "target_f1": args.target_f1,
        "target_reached_by_policy_recommendation": bool(recommended_by_policy["test_tuned_f1"] >= args.target_f1),
        "target_reached_any_test_observed": bool((ok_table["test_tuned_f1"] >= args.target_f1).any()),
        "selection_rule": (
            "recommended row is the highest-validation Baza-primary multi-member blend; "
            "two-member best-baza+best-aux blends are diagnostic only"
        ),
        "overfit_controls": [
            "Baza is the only validation/test domain.",
            "Auxiliary rows are used only in training.",
            "Auxiliary sample weights are capped by --aux-weights.",
            "Baza target encodings are out-of-fold on train and fitted only on train.",
            "High-cardinality KA target encoding is excluded by default.",
            "Blend weights and thresholds are selected only on validation scores.",
        ],
        "baza_rows": int(len(raw)),
        "baza_churn_rate": float(y.mean()),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "common_features": COMMON_FEATURES,
        "baza_feature_sets": parse_csv(args.baza_feature_sets),
        "baza_models": parse_csv(args.baza_models),
        "aux_groups": parse_csv(args.aux_groups),
        "aux_models": parse_csv(args.aux_models),
        "aux_weights": parse_float_csv(args.aux_weights),
        "recommended_by_policy": recommended_by_policy,
        "best_by_validation_diagnostic": best_by_validation,
        "best_test_observed_not_for_selection": best_test_observed,
        "best_baza_only_by_validation": best_baza_only,
        "best_auxiliary_by_validation": best_auxiliary,
        "row_count": int(len(final_table)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "table_path": str(args.table_out),
        "saved_individual_model_path": saved_model_path,
        "top_by_validation": ok_table.sort_values(["val_best_f1", "test_roc_auc"], ascending=[False, False])
        .head(args.top)
        .to_dict(orient="records"),
        "top_by_test_observed_not_for_selection": ok_table.sort_values(
            ["test_tuned_f1", "test_roc_auc"], ascending=[False, False]
        )
        .head(args.top)
        .to_dict(orient="records"),
    }
    args.json_out.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    display_cols = [
        "branch",
        "feature_set",
        "source_group",
        "source_weight",
        "model",
        "val_best_f1",
        "test_tuned_f1",
        "test_tuned_precision",
        "test_tuned_recall",
        "test_roc_auc",
        "threshold",
    ]
    print("\n=== Top by validation F1 (selection view) ===", flush=True)
    print(ok_table.sort_values(["val_best_f1", "test_roc_auc"], ascending=[False, False])[display_cols].head(args.top).to_string(index=False), flush=True)
    print("\n=== Top by test F1 (diagnostic only, not selection) ===", flush=True)
    print(ok_table.sort_values(["test_tuned_f1", "test_roc_auc"], ascending=[False, False])[display_cols].head(args.top).to_string(index=False), flush=True)
    print(f"\nSaved table: {args.table_out}", flush=True)
    print(f"Saved summary: {args.json_out}", flush=True)
    if saved_model_path:
        print(f"Saved selected individual model: {saved_model_path}", flush=True)
    print(
        "\nRecommended Baza-primary result: "
        f"F1={recommended_by_policy['test_tuned_f1']:.4f} "
        f"precision={recommended_by_policy.get('test_tuned_precision', recommended_by_policy.get('test_precision')):.4f} "
        f"recall={recommended_by_policy.get('test_tuned_recall', recommended_by_policy.get('test_recall')):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
