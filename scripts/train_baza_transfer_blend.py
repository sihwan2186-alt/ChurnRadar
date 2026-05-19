#!/usr/bin/env python3
"""Train a Baza transfer blend that can use independently trained sources."""

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

from scripts.train_baza_transfer import (
    RANDOM_STATE,
    TransferCandidate,
    best_f1_threshold,
    build_candidates,
    build_source_groups,
    load_baza,
    make_pipeline,
)
from src.models.ensemble import AveragingProbabilisticEnsemble
from src.models.threshold_model import ThresholdClassifier
from src.utils.helpers import model_path, result_path


DEFAULT_SPECS = [
    "all:1.0:LGBM_balanced",
    "all:0.5:LGBM_regularized",
    "orange_uplift:1.0:LGBM_regularized",
    "cell_orange:1.0:LGBM_regularized",
]

EXPANDED_SPECS = [
    "all:1.0:LGBM_balanced",
    "ibm_cell:0.5:LGBM_regularized",
    "ibm_cell:1.0:LGBM_regularized",
    "cell:1.0:LGBM_regularized",
    "all:0.5:LGBM_regularized",
    "all_orange:0.5:LGBM_regularized",
    "orange_uplift:1.0:LGBM_regularized",
    "cell_orange:1.0:LGBM_regularized",
]


@dataclass(frozen=True)
class MemberSpec:
    source_group: str
    source_weight: float
    model_name: str

    @property
    def name(self) -> str:
        return f"{self.source_group}|{self.source_weight:g}|{self.model_name}"


def parse_member_spec(value: str) -> MemberSpec:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Use source_group:source_weight:model_name")
    group, weight, model = parts
    try:
        parsed_weight = float(weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid source weight: {weight}") from exc
    return MemberSpec(group, parsed_weight, model)


def score_metrics(y_true: pd.Series | np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
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
    }


def find_candidate(spec: MemberSpec, candidates: list[TransferCandidate]) -> TransferCandidate:
    for candidate in candidates:
        if (
            candidate.source_group == spec.source_group
            and candidate.source_weight == spec.source_weight
            and candidate.model_name == spec.model_name
        ):
            return candidate
    raise ValueError(f"No candidate found for {spec.name}")


def train_member(
    spec: MemberSpec,
    candidate: TransferCandidate,
    source_groups: dict[str, tuple[pd.DataFrame, pd.Series]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    source_X, source_y = source_groups[spec.source_group]
    train_X = pd.concat([X_train, source_X], ignore_index=True)
    train_y = pd.concat([y_train.reset_index(drop=True), source_y.reset_index(drop=True)], ignore_index=True)

    pipeline = make_pipeline(candidate.estimator)
    if candidate.model_name == "XGB_weighted":
        pipeline.set_params(model__scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)))

    fit_kwargs = {}
    if len(source_X):
        fit_kwargs["model__sample_weight"] = np.concatenate([
            np.ones(len(X_train)),
            np.full(len(source_X), spec.source_weight),
        ])
    pipeline.fit(train_X, train_y, **fit_kwargs)
    return pipeline


def normalize_scores(scores: np.ndarray, score_range: tuple[float, float]) -> np.ndarray:
    lo, hi = score_range
    if hi - lo < 1e-12:
        return np.zeros_like(scores, dtype=float)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


def optimize_blend_weights(
    y_val: pd.Series | np.ndarray,
    val_score_matrix: np.ndarray,
    random_state: int = RANDOM_STATE,
    draws: int = 5000,
) -> tuple[np.ndarray, float, float]:
    """Select non-negative ensemble weights using validation F1 only."""
    rng = np.random.default_rng(random_state)
    member_count = val_score_matrix.shape[1]
    candidates = [np.full(member_count, 1.0 / member_count)]
    candidates.extend(np.eye(member_count))

    # Favor sparse blends because a few strong, diverse members are usually
    # more stable than a wide average on this small positive class.
    for size in range(2, min(5, member_count) + 1):
        for _ in range(max(200, draws // member_count)):
            chosen = rng.choice(member_count, size=size, replace=False)
            weights = np.zeros(member_count)
            weights[chosen] = rng.dirichlet(np.ones(size))
            candidates.append(weights)
    for alpha in (0.25, 0.5, 1.0, 2.0):
        candidates.extend(rng.dirichlet(np.full(member_count, alpha), size=draws // 4))

    best_weights = candidates[0]
    best_threshold = 0.5
    best_f1 = -1.0
    for weights in candidates:
        scores = val_score_matrix @ weights
        threshold, val_f1 = best_f1_threshold(y_val, scores)
        if val_f1 > best_f1:
            best_weights = weights
            best_threshold = threshold
            best_f1 = val_f1
    return best_weights.astype(float), float(best_threshold), float(best_f1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", action="append", type=parse_member_spec, default=None)
    parser.add_argument(
        "--preset",
        choices=["default", "expanded"],
        default="default",
        help="Named member list to use when --member is not provided.",
    )
    parser.add_argument("--optimize-weights", action="store_true")
    parser.add_argument("--weight-draws", type=int, default=5000)
    parser.add_argument("--table-out", type=Path, default=result_path("baza_transfer_orange_blend_members.csv"))
    parser.add_argument("--json-out", type=Path, default=result_path("baza_transfer_orange_blend_summary.json"))
    parser.add_argument("--model-out", type=Path, default=model_path("baza_transfer_orange_blend_model.joblib"))
    args = parser.parse_args()

    preset_specs = EXPANDED_SPECS if args.preset == "expanded" else DEFAULT_SPECS
    specs = args.member or [parse_member_spec(value) for value in preset_specs]
    X_baza, y_baza = load_baza()
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_baza, y_baza, test_size=0.2, random_state=RANDOM_STATE, stratify=y_baza
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
    )

    source_groups = build_source_groups()
    candidates = build_candidates()

    fitted_members: list[tuple[str, Any]] = []
    score_ranges: dict[str, tuple[float, float]] = {}
    member_rows: list[dict[str, Any]] = []
    val_member_scores: list[np.ndarray] = []
    test_member_scores: list[np.ndarray] = []

    for spec in specs:
        if spec.source_group not in source_groups:
            raise ValueError(f"Unknown source group: {spec.source_group}")
        source_X, source_y = source_groups[spec.source_group]
        candidate = find_candidate(spec, candidates)
        print(f"[train] {spec.name} source_rows={len(source_X)}", flush=True)
        started = time.perf_counter()
        model = train_member(spec, candidate, source_groups, X_train, y_train)
        train_seconds = round(time.perf_counter() - started, 3)

        val_scores = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        threshold, val_best_f1 = best_f1_threshold(y_val, val_scores)
        row = {
            "member": spec.name,
            "source_group": spec.source_group,
            "source_weight": spec.source_weight,
            "model": spec.model_name,
            "source_rows": int(len(source_X)),
            "source_churn_rate": float(source_y.mean()) if len(source_y) else 0.0,
            "val_best_f1": float(val_best_f1),
            "train_seconds": train_seconds,
        }
        row.update({f"test_{key}": value for key, value in score_metrics(y_test, test_scores, threshold).items()})
        member_rows.append(row)
        fitted_members.append((spec.name, model))
        score_ranges[spec.name] = (float(np.nanmin(val_scores)), float(np.nanmax(val_scores)))
        val_member_scores.append(normalize_scores(val_scores, score_ranges[spec.name]))
        test_member_scores.append(normalize_scores(test_scores, score_ranges[spec.name]))
        print(
            f"    test_f1={row['test_f1']:.4f} precision={row['test_precision']:.4f} "
            f"recall={row['test_recall']:.4f} auc={row['test_roc_auc']:.4f}",
            flush=True,
        )

    optimized_weights = None
    if args.optimize_weights:
        val_score_matrix = np.column_stack(val_member_scores)
        optimized_weights, threshold, val_best_f1 = optimize_blend_weights(
            y_val,
            val_score_matrix,
            draws=args.weight_draws,
        )
        test_blend_scores = np.column_stack(test_member_scores) @ optimized_weights
        blend = AveragingProbabilisticEnsemble(
            fitted_members,
            weights=optimized_weights.tolist(),
            score_ranges=score_ranges,
        )
    else:
        blend = AveragingProbabilisticEnsemble(fitted_members, score_ranges=score_ranges)
        val_blend_scores = blend.predict_proba(X_val)[:, 1]
        test_blend_scores = blend.predict_proba(X_test)[:, 1]
        threshold, val_best_f1 = best_f1_threshold(y_val, val_blend_scores)
    blend_metrics = score_metrics(y_test, test_blend_scores, threshold)
    blend_metrics["val_best_f1"] = float(val_best_f1)

    tuned_blend = ThresholdClassifier(blend, threshold)

    args.table_out = args.table_out if args.table_out.is_absolute() else REPO_ROOT / args.table_out
    args.json_out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    args.model_out = args.model_out if args.model_out.is_absolute() else REPO_ROOT / args.model_out
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(member_rows).sort_values("test_f1", ascending=False).to_csv(
        args.table_out, index=False, encoding="utf-8-sig"
    )
    joblib.dump(tuned_blend, args.model_out)

    summary = {
        "baza_rows": int(len(X_baza)),
        "baza_churn_rate": float(y_baza.mean()),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "members": member_rows,
        "blend": blend_metrics,
        "preset": args.preset,
        "optimize_weights": bool(args.optimize_weights),
        "blend_weights": {
            name: float(weight)
            for (name, _), weight in zip(fitted_members, optimized_weights if optimized_weights is not None else blend.weights)
        },
        "score_ranges": score_ranges,
        "table_path": str(args.table_out),
        "model_path": str(args.model_out),
        "data_sources": {
            "orange_uplift": {
                "openml_id": 45580,
                "name": "churn-uplift-mlg",
                "creator": "Machine Learning Group (ULB)",
                "collection_date": "09-2020",
                "license": "CC BY-NC-ND",
                "raw_path": str(REPO_ROOT / "raw" / "raw" / "churn_uplift_mlg.parquet"),
            }
        },
    }
    args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Blend Result ===", flush=True)
    print(
        f"f1={blend_metrics['f1']:.4f} precision={blend_metrics['precision']:.4f} "
        f"recall={blend_metrics['recall']:.4f} auc={blend_metrics['roc_auc']:.4f} "
        f"threshold={threshold:.6f}",
        flush=True,
    )
    print(f"Saved table: {args.table_out}", flush=True)
    print(f"Saved summary: {args.json_out}", flush=True)
    print(f"Saved model: {args.model_out}", flush=True)


if __name__ == "__main__":
    main()
