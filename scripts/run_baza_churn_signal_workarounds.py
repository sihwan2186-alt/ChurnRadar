#!/usr/bin/env python3
"""Run workaround tracks for missing pre-churn temporal signals on Baza.

The script separates real-world-valid experiments from leakage/simulation
demos. Label-conditioned simulated features and target-aligned external fusion
are useful for proving that the model pipeline reacts to pre-churn signals, but
they are not valid evidence of real predictive performance.
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
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


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_baza(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    return df.loc[valid].reset_index(drop=True), y.loc[valid].astype(int).reset_index(drop=True)


def make_base_splits(
    raw: pd.DataFrame,
    y: pd.Series,
    pid_aggregation_scope: str,
) -> dict[str, Any]:
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

    builder = PidSafeRiskFeatureBuilder(pid_aggregation_scope=pid_aggregation_scope)
    pid_aggregation_raw = raw if pid_aggregation_scope == "full_snapshot" else None
    X_train = builder.fit_transform(raw_train, y_train, pid_aggregation_raw=pid_aggregation_raw)
    X_val = builder.transform(raw_val)
    X_test = builder.transform(raw_test)

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "raw_train": raw_train,
        "raw_val": raw_val,
        "raw_test": raw_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "builder": builder,
    }


def label_conditioned_temporal_signals(y: pd.Series, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_values = y.to_numpy()
    n_rows = len(y)
    signals = pd.DataFrame(index=y.index)
    signals["sim_usage_drop_rate"] = np.where(
        y_values == 1,
        rng.uniform(0.40, 0.90, n_rows),
        rng.uniform(-0.10, 0.30, n_rows),
    )
    signals["sim_payment_failure_count"] = np.where(
        y_values == 1,
        rng.poisson(2.3, n_rows),
        rng.poisson(0.25, n_rows),
    )
    signals["sim_customer_inquiry_count"] = np.where(
        y_values == 1,
        rng.poisson(4.0, n_rows),
        rng.poisson(1.0, n_rows),
    )
    signals["sim_recent_complaint_spike"] = np.where(
        y_values == 1,
        rng.beta(4.0, 1.7, n_rows),
        rng.beta(1.2, 5.0, n_rows),
    )
    signals["sim_contract_or_payment_stress"] = (
        signals["sim_usage_drop_rate"].clip(lower=0)
        + 0.25 * signals["sim_payment_failure_count"]
        + 0.12 * signals["sim_customer_inquiry_count"]
    )
    return signals.astype(float)


def load_cell2cell_pool() -> tuple[pd.DataFrame, pd.Series]:
    path = raw_data_path("cell2cell_train.csv")
    if not path.exists():
        return pd.DataFrame(), pd.Series(dtype=int)
    df = pd.read_csv(path)
    y = df["Churn"].astype(str).str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)

    out = pd.DataFrame(index=df.index)
    minutes_change = numeric(df["PercChangeMinutes"])
    revenue_change = numeric(df["PercChangeRevenues"])
    out["cell_usage_drop_rate"] = (-minutes_change / 100.0).clip(-1, 1)
    out["cell_revenue_drop_rate"] = (-revenue_change / 100.0).clip(-1, 1)
    out["cell_customer_care_calls"] = numeric(df["CustomerCareCalls"])
    out["cell_retention_calls"] = numeric(df["RetentionCalls"])
    out["cell_made_retention_call"] = numeric(df["MadeCallToRetentionTeam"])
    out["cell_dropped_blocked_calls"] = numeric(df["DroppedBlockedCalls"])
    out["cell_unanswered_calls"] = numeric(df["UnansweredCalls"])
    out["cell_overage_minutes"] = numeric(df["OverageMinutes"])
    out["cell_roaming_calls"] = numeric(df["RoamingCalls"])
    out["cell_adjustments_credit_rating"] = numeric(df["AdjustmentsToCreditRating"])
    out["cell_months_in_service"] = numeric(df["MonthsInService"])
    out["cell_current_equipment_days"] = numeric(df["CurrentEquipmentDays"])
    out["cell_active_subs"] = numeric(df["ActiveSubs"])
    out["cell_unique_subs"] = numeric(df["UniqueSubs"])
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0), y


def load_ibm_pool() -> tuple[pd.DataFrame, pd.Series]:
    path = raw_data_path("ibm_telco_churn.csv")
    if not path.exists():
        return pd.DataFrame(), pd.Series(dtype=int)
    df = pd.read_csv(path)
    y = df["Churn"].map({"Yes": 1, "No": 0})
    valid = y.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)
    total_charges = numeric(df["TotalCharges"].replace(" ", np.nan))

    out = pd.DataFrame(index=df.index)
    out["ibm_monthly_charges"] = numeric(df["MonthlyCharges"])
    out["ibm_total_charges"] = total_charges
    out["ibm_tenure"] = numeric(df["tenure"])
    out["ibm_month_to_month_contract"] = df["Contract"].eq("Month-to-month").astype(float)
    out["ibm_electronic_check"] = df["PaymentMethod"].eq("Electronic check").astype(float)
    out["ibm_paperless_billing"] = df["PaperlessBilling"].eq("Yes").astype(float)
    out["ibm_no_tech_support"] = df["TechSupport"].eq("No").astype(float)
    out["ibm_no_online_security"] = df["OnlineSecurity"].eq("No").astype(float)
    out["ibm_fiber_optic"] = df["InternetService"].eq("Fiber optic").astype(float)
    out["ibm_charges_per_tenure"] = safe_divide(out["ibm_total_charges"], out["ibm_tenure"].clip(lower=1))
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0), y


def load_hf_telco_pool() -> tuple[pd.DataFrame, pd.Series]:
    paths = [
        raw_data_path("hf_telco_churn_train.csv"),
        raw_data_path("hf_telco_churn_validation.csv"),
        raw_data_path("hf_telco_churn_test.csv"),
    ]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame(), pd.Series(dtype=int)
    df = pd.concat(frames, ignore_index=True)
    y = numeric(df["Churn"])
    valid = y.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)

    out = pd.DataFrame(index=df.index)
    out["hf_churn_score"] = numeric(df["Churn Score"])
    out["hf_satisfaction_score"] = numeric(df["Satisfaction Score"])
    out["hf_monthly_charge"] = numeric(df["Monthly Charge"])
    out["hf_total_revenue"] = numeric(df["Total Revenue"])
    out["hf_total_extra_data_charges"] = numeric(df["Total Extra Data Charges"])
    out["hf_total_refunds"] = numeric(df["Total Refunds"])
    out["hf_tenure_months"] = numeric(df["Tenure in Months"])
    out["hf_avg_monthly_gb_download"] = numeric(df["Avg Monthly GB Download"])
    out["hf_number_referrals"] = numeric(df["Number of Referrals"])
    out["hf_premium_tech_support"] = numeric(df["Premium Tech Support"])
    out["hf_paperless_billing"] = numeric(df["Paperless Billing"])
    out["hf_month_to_month_contract"] = numeric(df["Contract"])
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0), y


def external_pool() -> tuple[pd.DataFrame, pd.Series]:
    pools = []
    labels = []
    for loader in [load_cell2cell_pool, load_ibm_pool, load_hf_telco_pool]:
        X_part, y_part = loader()
        if not X_part.empty:
            pools.append(X_part.reset_index(drop=True))
            labels.append(y_part.reset_index(drop=True))
    if not pools:
        return pd.DataFrame(), pd.Series(dtype=int)
    X_pool = pd.concat(pools, axis=0, ignore_index=True).fillna(0.0)
    y_pool = pd.concat(labels, axis=0, ignore_index=True).astype(int)
    X_pool = X_pool.add_prefix("fusion_")
    return X_pool, y_pool


def sampled_external_features(
    n_rows: int,
    pool_X: pd.DataFrame,
    pool_y: pd.Series,
    seed: int,
    target_y: pd.Series | None = None,
) -> pd.DataFrame:
    if pool_X.empty:
        return pd.DataFrame(index=np.arange(n_rows))
    rng = np.random.default_rng(seed)
    if target_y is None:
        chosen = rng.choice(len(pool_X), size=n_rows, replace=True)
    else:
        chosen_items = []
        all_indices = np.arange(len(pool_X))
        by_label = {
            0: np.flatnonzero(pool_y.to_numpy() == 0),
            1: np.flatnonzero(pool_y.to_numpy() == 1),
        }
        for label in target_y.to_numpy():
            source_indices = by_label.get(int(label), all_indices)
            if len(source_indices) == 0:
                source_indices = all_indices
            chosen_items.append(int(rng.choice(source_indices)))
        chosen = np.asarray(chosen_items, dtype=int)
    return pool_X.iloc[chosen].reset_index(drop=True)


def standard_candidates(scale_pos_weight: float) -> list[Candidate]:
    return [
        Candidate(
            "XGB_d2_spw",
            XGBClassifier(
                n_estimators=600,
                max_depth=2,
                learning_rate=0.03,
                min_child_weight=4,
                reg_lambda=5.0,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        Candidate(
            "XGB_d3_spw",
            XGBClassifier(
                n_estimators=600,
                max_depth=3,
                learning_rate=0.025,
                min_child_weight=6,
                reg_lambda=8.0,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        Candidate(
            "LGBM_leaf7_unbalance",
            LGBMClassifier(
                n_estimators=650,
                learning_rate=0.025,
                num_leaves=7,
                min_child_samples=35,
                reg_lambda=8.0,
                subsample=0.85,
                colsample_bytree=0.85,
                is_unbalance=True,
                objective="binary",
                verbose=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        Candidate(
            "HistGB_balanced",
            HistGradientBoostingClassifier(
                max_iter=500,
                learning_rate=0.03,
                max_leaf_nodes=15,
                l2_regularization=0.2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def cost_sensitive_candidates(scale_pos_weight: float) -> list[Candidate]:
    candidates: list[Candidate] = []
    for multiplier in [1.0, 1.5, 2.0, 3.0, 5.0]:
        spw = scale_pos_weight * multiplier
        candidates.append(
            Candidate(
                f"XGB_cost_spw_{spw:.1f}",
                XGBClassifier(
                    n_estimators=800,
                    max_depth=3,
                    learning_rate=0.025,
                    min_child_weight=4,
                    reg_lambda=6.0,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    scale_pos_weight=spw,
                    eval_metric="aucpr",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )
        )
        candidates.append(
            Candidate(
                f"LGBM_cost_spw_{spw:.1f}",
                LGBMClassifier(
                    n_estimators=800,
                    learning_rate=0.025,
                    num_leaves=15,
                    min_child_samples=25,
                    reg_lambda=6.0,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    scale_pos_weight=spw,
                    objective="binary",
                    verbose=-1,
                    random_state=RANDOM_STATE,
                ),
            )
        )
    candidates.append(
        Candidate(
            "RF_balanced_subsample_deep",
            RandomForestClassifier(
                n_estimators=600,
                max_depth=9,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
    )
    return candidates


def evaluate_candidate(
    scenario: str,
    strategy: str,
    valid_for_real_world: bool,
    notes: str,
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
    threshold, val_best_f1 = best_f1_threshold(y_val, val_scores)
    tuned = metrics_at_threshold(y_test, test_scores, threshold)
    default = metrics_at_threshold(y_test, test_scores, 0.5)

    row = {
        "scenario": scenario,
        "strategy": strategy,
        "valid_for_real_world": valid_for_real_world,
        "notes": notes,
        "model": candidate.name,
        "selected_threshold": threshold,
        "val_best_f1": val_best_f1,
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "test_average_precision": float(average_precision_score(y_test, test_scores)),
        "train_seconds": round(time.perf_counter() - started, 3),
    }
    row.update({f"test_tuned_{key}": value for key, value in tuned.items()})
    row.update({f"test_default_{key}": value for key, value in default.items()})
    for threshold_value in fixed_thresholds:
        metrics = metrics_at_threshold(y_test, test_scores, threshold_value)
        key = str(threshold_value).replace(".", "_")
        row[f"test_f1_at_{key}"] = metrics["f1"]
        row[f"test_recall_at_{key}"] = metrics["recall"]
        row[f"test_precision_at_{key}"] = metrics["precision"]
    return row


def run_scenario(
    scenario: str,
    strategy: str,
    valid_for_real_world: bool,
    notes: str,
    candidates: list[Candidate],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    fixed_thresholds: list[float],
) -> list[dict[str, Any]]:
    rows = []
    print(f"\n[{scenario}] features={X_train.shape[1]} candidates={len(candidates)}", flush=True)
    for index, candidate in enumerate(candidates, start=1):
        row = evaluate_candidate(
            scenario,
            strategy,
            valid_for_real_world,
            notes,
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
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=raw_data_path("baza_telecom_v2.csv"))
    parser.add_argument("--pid-aggregation-scope", choices=["full_snapshot", "train_only"], default="full_snapshot")
    parser.add_argument("--fixed-thresholds", type=str, default="0.25,0.3,0.35,0.4,0.5")
    parser.add_argument("--table-out", type=Path, default=result_path("baza_churn_signal_workarounds_table.csv"))
    parser.add_argument("--json-out", type=Path, default=result_path("baza_churn_signal_workarounds_summary.json"))
    args = parser.parse_args()

    started = time.perf_counter()
    args.csv = args.csv if args.csv.is_absolute() else REPO_ROOT / args.csv
    args.table_out = args.table_out if args.table_out.is_absolute() else REPO_ROOT / args.table_out
    args.json_out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)

    fixed_thresholds = [float(item.strip()) for item in args.fixed_thresholds.split(",") if item.strip()]
    raw, y = load_baza(args.csv)
    splits = make_base_splits(raw, y, args.pid_aggregation_scope)
    y_train = splits["y_train"]
    y_val = splits["y_val"]
    y_test = splits["y_test"]
    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    pid_unique = int(raw["PID"].nunique(dropna=True))
    duplicate_pid_rows = int(len(raw) - pid_unique)
    print(
        f"Baza rows={len(raw)} churn_rate={y.mean():.2%} unique_pid={pid_unique} "
        f"duplicate_pid_rows={duplicate_pid_rows} scale_pos_weight={scale_pos_weight:.2f}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    standard = standard_candidates(scale_pos_weight)

    rows.extend(
        run_scenario(
            "baseline_pid_risk",
            "static_pid_ka_zip_features",
            True,
            "Raw PID is excluded; only non-target PID aggregates and train-only KA/ZIP encodings are used.",
            standard,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            fixed_thresholds,
        )
    )

    sim_train = pd.concat(
        [X_train.reset_index(drop=True), label_conditioned_temporal_signals(y_train, RANDOM_STATE + 1)],
        axis=1,
    )
    sim_val = pd.concat(
        [X_val.reset_index(drop=True), label_conditioned_temporal_signals(y_val, RANDOM_STATE + 2)],
        axis=1,
    )
    sim_test = pd.concat(
        [X_test.reset_index(drop=True), label_conditioned_temporal_signals(y_test, RANDOM_STATE + 3)],
        axis=1,
    )
    rows.extend(
        run_scenario(
            "simulation_label_conditioned_temporal",
            "label_conditioned_simulation",
            False,
            "Demo only: simulated pre-churn signals are generated from the CHURN label, so metrics are leakage.",
            standard,
            sim_train,
            sim_val,
            sim_test,
            y_train,
            y_val,
            y_test,
            fixed_thresholds,
        )
    )

    pool_X, pool_y = external_pool()
    print(f"External fusion pool rows={len(pool_X)} features={pool_X.shape[1] if not pool_X.empty else 0}", flush=True)
    ext_all_random = sampled_external_features(len(raw), pool_X, pool_y, RANDOM_STATE + 10)
    ext_train_random = ext_all_random.iloc[splits["train_idx"]].reset_index(drop=True)
    ext_val_random = ext_all_random.iloc[splits["val_idx"]].reset_index(drop=True)
    ext_test_random = ext_all_random.iloc[splits["test_idx"]].reset_index(drop=True)
    rows.extend(
        run_scenario(
            "external_random_fusion",
            "unpaired_external_feature_sample",
            True,
            "External telco behavior columns are randomly assigned to Baza PID rows; no target alignment.",
            standard,
            pd.concat([X_train.reset_index(drop=True), ext_train_random], axis=1),
            pd.concat([X_val.reset_index(drop=True), ext_val_random], axis=1),
            pd.concat([X_test.reset_index(drop=True), ext_test_random], axis=1),
            y_train,
            y_val,
            y_test,
            fixed_thresholds,
        )
    )

    ext_all_aligned = sampled_external_features(len(raw), pool_X, pool_y, RANDOM_STATE + 20, target_y=y)
    ext_train_aligned = ext_all_aligned.iloc[splits["train_idx"]].reset_index(drop=True)
    ext_val_aligned = ext_all_aligned.iloc[splits["val_idx"]].reset_index(drop=True)
    ext_test_aligned = ext_all_aligned.iloc[splits["test_idx"]].reset_index(drop=True)
    rows.extend(
        run_scenario(
            "external_target_aligned_fusion_demo",
            "target_aligned_external_sample",
            False,
            "Demo only: external rows are sampled by matching Baza CHURN labels, so metrics are not deployable.",
            standard,
            pd.concat([X_train.reset_index(drop=True), ext_train_aligned], axis=1),
            pd.concat([X_val.reset_index(drop=True), ext_val_aligned], axis=1),
            pd.concat([X_test.reset_index(drop=True), ext_test_aligned], axis=1),
            y_train,
            y_val,
            y_test,
            fixed_thresholds,
        )
    )

    rows.extend(
        run_scenario(
            "cost_sensitive_static",
            "extreme_class_weight_and_thresholds",
            True,
            "Static features only; XGBoost/LightGBM positive-class weights are swept above the observed imbalance.",
            cost_sensitive_candidates(scale_pos_weight),
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            fixed_thresholds,
        )
    )

    table = pd.DataFrame(rows).sort_values(
        ["valid_for_real_world", "test_tuned_f1", "test_average_precision"],
        ascending=[False, False, False],
    )
    table.to_csv(args.table_out, index=False, encoding="utf-8-sig")

    valid_table = table[table["valid_for_real_world"]].copy()
    demo_table = table[~table["valid_for_real_world"]].copy()
    summary = {
        "target_f1": 0.6,
        "target_reached_real_world_valid": bool((valid_table["test_tuned_f1"] >= 0.6).any()),
        "target_reached_demo_only": bool((demo_table["test_tuned_f1"] >= 0.6).any()) if not demo_table.empty else False,
        "selection_warning": "Rows with valid_for_real_world=false use label-conditioned or target-aligned signals and are leakage demos.",
        "baza_rows": int(len(raw)),
        "baza_churn_rate": float(y.mean()),
        "unique_pid": pid_unique,
        "duplicate_pid_rows": duplicate_pid_rows,
        "pid_aggregation_scope": args.pid_aggregation_scope,
        "scale_pos_weight_base": scale_pos_weight,
        "external_pool_rows": int(len(pool_X)),
        "external_pool_features": int(pool_X.shape[1]) if not pool_X.empty else 0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "table_path": str(args.table_out),
        "best_real_world_valid_by_test_f1": valid_table.head(1).to_dict(orient="records")[0]
        if not valid_table.empty
        else None,
        "best_demo_by_test_f1": demo_table.sort_values("test_tuned_f1", ascending=False).head(1).to_dict(
            orient="records"
        )[0]
        if not demo_table.empty
        else None,
        "top_real_world_valid": valid_table.head(12).to_dict(orient="records"),
        "top_demo_only": demo_table.sort_values("test_tuned_f1", ascending=False).head(8).to_dict(orient="records"),
    }
    args.json_out.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    display_cols = [
        "scenario",
        "valid_for_real_world",
        "model",
        "val_best_f1",
        "test_tuned_f1",
        "test_tuned_precision",
        "test_tuned_recall",
        "test_recall_at_0_3",
        "test_average_precision",
    ]
    print("\n=== Baza Churn Signal Workaround Results ===", flush=True)
    print(table[display_cols].head(20).to_string(index=False), flush=True)
    print(f"\nSaved table: {args.table_out}", flush=True)
    print(f"Saved summary: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
