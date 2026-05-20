#!/usr/bin/env python3
"""Feature importance and ablation study for Baza public/geography features.

The goal is not to squeeze performance by adding more model families. It is to
test whether the feature ideas themselves matter:

- revenue and subscriber status features from the original Baza data
- billing-zone geography from GeoNames
- public demographic context from Bulgaria NSI tables

For report use, the key table is the ablation table: if removing a feature
group lowers F1/AUC, that group has evidence of contribution.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.helpers import processed_data_path, resolve_input_path, result_path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TARGET = "CHURN"


@dataclass(frozen=True)
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]
    feature_groups: dict[str, list[str]]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def present(columns: pd.Index, candidates: list[str]) -> list[str]:
    return [col for col in candidates if col in columns]


def load_features(path: Path) -> PreparedData:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)

    numeric_base = present(df.columns, [
        "Active_subscribers",
        "Not_Active_subscribers",
        "Suspended_subscribers",
        "Total_SUBs",
        "AvgMobileRevenue",
        "AvgFIXRevenue",
        "TotalRevenue",
        "ARPU",
    ])
    for col in numeric_base:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Not_Active_subscribers" in df.columns:
        df["Not_Active_subscribers"] = df["Not_Active_subscribers"].fillna(0.0)
    if "Suspended_subscribers" in df.columns:
        df["Suspended_subscribers"] = df["Suspended_subscribers"].fillna(0.0)
    if {"TotalRevenue", "Total_SUBs", "ARPU"}.issubset(df.columns):
        df["ARPU"] = df["ARPU"].fillna(safe_divide(df["TotalRevenue"], df["Total_SUBs"]))

    if {"Active_subscribers", "Total_SUBs"}.issubset(df.columns):
        df["Active_Ratio"] = safe_divide(df["Active_subscribers"], df["Total_SUBs"]).clip(0.0, 1.0)
    if {"Not_Active_subscribers", "Total_SUBs"}.issubset(df.columns):
        df["Inactive_Ratio"] = safe_divide(df["Not_Active_subscribers"], df["Total_SUBs"]).clip(0.0, 1.0)
        df["Has_Inactive"] = df["Not_Active_subscribers"].gt(0).astype(float)
    if {"Suspended_subscribers", "Total_SUBs"}.issubset(df.columns):
        df["Suspended_Ratio"] = safe_divide(df["Suspended_subscribers"], df["Total_SUBs"]).clip(0.0, 1.0)
        df["Has_Suspended"] = df["Suspended_subscribers"].gt(0).astype(float)
    if {"Not_Active_subscribers", "Suspended_subscribers", "Total_SUBs"}.issubset(df.columns):
        dormant = df["Not_Active_subscribers"] + df["Suspended_subscribers"]
        df["Dormant_Ratio"] = safe_divide(dormant, df["Total_SUBs"]).clip(0.0, 1.0)
    if {"AvgMobileRevenue", "TotalRevenue"}.issubset(df.columns):
        df["Mobile_Revenue_Ratio"] = safe_divide(df["AvgMobileRevenue"], df["TotalRevenue"]).clip(0.0, 1.0)
    if {"AvgFIXRevenue", "TotalRevenue"}.issubset(df.columns):
        df["Fixed_Revenue_Ratio"] = safe_divide(df["AvgFIXRevenue"], df["TotalRevenue"]).clip(0.0, 1.0)
    if {"TotalRevenue", "Total_SUBs"}.issubset(df.columns):
        df["Revenue_Per_Sub"] = safe_divide(df["TotalRevenue"], df["Total_SUBs"])
    if {"TotalRevenue", "Active_subscribers"}.issubset(df.columns):
        df["Revenue_Per_Active"] = safe_divide(df["TotalRevenue"], df["Active_subscribers"])
    if {"AvgMobileRevenue", "AvgFIXRevenue"}.issubset(df.columns):
        df["Mobile_To_Fixed_Ratio"] = safe_divide(df["AvgMobileRevenue"], df["AvgFIXRevenue"])
        df["Fixed_To_Mobile_Ratio"] = safe_divide(df["AvgFIXRevenue"], df["AvgMobileRevenue"])
        df["Mobile_Only"] = (df["AvgMobileRevenue"].gt(0) & df["AvgFIXRevenue"].eq(0)).astype(float)
        df["Fixed_Only"] = (df["AvgFIXRevenue"].gt(0) & df["AvgMobileRevenue"].eq(0)).astype(float)

    revenue_size = present(df.columns, [
        "Total_SUBs",
        "AvgMobileRevenue",
        "AvgFIXRevenue",
        "TotalRevenue",
        "ARPU",
        "Mobile_Revenue_Ratio",
        "Fixed_Revenue_Ratio",
        "Revenue_Per_Sub",
        "Revenue_Per_Active",
        "Mobile_To_Fixed_Ratio",
        "Fixed_To_Mobile_Ratio",
        "Mobile_Only",
        "Fixed_Only",
    ])
    subscriber_status = present(df.columns, [
        "Active_subscribers",
        "Not_Active_subscribers",
        "Suspended_subscribers",
        "Active_Ratio",
        "Inactive_Ratio",
        "Suspended_Ratio",
        "Dormant_Ratio",
        "Has_Inactive",
        "Has_Suspended",
    ])
    customer_profile = present(df.columns, [
        "CRM_PID_Value_Segment",
        "EffectiveSegment",
        "Billing_ZIP",
        "KA_name",
    ])
    geo_billing_zone = present(df.columns, [
        "Billing_ZIP_norm",
        "bg_zip_found",
        "bg_place_count",
        "bg_place_name",
        "bg_admin1_name",
        "bg_admin1_code",
        "bg_admin2_name",
        "bg_admin2_code",
        "bg_admin3_name",
        "bg_admin3_code",
        "bg_latitude",
        "bg_longitude",
        "bg_geo_accuracy",
        "bg_distance_sofia_km",
        "bg_distance_plovdiv_km",
        "bg_distance_varna_km",
        "bg_distance_burgas_km",
        "bg_min_big_city_distance_km",
    ])
    public_demographics = [
        col
        for col in df.columns
        if col.startswith(("nsi_", "district_nsi_", "municipality_nsi_"))
    ]

    feature_groups = {
        "revenue_size": revenue_size,
        "subscriber_status": subscriber_status,
        "customer_profile": customer_profile,
        "geo_billing_zone": geo_billing_zone,
        "public_demographics": public_demographics,
    }

    all_features: list[str] = []
    for group_features in feature_groups.values():
        all_features.extend(group_features)
    all_features = list(dict.fromkeys(all_features))

    numeric_features: list[str] = []
    categorical_features: list[str] = []
    for col in all_features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            categorical_features.append(col)
            df[col] = df[col].fillna("Unknown").astype(str)

    return PreparedData(
        X=df[numeric_features + categorical_features].copy(),
        y=y,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        feature_groups=feature_groups,
    )


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_features))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            categorical_features,
        ))
    return ColumnTransformer(transformers)


def make_model(kind: str) -> Any:
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    if kind == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    raise ValueError(kind)


def make_pipeline(kind: str, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(numeric_features, categorical_features)),
        ("model", make_model(kind)),
    ])


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


def subset_columns(
    selected_features: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[list[str], list[str]]:
    selected = set(selected_features)
    return (
        [col for col in numeric_features if col in selected],
        [col for col in categorical_features if col in selected],
    )


def evaluate_selected_features(
    label: str,
    model_kind: str,
    selected_features: list[str],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[dict[str, Any], Pipeline]:
    selected_numeric, selected_categorical = subset_columns(selected_features, numeric_features, categorical_features)
    pipeline = make_pipeline(model_kind, selected_numeric, selected_categorical)
    started = time.perf_counter()
    pipeline.fit(X_train[selected_features], y_train)
    val_scores = model_scores(pipeline, X_val[selected_features])
    test_scores = model_scores(pipeline, X_test[selected_features])
    threshold, val_best_f1 = best_f1_threshold(y_val, val_scores)
    oracle_threshold, oracle_f1 = best_f1_threshold(y_test, test_scores)
    tuned_pred = (test_scores >= threshold).astype(int)
    default_pred = pipeline.predict(X_test[selected_features])

    row: dict[str, Any] = {
        "ablation": label,
        "model": model_kind,
        "feature_count": int(len(selected_features)),
        "numeric_feature_count": int(len(selected_numeric)),
        "categorical_feature_count": int(len(selected_categorical)),
        "removed_feature_count": int(X_train.shape[1] - len(selected_features)),
        "threshold": threshold,
        "val_best_f1": val_best_f1,
        "test_oracle_threshold": oracle_threshold,
        "test_oracle_f1": oracle_f1,
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "test_average_precision": float(average_precision_score(y_test, test_scores)),
        "train_seconds": round(time.perf_counter() - started, 3),
    }
    row.update({f"test_tuned_{key}": value for key, value in metrics_from_prediction(y_test, tuned_pred).items()})
    row.update({f"test_default_{key}": value for key, value in metrics_from_prediction(y_test, default_pred).items()})
    return row, pipeline


def importance_rows(
    model_label: str,
    pipeline: Pipeline,
    feature_names: list[str],
    feature_groups: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    importances = pipeline.named_steps["model"].feature_importances_
    feature_to_group = {
        feature: group
        for group, group_features in feature_groups.items()
        for feature in group_features
    }
    rows = []
    for feature, importance in zip(feature_names, importances):
        rows.append({
            "model": model_label,
            "feature": feature,
            "group": feature_to_group.get(feature, "other"),
            "importance": float(importance),
        })

    table = pd.DataFrame(rows)
    group_rows = (
        table.groupby(["model", "group"], as_index=False)["importance"]
        .sum()
        .sort_values(["model", "importance"], ascending=[True, False])
        .to_dict(orient="records")
    )
    rows = table.sort_values(["model", "importance"], ascending=[True, False]).to_dict(orient="records")
    return rows, group_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=processed_data_path("baza_telecom_v2_bulgaria_public.csv"))
    parser.add_argument("--importance-out", type=Path, default=result_path("baza_public_feature_importance.csv"))
    parser.add_argument("--group-importance-out", type=Path, default=result_path("baza_public_feature_group_importance.csv"))
    parser.add_argument("--ablation-out", type=Path, default=result_path("baza_public_feature_ablation.csv"))
    parser.add_argument("--json-out", type=Path, default=result_path("baza_public_feature_ablation_summary.json"))
    args = parser.parse_args()

    args.csv = resolve_input_path(args.csv, processed_data_path("baza_telecom_v2_bulgaria_public.csv"))
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    prepared = load_features(args.csv)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        prepared.X, prepared.y, test_size=0.2, random_state=RANDOM_STATE, stratify=prepared.y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
    )
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    all_features = prepared.numeric_features + prepared.categorical_features
    started = time.perf_counter()
    print(
        f"Baza public features rows={len(prepared.X)} features={len(all_features)} "
        f"churn_rate={prepared.y.mean():.2%}",
        flush=True,
    )

    baseline_row, baseline_rf = evaluate_selected_features(
        "all_features",
        "random_forest",
        all_features,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        prepared.numeric_features,
        prepared.categorical_features,
    )
    print(
        f"RF baseline: f1={baseline_row['test_tuned_f1']:.4f} "
        f"auc={baseline_row['test_roc_auc']:.4f}",
        flush=True,
    )

    tree_row, baseline_tree = evaluate_selected_features(
        "all_features",
        "decision_tree",
        all_features,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        prepared.numeric_features,
        prepared.categorical_features,
    )
    print(
        f"DecisionTree baseline: f1={tree_row['test_tuned_f1']:.4f} "
        f"auc={tree_row['test_roc_auc']:.4f}",
        flush=True,
    )

    feature_importance, group_importance = importance_rows(
        "random_forest",
        baseline_rf,
        all_features,
        prepared.feature_groups,
    )
    tree_importance, tree_group_importance = importance_rows(
        "decision_tree",
        baseline_tree,
        all_features,
        prepared.feature_groups,
    )
    feature_importance.extend(tree_importance)
    group_importance.extend(tree_group_importance)

    rf_importance_table = pd.DataFrame(feature_importance)
    top_rf_features = (
        rf_importance_table[rf_importance_table["model"] == "random_forest"]
        .sort_values("importance", ascending=False)["feature"]
        .head(5)
        .tolist()
    )

    ablation_rows = [baseline_row, tree_row]
    baseline_f1 = float(baseline_row["test_tuned_f1"])
    baseline_auc = float(baseline_row["test_roc_auc"])
    for group, group_features in prepared.feature_groups.items():
        remove = set(group_features)
        selected = [feature for feature in all_features if feature not in remove]
        if not selected:
            continue
        row, _ = evaluate_selected_features(
            f"remove_group:{group}",
            "random_forest",
            selected,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            prepared.numeric_features,
            prepared.categorical_features,
        )
        row["removed_group"] = group
        row["removed_features"] = ";".join(group_features)
        row["delta_f1_vs_rf_baseline"] = float(row["test_tuned_f1"] - baseline_f1)
        row["delta_auc_vs_rf_baseline"] = float(row["test_roc_auc"] - baseline_auc)
        ablation_rows.append(row)
        print(
            f"remove {group}: f1={row['test_tuned_f1']:.4f} "
            f"delta={row['delta_f1_vs_rf_baseline']:+.4f}",
            flush=True,
        )

    for top_n in [1, 3, 5]:
        remove = set(top_rf_features[:top_n])
        selected = [feature for feature in all_features if feature not in remove]
        row, _ = evaluate_selected_features(
            f"remove_top{top_n}_rf_features",
            "random_forest",
            selected,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            prepared.numeric_features,
            prepared.categorical_features,
        )
        row["removed_group"] = f"top{top_n}_rf_features"
        row["removed_features"] = ";".join(top_rf_features[:top_n])
        row["delta_f1_vs_rf_baseline"] = float(row["test_tuned_f1"] - baseline_f1)
        row["delta_auc_vs_rf_baseline"] = float(row["test_roc_auc"] - baseline_auc)
        ablation_rows.append(row)
        print(
            f"remove top {top_n}: f1={row['test_tuned_f1']:.4f} "
            f"delta={row['delta_f1_vs_rf_baseline']:+.4f}",
            flush=True,
        )

    args.importance_out = args.importance_out if args.importance_out.is_absolute() else REPO_ROOT / args.importance_out
    args.group_importance_out = (
        args.group_importance_out if args.group_importance_out.is_absolute() else REPO_ROOT / args.group_importance_out
    )
    args.ablation_out = args.ablation_out if args.ablation_out.is_absolute() else REPO_ROOT / args.ablation_out
    args.json_out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    for path in [args.importance_out, args.group_importance_out, args.ablation_out, args.json_out]:
        path.parent.mkdir(parents=True, exist_ok=True)

    importance_table = pd.DataFrame(feature_importance).sort_values(["model", "importance"], ascending=[True, False])
    group_table = pd.DataFrame(group_importance).sort_values(["model", "importance"], ascending=[True, False])
    ablation_table = pd.DataFrame(ablation_rows)
    ablation_table = ablation_table.sort_values(
        ["model", "test_tuned_f1", "test_roc_auc"], ascending=[True, False, False]
    )

    importance_table.to_csv(args.importance_out, index=False, encoding="utf-8-sig")
    group_table.to_csv(args.group_importance_out, index=False, encoding="utf-8-sig")
    ablation_table.to_csv(args.ablation_out, index=False, encoding="utf-8-sig")

    best_drop = (
        ablation_table[ablation_table["ablation"].astype(str).str.startswith("remove_")]
        .sort_values("delta_f1_vs_rf_baseline")
        .head(1)
        .to_dict(orient="records")
    )
    summary = {
        "csv": str(args.csv),
        "rows": int(len(prepared.X)),
        "features": int(len(all_features)),
        "churn_rate": float(prepared.y.mean()),
        "class_counts": {
            "no_churn": int((prepared.y == 0).sum()),
            "churn": int((prepared.y == 1).sum()),
        },
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "feature_groups": {key: value for key, value in prepared.feature_groups.items()},
        "random_forest_baseline": baseline_row,
        "decision_tree_baseline": tree_row,
        "top_random_forest_features": top_rf_features,
        "largest_f1_drop_ablation": best_drop[0] if best_drop else {},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "importance_path": str(args.importance_out),
        "group_importance_path": str(args.group_importance_out),
        "ablation_path": str(args.ablation_out),
    }
    args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Top random forest importances ===", flush=True)
    print(
        importance_table[importance_table["model"] == "random_forest"]
        .head(12)
        .to_string(index=False),
        flush=True,
    )
    print("\n=== Ablation summary ===", flush=True)
    display_cols = [
        "ablation",
        "model",
        "feature_count",
        "test_tuned_f1",
        "test_tuned_precision",
        "test_tuned_recall",
        "test_roc_auc",
        "delta_f1_vs_rf_baseline",
    ]
    for col in display_cols:
        if col not in ablation_table.columns:
            ablation_table[col] = np.nan
    print(ablation_table[display_cols].to_string(index=False), flush=True)
    print(f"\nSaved importance: {args.importance_out}", flush=True)
    print(f"Saved group importance: {args.group_importance_out}", flush=True)
    print(f"Saved ablation: {args.ablation_out}", flush=True)
    print(f"Saved summary: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
