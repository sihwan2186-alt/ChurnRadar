#!/usr/bin/env python3
"""
Tune XGBoost for churn prediction with Optuna.

Examples:
    python scripts/tune_xgboost.py --trials 100 --scoring f1
    python scripts/tune_xgboost.py --trials 100 --scoring recall
    python scripts/tune_xgboost.py --trials 100 --scoring average_precision
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    import optuna
except ImportError:
    optuna = None

if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.helpers import model_path, raw_data_path, resolve_input_path
from src.utils.threshold_optimizer import evaluate_threshold, find_best_threshold

DEFAULT_CSV = raw_data_path("baza_telecom_v2.csv")
MODEL_OUT = model_path("model.joblib")

NUMERIC_FEATURES = [
    "Total_SUBs",
    "AvgMobileRevenue",
    "AvgFIXRevenue",
    "TotalRevenue",
    "ARPU",
    "Active_Ratio",
    "Not_Active_subscribers",
    "Mobile_Revenue_Ratio",
    "Inactive_Ratio",
    "Suspended_Ratio",
    "Revenue_per_Active_Sub",
    "Inactive_x_Revenue",
    "Revenue_Balance",
]
CAT_FEATURES = ["CRM_PID_Value_Segment", "EffectiveSegment"]
FEATURE_COLS = NUMERIC_FEATURES + CAT_FEATURES
TARGET = "CHURN"
THRESHOLDS = np.round(np.arange(0.10, 0.705, 0.005), 4)


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    mask = df["ARPU"].isna() & df["Total_SUBs"].gt(0)
    df.loc[mask, "ARPU"] = df.loc[mask, "TotalRevenue"] / df.loc[mask, "Total_SUBs"]

    df["Active_Ratio"] = df["Active_subscribers"] / df["Total_SUBs"].replace(0, np.nan)
    df["Active_Ratio"] = df["Active_Ratio"].fillna(0.0).clip(0.0, 1.0)

    df["Not_Active_subscribers"] = df["Not_Active_subscribers"].fillna(0.0)
    if "Suspended_subscribers" not in df.columns:
        df["Suspended_subscribers"] = 0.0
    df["Suspended_subscribers"] = df["Suspended_subscribers"].fillna(0.0)

    df["Mobile_Revenue_Ratio"] = df["AvgMobileRevenue"] / df["TotalRevenue"].replace(0, np.nan)
    df["Mobile_Revenue_Ratio"] = df["Mobile_Revenue_Ratio"].fillna(0.0).clip(0.0, 1.0)

    df["Inactive_Ratio"] = df["Not_Active_subscribers"] / df["Total_SUBs"].replace(0, np.nan)
    df["Inactive_Ratio"] = df["Inactive_Ratio"].fillna(0.0).clip(0.0, 1.0)

    df["Suspended_Ratio"] = df["Suspended_subscribers"] / df["Total_SUBs"].replace(0, np.nan)
    df["Suspended_Ratio"] = df["Suspended_Ratio"].fillna(0.0).clip(0.0, 1.0)

    df["Revenue_per_Active_Sub"] = df["TotalRevenue"] / df["Active_subscribers"].replace(0, np.nan)
    df["Revenue_per_Active_Sub"] = df["Revenue_per_Active_Sub"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["Inactive_x_Revenue"] = df["Inactive_Ratio"] * df["TotalRevenue"].fillna(0.0)

    revenue_pair = df[["AvgMobileRevenue", "AvgFIXRevenue"]].fillna(0.0)
    df["Revenue_Balance"] = revenue_pair.min(axis=1) / (revenue_pair.max(axis=1) + 1e-5)
    df["Revenue_Balance"] = df["Revenue_Balance"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    for col in CAT_FEATURES:
        df[col] = df[col].fillna("Unknown")

    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    return df.loc[valid, FEATURE_COLS].reset_index(drop=True), y[valid].astype(int).reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES),
    ])


def build_xgb_pipeline(params: dict[str, object]) -> Pipeline:
    from xgboost import XGBClassifier

    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", XGBClassifier(**params)),
    ])


def objective(
    trial,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str,
    scale_pos_weight_min: float,
    scale_pos_weight_max: float,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", scale_pos_weight_min, scale_pos_weight_max),
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(build_xgb_pipeline(params), X, y, cv=cv, scoring=scoring, n_jobs=1)
    return scores.mean()


def main() -> None:
    if optuna is None:
        raise SystemExit(
            "Optuna is not installed. Install optuna to run Bayesian tuning, "
            "or use scripts/run_baza_advanced_optimization_check.py for the local fallback search."
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=MODEL_OUT)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument(
        "--scoring",
        choices=["recall", "f1", "average_precision"],
        default="f1",
        help="Cross-validation objective for Optuna.",
    )
    parser.add_argument("--scale-pos-weight-min", type=float, default=8.0)
    parser.add_argument("--scale-pos-weight-max", type=float, default=40.0)
    args = parser.parse_args()
    args.csv = resolve_input_path(args.csv, DEFAULT_CSV)
    if not args.out.is_absolute():
        args.out = REPO_ROOT / args.out

    X, y = load_data(args.csv)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    print(f"Rows: {X.shape[0]} | Churn rate: {y.mean():.2%}")
    print(
        f"Optuna {args.trials} trials start | scoring={args.scoring} | "
        f"scale_pos_weight={args.scale_pos_weight_min:.1f}-{args.scale_pos_weight_max:.1f}\n"
    )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            args.scoring,
            args.scale_pos_weight_min,
            args.scale_pos_weight_max,
        ),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    print(f"\nBest CV {args.scoring}: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    best_params = {**study.best_params, "eval_metric": "aucpr", "random_state": 42, "n_jobs": -1}
    best_pipe = build_xgb_pipeline(best_params)
    best_pipe.fit(X_train, y_train)

    y_val_prob = best_pipe.predict_proba(X_val)[:, 1]
    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    validation_best = find_best_threshold(y_val, y_val_prob, THRESHOLDS)
    tuned = evaluate_threshold(y_test, y_prob, validation_best.threshold)
    default = evaluate_threshold(y_test, y_prob, 0.5)
    y_pred = (y_prob >= validation_best.threshold).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    print(f"\n{'=' * 50}")
    print("  XGBoost Optuna")
    print(f"{'=' * 50}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], zero_division=0))
    print(
        "  Validation-selected threshold: "
        f"{validation_best.threshold:.4f} "
        f"(val F1={validation_best.f1:.4f}, val Recall={validation_best.recall:.4f})"
    )
    print(
        "  Test tuned: "
        f"Precision={tuned.precision:.4f} | Recall={tuned.recall:.4f} | F1={tuned.f1:.4f}"
    )
    print(
        "  Test default 0.5: "
        f"Precision={default.precision:.4f} | Recall={default.recall:.4f} | F1={default.f1:.4f}"
    )
    print(f"  PR-AUC: {ap:.4f}  |  ROC-AUC: {auc:.4f}")

    final_pipe = build_xgb_pipeline(best_params)
    final_pipe.fit(X_train_val, y_train_val)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipe, args.out)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
