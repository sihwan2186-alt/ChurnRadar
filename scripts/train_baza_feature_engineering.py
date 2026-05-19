#!/usr/bin/env python3
"""Baza-only feature engineering, model sweep, and ensemble benchmark.

This script intentionally avoids auxiliary datasets. It uses only
``baza_telecom_v2.csv`` and tests whether deeper feature engineering, leakage
safe target encoding, imbalance handling, threshold tuning, and score blending
can improve the held-out Baza F1 score.
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
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import (
    BalancedBaggingClassifier,
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.threshold_model import ThresholdClassifier
from src.utils.helpers import model_path, raw_data_path, result_path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TARGET = "CHURN"


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: Any
    family: str
    sampler: str | None = None
    scaler: str = "standard"


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series:
    numerator = pd.Series(numerator)
    denominator = pd.Series(denominator).replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def logit_clip(values: pd.Series) -> pd.Series:
    clipped = values.clip(1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


class BazaFeatureBuilder:
    def __init__(self, mode: str, smoothing: float = 25.0, folds: int = 5) -> None:
        self.mode = mode
        self.smoothing = smoothing
        self.folds = folds
        self.global_mean = 0.0
        self.ordinal_maps: dict[str, dict[str, int]] = {}
        self.freq_maps: dict[str, dict[str, float]] = {}
        self.target_maps: dict[str, dict[str, float]] = {}
        self.woe_maps: dict[str, dict[str, float]] = {}
        self.feature_columns: list[str] = []

    @property
    def uses_target_encoding(self) -> bool:
        return self.mode in {"encoded", "encoded_no_ka", "encoded_compact"}

    @property
    def category_columns(self) -> list[str]:
        base = [
            "CRM_PID_Value_Segment",
            "EffectiveSegment",
            "Billing_ZIP_str",
            "KA_name",
            "crm_effective",
            "crm_ka",
            "effective_ka",
            "ka_zip",
            "segment_zip",
            "crm_effective_ka",
        ]
        if self.mode == "encoded_no_ka":
            return [col for col in base if "KA" not in col and "ka_" not in col and "_ka" not in col]
        if self.mode == "encoded_compact":
            return ["CRM_PID_Value_Segment", "EffectiveSegment", "Billing_ZIP_str", "crm_effective", "segment_zip"]
        if self.mode == "core":
            return ["CRM_PID_Value_Segment", "EffectiveSegment"]
        return base

    def _prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df.columns = df.columns.str.strip()

        numeric_cols = [
            "Billing_ZIP",
            "Active_subscribers",
            "Not_Active_subscribers",
            "Suspended_subscribers",
            "Total_SUBs",
            "AvgMobileRevenue",
            "AvgFIXRevenue",
            "TotalRevenue",
            "ARPU",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Not_Active_missing"] = df["Not_Active_subscribers"].isna().astype(float)
        df["Suspended_missing"] = df["Suspended_subscribers"].isna().astype(float)
        df["ARPU_missing"] = df["ARPU"].isna().astype(float)
        df["Billing_ZIP_missing"] = df["Billing_ZIP"].isna().astype(float)

        df["Not_Active_subscribers"] = df["Not_Active_subscribers"].fillna(0.0)
        df["Suspended_subscribers"] = df["Suspended_subscribers"].fillna(0.0)
        arpu_calc = safe_divide(df["TotalRevenue"], df["Total_SUBs"])
        df["ARPU"] = df["ARPU"].fillna(arpu_calc)

        for col in ["CRM_PID_Value_Segment", "EffectiveSegment", "KA_name"]:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df["Billing_ZIP_str"] = df["Billing_ZIP"].fillna(-1).round().astype(int).astype(str)
        df["zip_prefix_1"] = df["Billing_ZIP_str"].str[:1].replace("", "Unknown")
        df["zip_prefix_2"] = df["Billing_ZIP_str"].str[:2].replace("", "Unknown")
        df["crm_effective"] = df["CRM_PID_Value_Segment"] + "|" + df["EffectiveSegment"]
        df["crm_ka"] = df["CRM_PID_Value_Segment"] + "|" + df["KA_name"]
        df["effective_ka"] = df["EffectiveSegment"] + "|" + df["KA_name"]
        df["ka_zip"] = df["KA_name"] + "|" + df["Billing_ZIP_str"]
        df["segment_zip"] = df["CRM_PID_Value_Segment"] + "|" + df["Billing_ZIP_str"]
        df["crm_effective_ka"] = df["crm_effective"] + "|" + df["KA_name"]
        return df

    def _numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        active = df["Active_subscribers"]
        inactive = df["Not_Active_subscribers"]
        suspended = df["Suspended_subscribers"]
        total_subs = df["Total_SUBs"]
        mobile = df["AvgMobileRevenue"]
        fixed = df["AvgFIXRevenue"]
        revenue = df["TotalRevenue"]
        arpu = df["ARPU"]
        dormant = inactive + suspended
        revenue_per_sub = safe_divide(revenue, total_subs)
        revenue_per_active = safe_divide(revenue, active)
        active_ratio = safe_divide(active, total_subs).clip(0, 1)
        inactive_ratio = safe_divide(inactive, total_subs).clip(0, 1)
        suspended_ratio = safe_divide(suspended, total_subs).clip(0, 1)

        features = pd.DataFrame(index=df.index)
        base_cols = {
            "active_subscribers": active,
            "not_active_subscribers": inactive,
            "suspended_subscribers": suspended,
            "total_subs": total_subs,
            "avg_mobile_revenue": mobile,
            "avg_fix_revenue": fixed,
            "total_revenue": revenue,
            "arpu": arpu,
            "billing_zip": df["Billing_ZIP"],
            "not_active_missing": df["Not_Active_missing"],
            "suspended_missing": df["Suspended_missing"],
            "arpu_missing": df["ARPU_missing"],
            "billing_zip_missing": df["Billing_ZIP_missing"],
        }
        for name, values in base_cols.items():
            features[name] = values

        if self.mode == "core":
            features["active_ratio"] = active_ratio
            features["inactive_ratio"] = inactive_ratio
            features["mobile_revenue_ratio"] = safe_divide(mobile, revenue).clip(0, 1)
            features["fixed_revenue_ratio"] = safe_divide(fixed, revenue).clip(0, 1)
            return features

        derived = {
            "dormant_subscribers": dormant,
            "active_ratio": active_ratio,
            "inactive_ratio": inactive_ratio,
            "suspended_ratio": suspended_ratio,
            "dormant_ratio": safe_divide(dormant, total_subs).clip(0, 1),
            "mobile_revenue_ratio": safe_divide(mobile, revenue).clip(0, 1),
            "fixed_revenue_ratio": safe_divide(fixed, revenue).clip(0, 1),
            "mobile_to_fixed_ratio": safe_divide(mobile, fixed),
            "fixed_to_mobile_ratio": safe_divide(fixed, mobile),
            "revenue_per_sub": revenue_per_sub,
            "revenue_per_active": revenue_per_active,
            "mobile_per_sub": safe_divide(mobile, total_subs),
            "fixed_per_sub": safe_divide(fixed, total_subs),
            "mobile_per_active": safe_divide(mobile, active),
            "fixed_per_active": safe_divide(fixed, active),
            "arpu_minus_revenue_per_sub": arpu - revenue_per_sub,
            "arpu_ratio_to_calc": safe_divide(arpu, revenue_per_sub),
            "active_minus_dormant": active - dormant,
            "inactive_minus_suspended": inactive - suspended,
            "active_to_dormant_ratio": safe_divide(active, dormant),
            "subs_minus_active": total_subs - active,
            "has_inactive": inactive.gt(0).astype(float),
            "has_suspended": suspended.gt(0).astype(float),
            "has_dormant": dormant.gt(0).astype(float),
            "multi_subscriber": total_subs.gt(1).astype(float),
            "large_account": total_subs.ge(10).astype(float),
            "zero_fixed_revenue": fixed.eq(0).astype(float),
            "zero_mobile_revenue": mobile.eq(0).astype(float),
            "mobile_only": (mobile.gt(0) & fixed.eq(0)).astype(float),
            "fixed_only": (fixed.gt(0) & mobile.eq(0)).astype(float),
            "revenue_zero": revenue.eq(0).astype(float),
            "zip_numeric_bucket": (df["Billing_ZIP"].fillna(-1) // 100).astype(float),
        }
        for name, values in derived.items():
            features[name] = values

        for col in [
            "active_subscribers",
            "not_active_subscribers",
            "suspended_subscribers",
            "dormant_subscribers",
            "total_subs",
            "avg_mobile_revenue",
            "avg_fix_revenue",
            "total_revenue",
            "arpu",
            "revenue_per_sub",
            "revenue_per_active",
        ]:
            values = features[col].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0)
            features[f"log1p_{col}"] = np.log1p(values)
            features[f"sqrt_{col}"] = np.sqrt(values)

        return features

    def _fit_category_maps(self, prepared: pd.DataFrame, y: pd.Series) -> None:
        self.global_mean = float(y.mean())
        for col in self.category_columns:
            values = prepared[col].astype(str)
            uniques = pd.Index(values.unique())
            self.ordinal_maps[col] = {value: idx for idx, value in enumerate(uniques)}
            self.freq_maps[col] = values.value_counts(normalize=True).to_dict()
            stats = pd.DataFrame({"cat": values, "y": y.values}).groupby("cat")["y"].agg(["sum", "count"])
            target = (stats["sum"] + self.smoothing * self.global_mean) / (stats["count"] + self.smoothing)
            self.target_maps[col] = target.to_dict()
            self.woe_maps[col] = logit_clip(target).to_dict()

    def _oof_target_encoding(self, prepared: pd.DataFrame, y: pd.Series, col: str, kind: str) -> pd.Series:
        encoded = pd.Series(self.global_mean, index=prepared.index, dtype=float)
        splitter = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=RANDOM_STATE)
        for fold_train_idx, fold_valid_idx in splitter.split(prepared, y):
            train_values = prepared.iloc[fold_train_idx][col].astype(str)
            train_y = y.iloc[fold_train_idx]
            stats = pd.DataFrame({"cat": train_values, "y": train_y.values}).groupby("cat")["y"].agg(["sum", "count"])
            target = (stats["sum"] + self.smoothing * self.global_mean) / (stats["count"] + self.smoothing)
            mapping = logit_clip(target).to_dict() if kind == "woe" else target.to_dict()
            default = float(logit_clip(pd.Series([self.global_mean])).iloc[0]) if kind == "woe" else self.global_mean
            encoded.iloc[fold_valid_idx] = prepared.iloc[fold_valid_idx][col].astype(str).map(mapping).fillna(default)
        return encoded

    def fit(self, raw_train: pd.DataFrame, y_train: pd.Series) -> "BazaFeatureBuilder":
        prepared = self._prepare(raw_train)
        self._fit_category_maps(prepared, y_train.reset_index(drop=True))
        _ = self.transform(raw_train, y_for_oof=y_train)
        return self

    def transform(self, raw: pd.DataFrame, y_for_oof: pd.Series | None = None) -> pd.DataFrame:
        prepared = self._prepare(raw)
        features = self._numeric_features(prepared)

        if self.mode != "core":
            for col in self.category_columns:
                values = prepared[col].astype(str)
                features[f"{col}_ord"] = values.map(self.ordinal_maps[col]).fillna(-1).astype(float)
                features[f"{col}_freq"] = values.map(self.freq_maps[col]).fillna(0.0).astype(float)

        if self.uses_target_encoding:
            for col in self.category_columns:
                values = prepared[col].astype(str)
                if y_for_oof is None:
                    features[f"{col}_te"] = values.map(self.target_maps[col]).fillna(self.global_mean).astype(float)
                    default_woe = float(logit_clip(pd.Series([self.global_mean])).iloc[0])
                    features[f"{col}_woe"] = values.map(self.woe_maps[col]).fillna(default_woe).astype(float)
                else:
                    y_for_oof = y_for_oof.reset_index(drop=True)
                    features[f"{col}_te"] = self._oof_target_encoding(prepared, y_for_oof, col, "target")
                    features[f"{col}_woe"] = self._oof_target_encoding(prepared, y_for_oof, col, "woe")

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0.0).astype(float)
        if not self.feature_columns:
            self.feature_columns = list(features.columns)
        return features.reindex(columns=self.feature_columns, fill_value=0.0)

    def fit_transform(self, raw_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        self.fit(raw_train, y_train)
        return self.transform(raw_train, y_for_oof=y_train)


def load_baza(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    return df.loc[valid].reset_index(drop=True), y.loc[valid].astype(int).reset_index(drop=True)


def make_sampler(name: str | None):
    if name is None:
        return None
    if name == "ros":
        return RandomOverSampler(random_state=RANDOM_STATE)
    if name == "rus":
        return RandomUnderSampler(random_state=RANDOM_STATE)
    if name == "smote":
        return SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    if name == "borderline_smote":
        return BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    if name == "adasyn":
        return ADASYN(random_state=RANDOM_STATE, n_neighbors=5)
    if name == "smoteenn":
        return SMOTEENN(random_state=RANDOM_STATE)
    if name == "smotetomek":
        return SMOTETomek(random_state=RANDOM_STATE)
    raise ValueError(name)


def make_pipeline(candidate: Candidate):
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if candidate.scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif candidate.scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    elif candidate.scaler == "none":
        pass
    else:
        raise ValueError(candidate.scaler)

    sampler = make_sampler(candidate.sampler)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", candidate.estimator))
    return (ImbPipeline if sampler is not None else Pipeline)(steps)


def build_candidates(scale_pos_weight: float) -> list[Candidate]:
    lgbm_base = {
        "learning_rate": 0.03,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    xgb_base = {
        "learning_rate": 0.03,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    cat_base = {
        "iterations": 700,
        "learning_rate": 0.03,
        "auto_class_weights": "Balanced",
        "random_seed": RANDOM_STATE,
        "verbose": 0,
        "allow_writing_files": False,
    }

    return [
        Candidate("LogReg_L2_balanced", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=RANDOM_STATE), "linear"),
        Candidate("LogReg_L1_balanced", LogisticRegression(max_iter=1200, penalty="l1", solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE), "linear"),
        Candidate("LogReg_L2_C03_balanced", LogisticRegression(max_iter=2500, C=0.3, class_weight="balanced", random_state=RANDOM_STATE), "linear"),
        Candidate("LogReg_L2_C3_balanced", LogisticRegression(max_iter=2500, C=3.0, class_weight="balanced", random_state=RANDOM_STATE), "linear"),
        Candidate("LogReg_ROS", LogisticRegression(max_iter=4000, random_state=RANDOM_STATE), "linear_resampled", "ros"),
        Candidate("LogReg_SMOTE", LogisticRegression(max_iter=4000, random_state=RANDOM_STATE), "linear_resampled", "smote"),
        Candidate("SGD_log_balanced", SGDClassifier(loss="log_loss", max_iter=4000, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), "linear"),
        Candidate("Ridge_balanced_calibrated", CalibratedClassifierCV(RidgeClassifier(class_weight="balanced", random_state=RANDOM_STATE), cv=3), "linear"),
        Candidate("LinearSVC_calibrated", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, dual="auto", max_iter=6000), cv=3), "svm"),
        Candidate("SVC_rbf_balanced", SVC(class_weight="balanced", kernel="rbf", C=1.0, gamma="scale", probability=False, random_state=RANDOM_STATE), "svm"),
        Candidate("KNN_15_distance", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1), "neighbors", scaler="robust"),
        Candidate("KNN_35_distance", KNeighborsClassifier(n_neighbors=35, weights="distance", n_jobs=-1), "neighbors", scaler="robust"),
        Candidate("GaussianNB", GaussianNB(), "bayes", scaler="none"),
        Candidate("LDA_shrinkage", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "discriminant"),
        Candidate("MLP_64_32", MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=350, random_state=RANDOM_STATE), "neural_net"),
        Candidate("DecisionTree_balanced", DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=12, random_state=RANDOM_STATE), "tree", scaler="none"),
        Candidate("RandomForest_d5_balanced", RandomForestClassifier(n_estimators=500, max_depth=5, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), "forest", scaler="none"),
        Candidate("RandomForest_d8_balanced", RandomForestClassifier(n_estimators=500, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), "forest", scaler="none"),
        Candidate("RandomForest_leaf5_balanced", RandomForestClassifier(n_estimators=600, min_samples_leaf=5, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1), "forest", scaler="none"),
        Candidate("ExtraTrees_balanced", ExtraTreesClassifier(n_estimators=700, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), "forest", scaler="none"),
        Candidate("GradientBoosting_d2", GradientBoostingClassifier(n_estimators=450, learning_rate=0.03, max_depth=2, random_state=RANDOM_STATE), "boosting", scaler="none"),
        Candidate("GradientBoosting_d3", GradientBoostingClassifier(n_estimators=350, learning_rate=0.03, max_depth=3, random_state=RANDOM_STATE), "boosting", scaler="none"),
        Candidate("HistGB_balanced_l2", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, class_weight="balanced", l2_regularization=0.2, random_state=RANDOM_STATE), "boosting", scaler="none"),
        Candidate("HistGB_balanced_leaf15", HistGradientBoostingClassifier(max_iter=600, learning_rate=0.02, max_leaf_nodes=15, class_weight="balanced", l2_regularization=0.5, random_state=RANDOM_STATE), "boosting", scaler="none"),
        Candidate("AdaBoost_tree_d1", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE), n_estimators=500, learning_rate=0.03, random_state=RANDOM_STATE), "boosting", scaler="none"),
        Candidate("AdaBoost_tree_d2", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE), n_estimators=400, learning_rate=0.03, random_state=RANDOM_STATE), "boosting", scaler="none"),
        Candidate("LightGBM_leaf7", LGBMClassifier(n_estimators=800, num_leaves=7, min_child_samples=35, reg_lambda=4.0, **lgbm_base), "boosting", scaler="none"),
        Candidate("LightGBM_leaf15", LGBMClassifier(n_estimators=650, num_leaves=15, min_child_samples=25, reg_lambda=2.0, **lgbm_base), "boosting", scaler="none"),
        Candidate("LightGBM_leaf31", LGBMClassifier(n_estimators=550, num_leaves=31, min_child_samples=20, reg_lambda=1.0, **lgbm_base), "boosting", scaler="none"),
        Candidate("LightGBM_goss", LGBMClassifier(n_estimators=650, num_leaves=15, min_child_samples=25, boosting_type="goss", reg_lambda=2.0, **lgbm_base), "boosting", scaler="none"),
        Candidate("XGBoost_d1", XGBClassifier(n_estimators=750, max_depth=1, min_child_weight=5, reg_lambda=5.0, **xgb_base), "boosting", scaler="none"),
        Candidate("XGBoost_d2", XGBClassifier(n_estimators=650, max_depth=2, min_child_weight=4, reg_lambda=4.0, **xgb_base), "boosting", scaler="none"),
        Candidate("XGBoost_d3", XGBClassifier(n_estimators=550, max_depth=3, min_child_weight=3, reg_lambda=3.0, subsample=0.9, colsample_bytree=0.9, **xgb_base), "boosting", scaler="none"),
        Candidate("XGBoost_d4", XGBClassifier(n_estimators=450, max_depth=4, min_child_weight=3, reg_lambda=4.0, subsample=0.85, colsample_bytree=0.85, **xgb_base), "boosting", scaler="none"),
        Candidate("CatBoost_d4", CatBoostClassifier(depth=4, l2_leaf_reg=8.0, **cat_base), "boosting", scaler="none"),
        Candidate("CatBoost_d6", CatBoostClassifier(depth=6, l2_leaf_reg=10.0, **cat_base), "boosting", scaler="none"),
        Candidate("BalancedRF_d5", BalancedRandomForestClassifier(n_estimators=500, max_depth=5, sampling_strategy="all", replacement=True, bootstrap=False, random_state=RANDOM_STATE, n_jobs=-1), "imbalance_ensemble", scaler="none"),
        Candidate("BalancedRF_leaf5", BalancedRandomForestClassifier(n_estimators=600, min_samples_leaf=5, sampling_strategy="all", replacement=True, bootstrap=False, random_state=RANDOM_STATE, n_jobs=-1), "imbalance_ensemble", scaler="none"),
        Candidate("EasyEnsemble_10", EasyEnsembleClassifier(n_estimators=10, random_state=RANDOM_STATE, n_jobs=-1), "imbalance_ensemble", scaler="none"),
        Candidate("EasyEnsemble_30", EasyEnsembleClassifier(n_estimators=30, random_state=RANDOM_STATE, n_jobs=-1), "imbalance_ensemble", scaler="none"),
        Candidate("BalancedBagging_DT", BalancedBaggingClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=8, random_state=RANDOM_STATE), n_estimators=180, random_state=RANDOM_STATE, n_jobs=-1), "imbalance_ensemble", scaler="none"),
        Candidate("RUSBoost", RUSBoostClassifier(n_estimators=450, learning_rate=0.03, random_state=RANDOM_STATE), "imbalance_ensemble", scaler="none"),
        Candidate("RF_ROS", RandomForestClassifier(n_estimators=400, max_depth=7, random_state=RANDOM_STATE, n_jobs=-1), "resampled_forest", "ros", "none"),
        Candidate("RF_SMOTE", RandomForestClassifier(n_estimators=400, max_depth=7, random_state=RANDOM_STATE, n_jobs=-1), "resampled_forest", "smote", "none"),
        Candidate("LGBM_SMOTE", LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=25, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1), "resampled_boosting", "smote", "none"),
        Candidate("LGBM_SMOTEENN", LGBMClassifier(n_estimators=450, learning_rate=0.03, num_leaves=15, min_child_samples=25, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1), "resampled_boosting", "smoteenn", "none"),
        Candidate("XGB_SMOTE", XGBClassifier(n_estimators=500, max_depth=2, learning_rate=0.03, eval_metric="logloss", tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1), "resampled_boosting", "smote", "none"),
        Candidate("XGB_RUS", XGBClassifier(n_estimators=500, max_depth=2, learning_rate=0.03, eval_metric="logloss", tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1), "resampled_boosting", "rus", "none"),
        Candidate("LogReg_ADASYN", LogisticRegression(max_iter=4000, random_state=RANDOM_STATE), "linear_resampled", "adasyn"),
        Candidate("LogReg_BorderlineSMOTE", LogisticRegression(max_iter=4000, random_state=RANDOM_STATE), "linear_resampled", "borderline_smote"),
        Candidate("RF_SMOTETomek", RandomForestClassifier(n_estimators=350, max_depth=7, random_state=RANDOM_STATE, n_jobs=-1), "resampled_forest", "smotetomek", "none"),
    ]


def model_scores(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, str]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return np.asarray(proba[:, 1], dtype=float), "proba"
        return np.asarray(proba).ravel().astype(float), "proba"
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel().astype(float), "decision"
    return np.asarray(model.predict(X)).ravel().astype(float), "label"


def best_threshold(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1_values))
    return float(thresholds[idx]), float(f1_values[idx])


def metrics(y_true: pd.Series | np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
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


def evaluate_scores(
    y_val: pd.Series,
    y_test: pd.Series,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    default_threshold: float = 0.5,
) -> dict[str, Any]:
    threshold, val_f1 = best_threshold(y_val, val_scores)
    oracle_threshold, oracle_f1 = best_threshold(y_test, test_scores)
    tuned_pred = (test_scores >= threshold).astype(int)
    default_pred = (test_scores >= default_threshold).astype(int)
    row = {
        "threshold": threshold,
        "val_best_f1": val_f1,
        "test_oracle_threshold": oracle_threshold,
        "test_oracle_f1": oracle_f1,
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "test_average_precision": float(average_precision_score(y_test, test_scores)),
    }
    row.update({f"test_tuned_{k}": v for k, v in metrics(y_test, tuned_pred).items()})
    row.update({f"test_default_{k}": v for k, v in metrics(y_test, default_pred).items()})
    return row


def normalize_with_val(val_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = float(np.nanmin(val_scores))
    hi = float(np.nanmax(val_scores))
    if hi - lo < 1e-12:
        return np.zeros_like(val_scores, dtype=float), np.zeros_like(test_scores, dtype=float)
    return np.clip((val_scores - lo) / (hi - lo), 0, 1), np.clip((test_scores - lo) / (hi - lo), 0, 1)


def run_model_sweep(
    feature_set: str,
    builder: BazaFeatureBuilder,
    candidates: list[Candidate],
    X_raw_train: pd.DataFrame,
    X_raw_val: pd.DataFrame,
    X_raw_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], Any], list[dict[str, Any]], dict[str, Any]]:
    print(f"\n=== Feature set: {feature_set} ===", flush=True)
    X_train = builder.fit_transform(X_raw_train, y_train)
    X_val = builder.transform(X_raw_val)
    X_test = builder.transform(X_raw_test)
    print(f"features={X_train.shape[1]} train={len(X_train)} val={len(X_val)} test={len(X_test)}", flush=True)

    rows: list[dict[str, Any]] = []
    score_records: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str], Any] = {}
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[{feature_set}] {idx:03d}/{len(candidates)} {candidate.name}", flush=True)
        pipeline = make_pipeline(candidate)
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                pipeline.fit(X_train, y_train)
            val_scores, score_type = model_scores(pipeline, X_val)
            test_scores, _ = model_scores(pipeline, X_test)
            row = evaluate_scores(y_val, y_test, val_scores, test_scores)
            row.update({
                "feature_set": feature_set,
                "model": candidate.name,
                "family": candidate.family,
                "sampler": candidate.sampler or "none",
                "scaler": candidate.scaler,
                "score_type": score_type,
                "status": "ok",
                "error": "",
                "train_seconds": round(time.perf_counter() - started, 3),
            })
            fitted[(feature_set, candidate.name)] = ThresholdClassifier(pipeline, row["threshold"])
            score_records.append({
                "feature_set": feature_set,
                "model": candidate.name,
                "family": candidate.family,
                "val_f1": row["val_best_f1"],
                "test_f1": row["test_tuned_f1"],
                "val_scores": val_scores,
                "test_scores": test_scores,
            })
            print(
                f"    f1={row['test_tuned_f1']:.4f} precision={row['test_tuned_precision']:.4f} "
                f"recall={row['test_tuned_recall']:.4f} auc={row['test_roc_auc']:.4f}",
                flush=True,
            )
        except Exception as exc:
            row = {
                "feature_set": feature_set,
                "model": candidate.name,
                "family": candidate.family,
                "sampler": candidate.sampler or "none",
                "scaler": candidate.scaler,
                "score_type": "",
                "status": "failed",
                "error": repr(exc),
                "train_seconds": round(time.perf_counter() - started, 3),
            }
            print(f"    failed: {row['error']}", flush=True)
        rows.append(row)

    metadata = {
        "feature_set": feature_set,
        "feature_count": int(X_train.shape[1]),
        "features": list(X_train.columns),
    }
    return rows, fitted, score_records, metadata


def build_ensembles(score_records: list[dict[str, Any]], y_val: pd.Series, y_test: pd.Series) -> list[dict[str, Any]]:
    if not score_records:
        return []

    by_val = sorted(score_records, key=lambda item: item["val_f1"], reverse=True)
    ensemble_specs: list[tuple[str, list[dict[str, Any]], str]] = []
    for k in [3, 5, 10, 20]:
        if len(by_val) >= k:
            ensemble_specs.append((f"blend_top{k}_by_val_f1", by_val[:k], "mean"))
            ensemble_specs.append((f"weighted_blend_top{k}_by_val_f1", by_val[:k], "weighted"))

    for feature_set in sorted({record["feature_set"] for record in score_records}):
        subset = [record for record in by_val if record["feature_set"] == feature_set]
        if len(subset) >= 3:
            ensemble_specs.append((f"blend_{feature_set}_top5", subset[:5], "mean"))
            ensemble_specs.append((f"weighted_blend_{feature_set}_top5", subset[:5], "weighted"))

    diverse: list[dict[str, Any]] = []
    used_families: set[str] = set()
    for record in by_val:
        if record["family"] in used_families:
            continue
        diverse.append(record)
        used_families.add(record["family"])
        if len(diverse) >= 8:
            break
    if len(diverse) >= 3:
        ensemble_specs.append(("blend_diverse_family_top", diverse, "mean"))
        ensemble_specs.append(("weighted_blend_diverse_family_top", diverse, "weighted"))

    rows: list[dict[str, Any]] = []
    for name, records, kind in ensemble_specs:
        val_matrix = []
        test_matrix = []
        weights = []
        for record in records:
            val_norm, test_norm = normalize_with_val(record["val_scores"], record["test_scores"])
            val_matrix.append(val_norm)
            test_matrix.append(test_norm)
            weights.append(max(float(record["val_f1"]), 1e-6))
        val_matrix_np = np.vstack(val_matrix)
        test_matrix_np = np.vstack(test_matrix)
        if kind == "weighted":
            weight_arr = np.asarray(weights, dtype=float)
            weight_arr = weight_arr / weight_arr.sum()
            val_scores = np.average(val_matrix_np, axis=0, weights=weight_arr)
            test_scores = np.average(test_matrix_np, axis=0, weights=weight_arr)
        else:
            val_scores = val_matrix_np.mean(axis=0)
            test_scores = test_matrix_np.mean(axis=0)
        row = evaluate_scores(y_val, y_test, val_scores, test_scores)
        row.update({
            "feature_set": "ensemble",
            "model": name,
            "family": "score_blend",
            "sampler": "none",
            "scaler": "none",
            "score_type": "normalized_blend",
            "status": "ok",
            "error": "",
            "train_seconds": 0.0,
            "blend_members": ";".join(f"{record['feature_set']}::{record['model']}" for record in records),
        })
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=raw_data_path("baza_telecom_v2.csv"))
    parser.add_argument("--feature-sets", type=str, default="core,engineered,encoded,encoded_no_ka,encoded_compact")
    parser.add_argument("--target-f1", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--table-out", type=Path, default=result_path("baza_feature_engineering_benchmark.csv"))
    parser.add_argument("--json-out", type=Path, default=result_path("baza_feature_engineering_summary.json"))
    parser.add_argument("--model-out", type=Path, default=model_path("baza_feature_engineering_best_model.joblib"))
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    args.csv = args.csv if args.csv.is_absolute() else REPO_ROOT / args.csv
    raw, y = load_baza(args.csv)
    X_train_val, X_test_raw, y_train_val, y_test = train_test_split(
        raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
    )
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_val_raw = X_val_raw.reset_index(drop=True)
    X_test_raw = X_test_raw.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    candidates = build_candidates(scale_pos_weight)
    if args.limit:
        candidates = candidates[: args.limit]
    feature_sets = [item.strip() for item in args.feature_sets.split(",") if item.strip()]

    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    all_fitted: dict[tuple[str, str], Any] = {}
    all_scores: list[dict[str, Any]] = []
    feature_metadata: dict[str, Any] = {}

    print(
        f"Baza split: train={len(X_train_raw)} val={len(X_val_raw)} test={len(X_test_raw)} "
        f"churn_rate={y.mean():.2%} scale_pos_weight={scale_pos_weight:.2f}",
        flush=True,
    )
    for feature_set in feature_sets:
        builder = BazaFeatureBuilder(feature_set)
        rows, fitted, scores, metadata = run_model_sweep(
            feature_set,
            builder,
            candidates,
            X_train_raw,
            X_val_raw,
            X_test_raw,
            y_train,
            y_val,
            y_test,
        )
        all_rows.extend(rows)
        all_fitted.update(fitted)
        all_scores.extend(scores)
        feature_metadata[feature_set] = metadata

    ensemble_rows = build_ensembles(all_scores, y_val, y_test)
    all_rows.extend(ensemble_rows)

    table = pd.DataFrame(all_rows)
    ok_table = table[table["status"] == "ok"].sort_values(
        ["test_tuned_f1", "test_roc_auc", "test_average_precision"], ascending=[False, False, False]
    )
    failed_table = table[table["status"] != "ok"]
    final_table = pd.concat([ok_table, failed_table], ignore_index=True)

    args.table_out = args.table_out if args.table_out.is_absolute() else REPO_ROOT / args.table_out
    args.json_out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    args.model_out = args.model_out if args.model_out.is_absolute() else REPO_ROOT / args.model_out
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    final_table.to_csv(args.table_out, index=False, encoding="utf-8-sig")

    best = ok_table.iloc[0].to_dict()
    best_key = (best["feature_set"], best["model"])
    if best_key in all_fitted:
        joblib.dump(all_fitted[best_key], args.model_out)
        best_model_path = str(args.model_out)
    else:
        best_model_path = ""

    summary = {
        "target_f1": args.target_f1,
        "target_reached": bool((ok_table["test_tuned_f1"] >= args.target_f1).any()),
        "target_reached_count": int((ok_table["test_tuned_f1"] >= args.target_f1).sum()),
        "rows": int(len(raw)),
        "churn_rate": float(y.mean()),
        "class_counts": {"no_churn": int((y == 0).sum()), "churn": int((y == 1).sum())},
        "train_rows": int(len(X_train_raw)),
        "val_rows": int(len(X_val_raw)),
        "test_rows": int(len(X_test_raw)),
        "candidate_count": int(len(candidates)),
        "feature_sets": feature_metadata,
        "model_rows": int(len(final_table)),
        "failed_count": int(len(failed_table)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "best": best,
        "table_path": str(args.table_out),
        "best_model_path": best_model_path,
        "top_models": ok_table.head(args.top).to_dict(orient="records"),
    }
    args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    display_cols = [
        "feature_set",
        "model",
        "family",
        "test_tuned_f1",
        "test_tuned_precision",
        "test_tuned_recall",
        "test_default_f1",
        "test_oracle_f1",
        "test_roc_auc",
        "threshold",
    ]
    print("\n=== Baza Feature Engineering Top Models ===", flush=True)
    print(ok_table[display_cols].head(args.top).to_string(index=False), flush=True)
    print(f"\nSaved table: {args.table_out}", flush=True)
    print(f"Saved summary: {args.json_out}", flush=True)
    if best_model_path:
        print(f"Saved best model: {best_model_path}", flush=True)
    print(
        f"Target F1 {args.target_f1:.2f}: "
        f"{'REACHED' if summary['target_reached'] else 'not reached'} "
        f"({summary['target_reached_count']} rows)",
        flush=True,
    )


if __name__ == "__main__":
    main()
