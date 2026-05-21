#!/usr/bin/env python3
"""PID-safe Baza churn experiment with leakage-safe risk features.

This is a focused Baza-only experiment for:

* keeping PID as a tracking key only,
* target encoding KA_name on train folds only,
* deriving Billing_ZIP region risk from train folds only,
* one-hot encoding the value segment,
* using imbalance-aware tree models,
* tuning the churn threshold on validation scores,
* reattaching PID only after predictions for monitoring output.
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
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.threshold_model import ThresholdClassifier  # noqa: E402
from src.utils.helpers import model_path, raw_data_path, result_path  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency in some envs
    CatBoostClassifier = None


RANDOM_STATE = 42
TARGET = "CHURN"
TRACKING_COLUMNS = ["PID"]


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: Any


class WeightedProbabilityEnsemble:
    """Small validation-selected probability ensemble.

    Optional reference scores are validation-only distributions used to convert
    each model score to a stable percentile before averaging.
    """

    def __init__(
        self,
        estimators: list[Any],
        weights: np.ndarray,
        reference_scores: list[np.ndarray] | None = None,
    ) -> None:
        self.estimators = estimators
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.reference_scores = reference_scores

    @staticmethod
    def _percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
        reference = np.sort(np.asarray(reference, dtype=float))
        if reference.size == 0:
            return np.zeros_like(values, dtype=float)
        return np.searchsorted(reference, values, side="right") / float(reference.size)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score_columns = []
        for index, estimator in enumerate(self.estimators):
            scores = model_scores(estimator, X)
            if self.reference_scores is not None:
                scores = self._percentile(self.reference_scores[index], scores)
            score_columns.append(scores)
        blended = np.column_stack(score_columns).dot(self.weights)
        blended = np.clip(blended, 0.0, 1.0)
        return np.column_stack([1.0 - blended, blended])


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def logit_clip(values: pd.Series) -> pd.Series:
    clipped = values.clip(1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def load_baza(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    return df.loc[valid].reset_index(drop=True), y.loc[valid].astype(int).reset_index(drop=True)


class PidSafeRiskFeatureBuilder:
    """Builds risk and PID-aggregation features without carrying raw PID into X."""

    def __init__(
        self,
        smoothing: float = 30.0,
        region_smoothing: float = 60.0,
        folds: int = 5,
        pid_aggregation_scope: str = "full_snapshot",
    ) -> None:
        self.smoothing = smoothing
        self.region_smoothing = region_smoothing
        self.folds = folds
        self.pid_aggregation_scope = pid_aggregation_scope
        self.global_mean = 0.0
        self.ka_map: dict[str, float] = {}
        self.ka_woe_map: dict[str, float] = {}
        self.region_map: dict[str, float] = {}
        self.region_woe_map: dict[str, float] = {}
        self.ka_freq: dict[str, float] = {}
        self.region_freq: dict[str, float] = {}
        self.pid_agg_map = pd.DataFrame()
        self.pid_agg_columns: list[str] = []
        self.value_segments: list[str] = []
        self.effective_segments: list[str] = []
        self.feature_columns: list[str] = []

    def _prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df.columns = df.columns.str.strip()
        for col in [
            "Billing_ZIP",
            "Active_subscribers",
            "Not_Active_subscribers",
            "Suspended_subscribers",
            "Total_SUBs",
            "AvgMobileRevenue",
            "AvgFIXRevenue",
            "TotalRevenue",
            "ARPU",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["pid_safe"] = df["PID"].fillna("Unknown").astype(str).str.strip()
        df["Not_Active_missing"] = df["Not_Active_subscribers"].isna().astype(float)
        df["Suspended_missing"] = df["Suspended_subscribers"].isna().astype(float)
        df["ARPU_missing"] = df["ARPU"].isna().astype(float)
        df["Billing_ZIP_missing"] = df["Billing_ZIP"].isna().astype(float)
        df["Not_Active_subscribers"] = df["Not_Active_subscribers"].fillna(0.0)
        df["Suspended_subscribers"] = df["Suspended_subscribers"].fillna(0.0)
        arpu_calc = safe_divide(df["TotalRevenue"], df["Total_SUBs"])
        df["ARPU"] = df["ARPU"].fillna(arpu_calc)

        df["value_segment"] = df["CRM_PID_Value_Segment"].fillna("Unknown").astype(str).str.strip()
        df["effective_segment"] = df["EffectiveSegment"].fillna("Unknown").astype(str).str.strip()
        df["ka_name_safe"] = df["KA_name"].fillna("Unknown").astype(str).str.strip()
        zip_numeric = df["Billing_ZIP"].fillna(-1).round().astype(int)
        df["billing_zip_safe"] = zip_numeric.astype(str)
        df["zip_prefix_1"] = df["billing_zip_safe"].str[:1].replace("", "Unknown")
        df["zip_prefix_2"] = df["billing_zip_safe"].str[:2].replace("", "Unknown")
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
        active_ratio = safe_divide(active, total_subs).clip(0, 1)
        inactive_ratio = safe_divide(inactive, total_subs).clip(0, 1)
        suspended_ratio = safe_divide(suspended, total_subs).clip(0, 1)
        dormant_ratio = safe_divide(dormant, total_subs).clip(0, 1)
        revenue_per_sub = safe_divide(revenue, total_subs)
        revenue_per_active = safe_divide(revenue, active)
        mobile_ratio = safe_divide(mobile, revenue).clip(0, 1)
        fixed_ratio = safe_divide(fixed, revenue).clip(0, 1)

        features = pd.DataFrame(index=df.index)
        base = {
            "active_subscribers": active,
            "not_active_subscribers": inactive,
            "suspended_subscribers": suspended,
            "dormant_subscribers": dormant,
            "total_subs": total_subs,
            "avg_mobile_revenue": mobile,
            "avg_fix_revenue": fixed,
            "total_revenue": revenue,
            "arpu": arpu,
            "billing_zip_numeric": df["Billing_ZIP"],
            "active_ratio": active_ratio,
            "inactive_ratio": inactive_ratio,
            "suspended_ratio": suspended_ratio,
            "dormant_ratio": dormant_ratio,
            "mobile_revenue_ratio": mobile_ratio,
            "fixed_revenue_ratio": fixed_ratio,
            "revenue_per_sub": revenue_per_sub,
            "revenue_per_active": revenue_per_active,
            "mobile_per_sub": safe_divide(mobile, total_subs),
            "fixed_per_sub": safe_divide(fixed, total_subs),
            "mobile_to_fixed_ratio": safe_divide(mobile, fixed),
            "fixed_to_mobile_ratio": safe_divide(fixed, mobile),
            "inactive_x_revenue": inactive_ratio * revenue.fillna(0.0),
            "suspended_x_revenue": suspended_ratio * revenue.fillna(0.0),
            "revenue_balance": (
                pd.concat([mobile.fillna(0.0), fixed.fillna(0.0)], axis=1).min(axis=1)
                / (pd.concat([mobile.fillna(0.0), fixed.fillna(0.0)], axis=1).max(axis=1) + 1e-5)
            ).clip(0, 1),
            "not_active_missing": df["Not_Active_missing"],
            "suspended_missing": df["Suspended_missing"],
            "arpu_missing": df["ARPU_missing"],
            "billing_zip_missing": df["Billing_ZIP_missing"],
            "has_inactive": inactive.gt(0).astype(float),
            "has_suspended": suspended.gt(0).astype(float),
            "has_dormant": dormant.gt(0).astype(float),
            "multi_subscriber": total_subs.gt(1).astype(float),
            "large_account": total_subs.ge(10).astype(float),
            "mobile_only": (mobile.gt(0) & fixed.eq(0)).astype(float),
            "fixed_only": (fixed.gt(0) & mobile.eq(0)).astype(float),
            "zero_revenue": revenue.eq(0).astype(float),
        }
        for name, values in base.items():
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

        return features

    def _fit_pid_aggregates(self, raw_for_pid: pd.DataFrame) -> None:
        prepared = self._prepare(raw_for_pid)
        agg = prepared.groupby("pid_safe").agg(
            pid_row_count=("pid_safe", "size"),
            pid_active_subscribers_sum=("Active_subscribers", "sum"),
            pid_not_active_subscribers_sum=("Not_Active_subscribers", "sum"),
            pid_suspended_subscribers_sum=("Suspended_subscribers", "sum"),
            pid_total_subs_sum=("Total_SUBs", "sum"),
            pid_total_subs_mean=("Total_SUBs", "mean"),
            pid_total_subs_max=("Total_SUBs", "max"),
            pid_mobile_revenue_sum=("AvgMobileRevenue", "sum"),
            pid_fixed_revenue_sum=("AvgFIXRevenue", "sum"),
            pid_total_revenue_sum=("TotalRevenue", "sum"),
            pid_total_revenue_mean=("TotalRevenue", "mean"),
            pid_total_revenue_max=("TotalRevenue", "max"),
            pid_total_revenue_std=("TotalRevenue", "std"),
            pid_arpu_mean=("ARPU", "mean"),
            pid_arpu_max=("ARPU", "max"),
            pid_zip_nunique=("billing_zip_safe", "nunique"),
            pid_ka_nunique=("ka_name_safe", "nunique"),
            pid_value_segment_nunique=("value_segment", "nunique"),
            pid_effective_segment_nunique=("effective_segment", "nunique"),
        )
        agg["pid_multi_row"] = agg["pid_row_count"].gt(1).astype(float)
        agg["pid_dormant_subscribers_sum"] = (
            agg["pid_not_active_subscribers_sum"] + agg["pid_suspended_subscribers_sum"]
        )
        agg["pid_dormant_ratio"] = safe_divide(
            agg["pid_dormant_subscribers_sum"],
            agg["pid_total_subs_sum"],
        ).clip(0, 1)
        agg["pid_mobile_revenue_ratio"] = safe_divide(
            agg["pid_mobile_revenue_sum"],
            agg["pid_total_revenue_sum"],
        ).clip(0, 1)
        agg["pid_fixed_revenue_ratio"] = safe_divide(
            agg["pid_fixed_revenue_sum"],
            agg["pid_total_revenue_sum"],
        ).clip(0, 1)
        agg = agg.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        self.pid_agg_map = agg
        self.pid_agg_columns = list(agg.columns)

    def _pid_aggregate_features(self, prepared: pd.DataFrame) -> pd.DataFrame:
        mapped = self.pid_agg_map.reindex(prepared["pid_safe"].astype(str)).reset_index(drop=True)
        mapped = mapped.reindex(columns=self.pid_agg_columns)

        fallback = pd.DataFrame(index=prepared.index)
        fallback["pid_row_count"] = 1.0
        fallback["pid_active_subscribers_sum"] = prepared["Active_subscribers"]
        fallback["pid_not_active_subscribers_sum"] = prepared["Not_Active_subscribers"]
        fallback["pid_suspended_subscribers_sum"] = prepared["Suspended_subscribers"]
        fallback["pid_total_subs_sum"] = prepared["Total_SUBs"]
        fallback["pid_total_subs_mean"] = prepared["Total_SUBs"]
        fallback["pid_total_subs_max"] = prepared["Total_SUBs"]
        fallback["pid_mobile_revenue_sum"] = prepared["AvgMobileRevenue"]
        fallback["pid_fixed_revenue_sum"] = prepared["AvgFIXRevenue"]
        fallback["pid_total_revenue_sum"] = prepared["TotalRevenue"]
        fallback["pid_total_revenue_mean"] = prepared["TotalRevenue"]
        fallback["pid_total_revenue_max"] = prepared["TotalRevenue"]
        fallback["pid_total_revenue_std"] = 0.0
        fallback["pid_arpu_mean"] = prepared["ARPU"]
        fallback["pid_arpu_max"] = prepared["ARPU"]
        fallback["pid_zip_nunique"] = 1.0
        fallback["pid_ka_nunique"] = 1.0
        fallback["pid_value_segment_nunique"] = 1.0
        fallback["pid_effective_segment_nunique"] = 1.0
        fallback["pid_multi_row"] = 0.0
        fallback["pid_dormant_subscribers_sum"] = (
            prepared["Not_Active_subscribers"] + prepared["Suspended_subscribers"]
        )
        fallback["pid_dormant_ratio"] = safe_divide(
            fallback["pid_dormant_subscribers_sum"],
            fallback["pid_total_subs_sum"],
        ).clip(0, 1)
        fallback["pid_mobile_revenue_ratio"] = safe_divide(
            fallback["pid_mobile_revenue_sum"],
            fallback["pid_total_revenue_sum"],
        ).clip(0, 1)
        fallback["pid_fixed_revenue_ratio"] = safe_divide(
            fallback["pid_fixed_revenue_sum"],
            fallback["pid_total_revenue_sum"],
        ).clip(0, 1)

        mapped = mapped.combine_first(fallback).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        mapped["pid_revenue_share"] = safe_divide(
            prepared["TotalRevenue"].reset_index(drop=True),
            mapped["pid_total_revenue_sum"],
        ).clip(0, 1)
        mapped["pid_total_subs_share"] = safe_divide(
            prepared["Total_SUBs"].reset_index(drop=True),
            mapped["pid_total_subs_sum"],
        ).clip(0, 1)
        mapped["pid_revenue_vs_pid_mean"] = (
            prepared["TotalRevenue"].reset_index(drop=True) - mapped["pid_total_revenue_mean"]
        )
        mapped["pid_arpu_vs_pid_mean"] = prepared["ARPU"].reset_index(drop=True) - mapped["pid_arpu_mean"]
        for col in [
            "pid_row_count",
            "pid_total_subs_sum",
            "pid_total_revenue_sum",
            "pid_mobile_revenue_sum",
            "pid_fixed_revenue_sum",
        ]:
            mapped[f"log1p_{col}"] = np.log1p(mapped[col].clip(lower=0))
        return mapped

    def _fit_smoothed_map(
        self,
        categories: pd.Series,
        y: pd.Series,
        smoothing: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        stats = pd.DataFrame({"cat": categories.astype(str), "y": y.values}).groupby("cat")["y"].agg(["sum", "count"])
        target = (stats["sum"] + smoothing * self.global_mean) / (stats["count"] + smoothing)
        return target.to_dict(), logit_clip(target).to_dict()

    def _oof_encoding(
        self,
        categories: pd.Series,
        y: pd.Series,
        smoothing: float,
        kind: str,
    ) -> pd.Series:
        encoded = pd.Series(self.global_mean, index=categories.index, dtype=float)
        splitter = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=RANDOM_STATE)
        for fold_train_idx, fold_valid_idx in splitter.split(categories.to_frame(), y):
            train_cats = categories.iloc[fold_train_idx].astype(str)
            train_y = y.iloc[fold_train_idx].reset_index(drop=True)
            fold_global = float(train_y.mean())
            stats = pd.DataFrame({"cat": train_cats.values, "y": train_y.values}).groupby("cat")["y"].agg(["sum", "count"])
            target = (stats["sum"] + smoothing * fold_global) / (stats["count"] + smoothing)
            mapping = logit_clip(target).to_dict() if kind == "woe" else target.to_dict()
            default = float(logit_clip(pd.Series([fold_global])).iloc[0]) if kind == "woe" else fold_global
            encoded.iloc[fold_valid_idx] = categories.iloc[fold_valid_idx].astype(str).map(mapping).fillna(default)
        return encoded

    def fit(
        self,
        raw_train: pd.DataFrame,
        y_train: pd.Series,
        pid_aggregation_raw: pd.DataFrame | None = None,
    ) -> "PidSafeRiskFeatureBuilder":
        prepared = self._prepare(raw_train)
        y_train = y_train.reset_index(drop=True)
        self.global_mean = float(y_train.mean())
        self._fit_pid_aggregates(pid_aggregation_raw if pid_aggregation_raw is not None else raw_train)
        self.ka_map, self.ka_woe_map = self._fit_smoothed_map(prepared["ka_name_safe"], y_train, self.smoothing)
        self.region_map, self.region_woe_map = self._fit_smoothed_map(
            prepared["zip_prefix_2"], y_train, self.region_smoothing
        )
        self.ka_freq = prepared["ka_name_safe"].astype(str).value_counts(normalize=True).to_dict()
        self.region_freq = prepared["zip_prefix_2"].astype(str).value_counts(normalize=True).to_dict()
        self.value_segments = sorted(prepared["value_segment"].astype(str).unique().tolist())
        self.effective_segments = sorted(prepared["effective_segment"].astype(str).unique().tolist())
        _ = self.transform(raw_train, y_for_oof=y_train)
        return self

    def _one_hot(self, features: pd.DataFrame, values: pd.Series, prefix: str, categories: list[str]) -> None:
        values = values.astype(str)
        for category in categories:
            safe_category = "".join(ch if ch.isalnum() else "_" for ch in category) or "Unknown"
            features[f"{prefix}_{safe_category}"] = values.eq(category).astype(float)

    def transform(self, raw: pd.DataFrame, y_for_oof: pd.Series | None = None) -> pd.DataFrame:
        prepared = self._prepare(raw)
        features = self._numeric_features(prepared)
        features = pd.concat([features, self._pid_aggregate_features(prepared)], axis=1)

        if y_for_oof is not None:
            y_for_oof = y_for_oof.reset_index(drop=True)
            features["ka_name_target_rate"] = self._oof_encoding(
                prepared["ka_name_safe"], y_for_oof, self.smoothing, "target"
            )
            features["ka_name_woe"] = self._oof_encoding(
                prepared["ka_name_safe"], y_for_oof, self.smoothing, "woe"
            )
            features["region_risk_score"] = self._oof_encoding(
                prepared["zip_prefix_2"], y_for_oof, self.region_smoothing, "target"
            )
            features["region_risk_woe"] = self._oof_encoding(
                prepared["zip_prefix_2"], y_for_oof, self.region_smoothing, "woe"
            )
        else:
            default_woe = float(logit_clip(pd.Series([self.global_mean])).iloc[0])
            features["ka_name_target_rate"] = (
                prepared["ka_name_safe"].astype(str).map(self.ka_map).fillna(self.global_mean).astype(float)
            )
            features["ka_name_woe"] = (
                prepared["ka_name_safe"].astype(str).map(self.ka_woe_map).fillna(default_woe).astype(float)
            )
            features["region_risk_score"] = (
                prepared["zip_prefix_2"].astype(str).map(self.region_map).fillna(self.global_mean).astype(float)
            )
            features["region_risk_woe"] = (
                prepared["zip_prefix_2"].astype(str).map(self.region_woe_map).fillna(default_woe).astype(float)
            )

        features["ka_name_frequency"] = prepared["ka_name_safe"].astype(str).map(self.ka_freq).fillna(0.0).astype(float)
        features["region_frequency"] = (
            prepared["zip_prefix_2"].astype(str).map(self.region_freq).fillna(0.0).astype(float)
        )
        self._one_hot(features, prepared["value_segment"], "value_segment", self.value_segments)
        self._one_hot(features, prepared["effective_segment"], "effective_segment", self.effective_segments)

        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if not self.feature_columns:
            self.feature_columns = list(features.columns)
        features = features.reindex(columns=self.feature_columns, fill_value=0.0)
        blocked = [col for col in features.columns if col.upper() == "PID" or col.upper().startswith("CRM_PID")]
        if blocked:
            raise ValueError(f"Tracking/id columns leaked into model features: {blocked}")
        return features

    def fit_transform(
        self,
        raw_train: pd.DataFrame,
        y_train: pd.Series,
        pid_aggregation_raw: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        self.fit(raw_train, y_train, pid_aggregation_raw=pid_aggregation_raw)
        return self.transform(raw_train, y_for_oof=y_train)


def best_f1_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1_values))
    return float(thresholds[idx]), float(f1_values[idx])


def metrics_at_threshold(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pred_pos_rate": float(np.mean(pred)),
    }


def model_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba[:, 1] if proba.ndim == 2 else proba).ravel().astype(float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X)).ravel().astype(float)
        lo, hi = float(np.min(scores)), float(np.max(scores))
        return (scores - lo) / max(hi - lo, 1e-12)
    return np.asarray(model.predict(X)).ravel().astype(float)


def summarize_scores(
    name: str,
    branch: str,
    y_val: pd.Series,
    y_test: pd.Series,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    fixed_thresholds: list[float],
    started_at: float,
    scale_pos_weight: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    threshold, val_best_f1 = best_f1_threshold(y_val, val_scores)
    selected_metrics = metrics_at_threshold(y_test, test_scores, threshold)
    default_metrics = metrics_at_threshold(y_test, test_scores, 0.5)
    fixed_rows = [metrics_at_threshold(y_test, test_scores, item) for item in fixed_thresholds]
    row = {
        "model": name,
        "scale_pos_weight": scale_pos_weight,
        "imbalance_mode": branch,
        "selected_threshold": threshold,
        "val_best_f1": val_best_f1,
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "test_average_precision": float(average_precision_score(y_test, test_scores)),
        "train_seconds": round(time.perf_counter() - started_at, 3),
    }
    row.update({f"test_tuned_{key}": value for key, value in selected_metrics.items()})
    row.update({f"test_default_{key}": value for key, value in default_metrics.items()})
    for item in fixed_rows:
        key = str(item["threshold"]).replace(".", "_")
        row[f"test_f1_at_{key}"] = item["f1"]
        row[f"test_recall_at_{key}"] = item["recall"]
        row[f"test_precision_at_{key}"] = item["precision"]
    return row, fixed_rows


def build_candidates(scale_pos_weight: float) -> list[Candidate]:
    candidates = [
        Candidate(
            "XGB_aucpr_d2_spw",
            XGBClassifier(
                n_estimators=700,
                max_depth=2,
                learning_rate=0.03,
                min_child_weight=4,
                reg_lambda=5.0,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        Candidate(
            "XGB_aucpr_d3_spw",
            XGBClassifier(
                n_estimators=550,
                max_depth=3,
                learning_rate=0.03,
                min_child_weight=5,
                reg_lambda=6.0,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        Candidate(
            "LGBM_is_unbalance_leaf7",
            LGBMClassifier(
                n_estimators=700,
                learning_rate=0.025,
                num_leaves=7,
                min_child_samples=35,
                reg_lambda=5.0,
                subsample=0.85,
                colsample_bytree=0.85,
                is_unbalance=True,
                objective="binary",
                n_jobs=-1,
                verbose=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        Candidate(
            "LGBM_is_unbalance_leaf15",
            LGBMClassifier(
                n_estimators=550,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=30,
                reg_lambda=3.0,
                subsample=0.85,
                colsample_bytree=0.85,
                is_unbalance=True,
                objective="binary",
                n_jobs=-1,
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
        Candidate(
            "RF_balanced_subsample",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=7,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
    if CatBoostClassifier is not None:
        candidates.append(
            Candidate(
                "CatBoost_balanced_d4",
                CatBoostClassifier(
                    iterations=700,
                    learning_rate=0.03,
                    depth=4,
                    l2_leaf_reg=8.0,
                    auto_class_weights="Balanced",
                    eval_metric="PRAUC",
                    random_seed=RANDOM_STATE,
                    verbose=0,
                    allow_writing_files=False,
                ),
            )
        )
    return candidates


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=raw_data_path("baza_telecom_v2.csv"))
    parser.add_argument("--target-f1", type=float, default=0.6)
    parser.add_argument("--fixed-thresholds", type=str, default="0.3,0.35,0.4,0.5")
    parser.add_argument(
        "--pid-aggregation-scope",
        choices=["train_only", "full_snapshot"],
        default="full_snapshot",
        help="Use train-only PID aggregates for strict validation or full non-target snapshot aggregates for batch scoring.",
    )
    parser.add_argument("--table-out", type=Path, default=result_path("baza_pid_safe_risk_feature_table.csv"))
    parser.add_argument("--json-out", type=Path, default=result_path("baza_pid_safe_risk_feature_summary.json"))
    parser.add_argument("--predictions-out", type=Path, default=result_path("baza_pid_safe_risk_monitoring_predictions.csv"))
    parser.add_argument("--model-out", type=Path, default=model_path("baza_pid_safe_risk_best_model.joblib"))
    args = parser.parse_args()

    started = time.perf_counter()
    args.csv = args.csv if args.csv.is_absolute() else REPO_ROOT / args.csv
    raw, y = load_baza(args.csv)
    pid = raw["PID"].copy()

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
    pid_test = pid.iloc[test_idx].reset_index(drop=True)

    builder = PidSafeRiskFeatureBuilder(pid_aggregation_scope=args.pid_aggregation_scope)
    pid_aggregation_raw = raw if args.pid_aggregation_scope == "full_snapshot" else None
    X_train = builder.fit_transform(raw_train, y_train, pid_aggregation_raw=pid_aggregation_raw)
    X_val = builder.transform(raw_val)
    X_test = builder.transform(raw_test)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    fixed_thresholds = [float(item.strip()) for item in args.fixed_thresholds.split(",") if item.strip()]
    candidates = build_candidates(scale_pos_weight)

    print(
        f"Baza PID-safe split: train={len(X_train)} val={len(X_val)} test={len(X_test)} "
        f"churn_rate={y.mean():.2%} scale_pos_weight={scale_pos_weight:.2f} features={X_train.shape[1]}",
        flush=True,
    )
    pid_unique = int(raw["PID"].nunique(dropna=True))
    pid_duplicate_rows = int(len(raw) - pid_unique)
    print(
        "PID isolation: raw PID is excluded from X; non-target PID aggregates are used, "
        f"then PID is reattached only in predictions output. pid_unique={pid_unique} "
        f"duplicate_rows={pid_duplicate_rows} scope={args.pid_aggregation_scope}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    score_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    threshold_tables: dict[str, list[dict[str, Any]]] = {}

    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index:02d}/{len(candidates):02d}] {candidate.name}", flush=True)
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", clone(candidate.estimator)),
            ]
        )
        model_started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        val_scores = model_scores(pipeline, X_val)
        test_scores = model_scores(pipeline, X_test)
        branch = (
            "scale_pos_weight"
            if candidate.name.startswith("XGB")
            else "is_unbalance"
            if candidate.name.startswith("LGBM")
            else "class_weight"
        )
        row, fixed_rows = summarize_scores(
            candidate.name,
            branch,
            y_val,
            y_test,
            val_scores,
            test_scores,
            fixed_thresholds,
            model_started,
            scale_pos_weight if candidate.name.startswith("XGB") else None,
        )
        threshold_tables[candidate.name] = fixed_rows
        rows.append(row)
        fitted[candidate.name] = pipeline
        score_cache[candidate.name] = (val_scores, test_scores)
        print(
            f"    val_f1={row['val_best_f1']:.4f} test_f1={row['test_tuned_f1']:.4f} "
            f"precision={row['test_tuned_precision']:.4f} recall={row['test_tuned_recall']:.4f} "
            f"aucpr={row['test_average_precision']:.4f}",
            flush=True,
        )

    individual_table = pd.DataFrame(rows).sort_values(
        ["val_best_f1", "test_average_precision"], ascending=[False, False]
    )
    val_f1_by_name = individual_table.set_index("model")["val_best_f1"].to_dict()

    def append_blend(name: str, member_names: list[str], use_percentiles: bool, weighted: bool) -> None:
        if len(member_names) < 2:
            return
        weights = np.asarray(
            [max(float(val_f1_by_name.get(member, 0.0)), 1e-6) for member in member_names]
            if weighted
            else [1.0] * len(member_names),
            dtype=float,
        )
        reference_scores = [score_cache[member][0] for member in member_names] if use_percentiles else None
        ensemble = WeightedProbabilityEnsemble(
            [fitted[member] for member in member_names],
            weights,
            reference_scores,
        )
        val_scores = model_scores(ensemble, X_val)
        test_scores = model_scores(ensemble, X_test)
        row, fixed_rows = summarize_scores(
            name,
            "validation_percentile_blend" if use_percentiles else "validation_probability_blend",
            y_val,
            y_test,
            val_scores,
            test_scores,
            fixed_thresholds,
            started,
        )
        row["blend_members"] = ";".join(member_names)
        row["blend_weighting"] = "validation_f1" if weighted else "equal"
        threshold_tables[name] = fixed_rows
        rows.append(row)
        fitted[name] = ensemble
        score_cache[name] = (val_scores, test_scores)
        print(
            f"[blend] {name}: val_f1={row['val_best_f1']:.4f} test_f1={row['test_tuned_f1']:.4f} "
            f"precision={row['test_tuned_precision']:.4f} recall={row['test_tuned_recall']:.4f}",
            flush=True,
        )

    ranked_names = individual_table["model"].head(5).tolist()
    append_blend("blend_top3_probability_equal", ranked_names[:3], use_percentiles=False, weighted=False)
    append_blend("blend_top3_percentile_equal", ranked_names[:3], use_percentiles=True, weighted=False)
    append_blend("blend_top4_percentile_valf1", ranked_names[:4], use_percentiles=True, weighted=True)
    append_blend("blend_top5_percentile_equal", ranked_names[:5], use_percentiles=True, weighted=False)

    diverse_names: list[str] = []
    for prefix in ["XGB", "LGBM", "CatBoost", "HistGB", "RF"]:
        matches = [name for name in individual_table["model"].tolist() if name.startswith(prefix)]
        if matches:
            diverse_names.append(matches[0])
    append_blend("blend_diverse_percentile_equal", diverse_names, use_percentiles=True, weighted=False)

    table = pd.DataFrame(rows).sort_values(["val_best_f1", "test_average_precision"], ascending=[False, False])
    best = table.iloc[0].to_dict()
    best_model_name = best["model"]
    best_model = fitted[best_model_name]
    best_threshold = float(best["selected_threshold"])
    best_test_scores = model_scores(best_model, X_test)
    best_test_pred = (best_test_scores >= best_threshold).astype(int)

    args.table_out = args.table_out if args.table_out.is_absolute() else REPO_ROOT / args.table_out
    args.json_out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    args.predictions_out = args.predictions_out if args.predictions_out.is_absolute() else REPO_ROOT / args.predictions_out
    args.model_out = args.model_out if args.model_out.is_absolute() else REPO_ROOT / args.model_out
    for path in [args.table_out, args.json_out, args.predictions_out, args.model_out]:
        path.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(args.table_out, index=False, encoding="utf-8-sig")
    monitoring = pd.DataFrame(
        {
            "PID": pid_test,
            "actual_churn": y_test,
            "churn_probability": best_test_scores,
            "selected_threshold": best_threshold,
            "predicted_churn": best_test_pred,
        }
    )
    monitoring.to_csv(args.predictions_out, index=False, encoding="utf-8-sig")

    model_bundle = {
        "builder": builder,
        "model": ThresholdClassifier(best_model, best_threshold),
        "raw_tracking_columns": TRACKING_COLUMNS,
        "feature_columns": list(X_train.columns),
        "selected_threshold": best_threshold,
        "selected_model": best_model_name,
    }
    joblib.dump(model_bundle, args.model_out)

    summary = {
        "target_f1": args.target_f1,
        "target_reached": bool((table["test_tuned_f1"] >= args.target_f1).any()),
        "selection_rule": "highest validation F1; test metrics reported after selection",
        "pid_isolation": {
            "tracking_columns_excluded_from_features": TRACKING_COLUMNS,
            "feature_matrix_contains_pid": bool(any(col.upper() == "PID" for col in X_train.columns)),
            "feature_matrix_contains_crm_pid_identifier": bool(
                any(col.upper().startswith("CRM_PID") for col in X_train.columns)
            ),
            "note": (
                "PID is used only to create non-target customer aggregation features. "
                "CRM_PID_Value_Segment is treated as value_segment and one-hot encoded; "
                "raw PID is reattached only in monitoring predictions."
            ),
        },
        "feature_strategy": {
            "ka_name": "out-of-fold smoothed target encoding + train-only frequency",
            "billing_zip": "zip_prefix_2 smoothed Region_Risk_Score + train-only frequency",
            "value_segment": "one-hot encoded from train categories",
            "pid_aggregation": (
                "non-target service/revenue/count/diversity aggregates; no raw PID and no PID target mean"
            ),
            "pid_aggregation_scope": args.pid_aggregation_scope,
            "thresholds_tested": fixed_thresholds,
            "scale_pos_weight": scale_pos_weight,
            "small_blends": "top validation models blended with validation-only thresholds; percentile blends use validation score distributions only",
        },
        "baza_rows": int(len(raw)),
        "unique_pid": pid_unique,
        "duplicate_pid_rows": pid_duplicate_rows,
        "baza_churn_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "best_by_validation": best,
        "threshold_tables": threshold_tables,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "table_path": str(args.table_out),
        "predictions_path": str(args.predictions_out),
        "model_path": str(args.model_out),
        "top_models": table.head(20).to_dict(orient="records"),
    }
    args.json_out.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    display_cols = [
        "model",
        "val_best_f1",
        "test_tuned_f1",
        "test_tuned_precision",
        "test_tuned_recall",
        "test_default_f1",
        "test_recall_at_0_3",
        "test_f1_at_0_35",
        "test_average_precision",
        "selected_threshold",
    ]
    print("\n=== PID-safe Baza Risk Feature Results ===", flush=True)
    print(table[display_cols].to_string(index=False), flush=True)
    print(f"\nSaved table: {args.table_out}", flush=True)
    print(f"Saved monitoring predictions: {args.predictions_out}", flush=True)
    print(f"Saved model bundle: {args.model_out}", flush=True)


if __name__ == "__main__":
    main()
