#!/usr/bin/env python3
"""
Train Baza-focused churn models with auxiliary telco datasets.

Evaluation is always done on a held-out Baza split. IBM, Cell2Cell, Iranian,
India monthly usage, and BigML
datasets are used only as auxiliary training rows after mapping them into a
shared telco feature schema.
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
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.models.threshold_model import ThresholdClassifier
from src.utils.helpers import model_path, processed_data_path, raw_data_path, result_path

RANDOM_STATE = 42

COMMON_FEATURES = [
    "monthly_revenue",
    "mobile_revenue",
    "fixed_revenue",
    "total_revenue",
    "arpu",
    "tenure",
    "total_subs",
    "active_subs",
    "inactive_subs",
    "suspended_subs",
    "active_ratio",
    "inactive_ratio",
    "suspended_ratio",
    "usage_minutes",
    "usage_seconds",
    "usage_frequency",
    "sms_frequency",
    "distinct_contacts",
    "dropped_calls",
    "blocked_calls",
    "unanswered_calls",
    "customer_care_calls",
    "revenue_change",
    "minutes_change",
    "complaints",
    "retention_calls",
    "contract_risk",
    "service_status",
    "customer_value",
    "age",
    "zip_code",
    "segment_code",
    "dataset_code",
]


@dataclass(frozen=True)
class TransferCandidate:
    source_group: str
    source_weight: float
    model_name: str
    estimator: Any


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).fillna(0.0)


def blank_frame(index) -> pd.DataFrame:
    return pd.DataFrame(index=index, columns=COMMON_FEATURES, dtype=float)


def load_baza() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(raw_data_path("baza_telecom_v2.csv"))
    df.columns = df.columns.str.strip()
    for col in [
        "AvgMobileRevenue",
        "AvgFIXRevenue",
        "TotalRevenue",
        "ARPU",
        "Total_SUBs",
        "Active_subscribers",
        "Not_Active_subscribers",
        "Suspended_subscribers",
        "Billing_ZIP",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask = df["ARPU"].isna() & df["Total_SUBs"].gt(0)
    df.loc[mask, "ARPU"] = df.loc[mask, "TotalRevenue"] / df.loc[mask, "Total_SUBs"]

    X = blank_frame(df.index)
    X["monthly_revenue"] = df["TotalRevenue"]
    X["mobile_revenue"] = df["AvgMobileRevenue"]
    X["fixed_revenue"] = df["AvgFIXRevenue"]
    X["total_revenue"] = df["TotalRevenue"]
    X["arpu"] = df["ARPU"]
    X["total_subs"] = df["Total_SUBs"]
    X["active_subs"] = df["Active_subscribers"]
    X["inactive_subs"] = df["Not_Active_subscribers"].fillna(0)
    X["suspended_subs"] = df["Suspended_subscribers"].fillna(0)
    X["active_ratio"] = safe_divide(X["active_subs"], X["total_subs"]).clip(0, 1)
    X["inactive_ratio"] = safe_divide(X["inactive_subs"], X["total_subs"]).clip(0, 1)
    X["suspended_ratio"] = safe_divide(X["suspended_subs"], X["total_subs"]).clip(0, 1)
    X["zip_code"] = df["Billing_ZIP"]
    X["segment_code"] = pd.factorize(
        df["CRM_PID_Value_Segment"].astype(str)
        + "_"
        + df["EffectiveSegment"].astype(str)
        + "_"
        + df["KA_name"].astype(str)
    )[0]
    X["dataset_code"] = 0
    y = df["CHURN"].astype(str).str.lower().map({"yes": 1, "no": 0}).astype(int)
    return X[COMMON_FEATURES], y


def load_ibm() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(raw_data_path("ibm_telco_churn.csv"))
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    X = blank_frame(df.index)
    X["monthly_revenue"] = df["MonthlyCharges"]
    X["total_revenue"] = df["TotalCharges"]
    X["arpu"] = df["MonthlyCharges"]
    X["tenure"] = df["tenure"]
    X["contract_risk"] = (df["Contract"] == "Month-to-month").astype(int)
    X["service_status"] = (df["InternetService"] != "No").astype(int)
    X["customer_value"] = df["TotalCharges"]
    X["segment_code"] = pd.factorize(df["PaymentMethod"].astype(str) + "_" + df["Contract"].astype(str))[0]
    X["dataset_code"] = 1
    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return X[COMMON_FEATURES], y


def load_hf_telco() -> tuple[pd.DataFrame, pd.Series]:
    paths = [
        raw_data_path("hf_telco_churn_train.csv"),
        raw_data_path("hf_telco_churn_validation.csv"),
        raw_data_path("hf_telco_churn_test.csv"),
    ]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame(columns=COMMON_FEATURES), pd.Series(dtype=int)

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Customer ID"]).reset_index(drop=True)
    X = blank_frame(df.index)
    monthly_charge = pd.to_numeric(df["Monthly Charge"], errors="coerce")
    total_revenue = pd.to_numeric(df["Total Revenue"], errors="coerce")
    total_charges = pd.to_numeric(df["Total Charges"], errors="coerce")
    long_distance_monthly = pd.to_numeric(df["Avg Monthly Long Distance Charges"], errors="coerce")
    long_distance_total = pd.to_numeric(df["Total Long Distance Charges"], errors="coerce")
    extra_data_charges = pd.to_numeric(df["Total Extra Data Charges"], errors="coerce")
    gb_download = pd.to_numeric(df["Avg Monthly GB Download"], errors="coerce")
    phone_service = pd.to_numeric(df["Phone Service"], errors="coerce").fillna(0.0)
    internet_service = pd.to_numeric(df["Internet Service"], errors="coerce").fillna(0.0)
    multiple_lines = pd.to_numeric(df["Multiple Lines"], errors="coerce").fillna(0.0)
    tenure = pd.to_numeric(df["Tenure in Months"], errors="coerce")
    service_count = (
        phone_service
        + internet_service
        + multiple_lines
        + pd.to_numeric(df["Online Security"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["Online Backup"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["Device Protection Plan"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["Premium Tech Support"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["Streaming TV"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["Streaming Movies"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["Streaming Music"], errors="coerce").fillna(0.0)
    ).clip(lower=1.0)

    X["monthly_revenue"] = monthly_charge
    X["mobile_revenue"] = long_distance_monthly.fillna(0.0) + extra_data_charges.fillna(0.0) / tenure.replace(0, np.nan)
    X["fixed_revenue"] = (monthly_charge - long_distance_monthly.fillna(0.0)).clip(lower=0.0)
    X["total_revenue"] = total_revenue.fillna(total_charges)
    X["arpu"] = monthly_charge
    X["tenure"] = tenure
    X["total_subs"] = service_count
    X["active_subs"] = service_count
    X["inactive_subs"] = 0.0
    X["suspended_subs"] = 0.0
    X["active_ratio"] = 1.0
    X["inactive_ratio"] = 0.0
    X["suspended_ratio"] = 0.0
    X["usage_minutes"] = gb_download
    X["usage_seconds"] = gb_download * 60.0
    X["usage_frequency"] = gb_download
    X["sms_frequency"] = 0.0
    X["distinct_contacts"] = pd.to_numeric(df["Number of Referrals"], errors="coerce").fillna(0.0)
    X["dropped_calls"] = 0.0
    X["blocked_calls"] = 0.0
    X["unanswered_calls"] = 0.0
    X["customer_care_calls"] = 0.0
    X["revenue_change"] = total_revenue.fillna(0.0) - total_charges.fillna(0.0) - long_distance_total.fillna(0.0)
    X["minutes_change"] = gb_download
    X["complaints"] = 0.0
    X["retention_calls"] = 0.0
    X["contract_risk"] = df["Contract"].astype(str).str.lower().eq("month-to-month").astype(float)
    X["service_status"] = (phone_service.gt(0) | internet_service.gt(0)).astype(float)
    X["customer_value"] = total_revenue.fillna(total_charges)
    X["age"] = pd.to_numeric(df["Age"], errors="coerce")
    X["zip_code"] = pd.to_numeric(df["Zip Code"], errors="coerce")
    X["segment_code"] = pd.factorize(
        df["Contract"].astype(str)
        + "_"
        + df["Internet Type"].astype(str)
        + "_"
        + df["Payment Method"].astype(str)
        + "_"
        + df["Offer"].astype(str)
    )[0]
    X["dataset_code"] = 10
    y = pd.to_numeric(df["Churn"], errors="coerce").fillna(0).astype(int)
    return X[COMMON_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0), y


def load_cell2cell() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(raw_data_path("cell2cell_train.csv"))
    X = blank_frame(df.index)
    X["monthly_revenue"] = df["MonthlyRevenue"]
    X["mobile_revenue"] = df["MonthlyRevenue"]
    X["total_revenue"] = df["TotalRecurringCharge"] * df["MonthsInService"]
    X["arpu"] = df["MonthlyRevenue"]
    X["tenure"] = df["MonthsInService"]
    X["total_subs"] = df["UniqueSubs"]
    X["active_subs"] = df["ActiveSubs"]
    X["active_ratio"] = safe_divide(df["ActiveSubs"], df["UniqueSubs"]).clip(0, 1)
    X["usage_minutes"] = df["MonthlyMinutes"]
    X["dropped_calls"] = df["DroppedCalls"]
    X["blocked_calls"] = df["BlockedCalls"]
    X["unanswered_calls"] = df["UnansweredCalls"]
    X["customer_care_calls"] = df["CustomerCareCalls"]
    X["revenue_change"] = df["PercChangeRevenues"]
    X["minutes_change"] = df["PercChangeMinutes"]
    X["retention_calls"] = df["RetentionCalls"]
    X["contract_risk"] = df["MadeCallToRetentionTeam"].map({"Yes": 1, "No": 0})
    X["customer_value"] = df["MonthlyRevenue"] * df["MonthsInService"]
    X["age"] = df[["AgeHH1", "AgeHH2"]].replace(0, np.nan).mean(axis=1)
    X["segment_code"] = pd.factorize(df["CreditRating"].astype(str) + "_" + df["PrizmCode"].astype(str))[0]
    X["dataset_code"] = 2
    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return X[COMMON_FEATURES], y


def load_iranian() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(raw_data_path("iranian_churn.csv"))
    df.columns = df.columns.str.replace("  ", " ").str.replace(" ", "_")
    X = blank_frame(df.index)
    X["monthly_revenue"] = df["Charge_Amount"]
    X["arpu"] = df["Charge_Amount"]
    X["tenure"] = df["Subscription_Length"]
    X["usage_seconds"] = df["Seconds_of_Use"]
    X["usage_frequency"] = df["Frequency_of_use"]
    X["sms_frequency"] = df["Frequency_of_SMS"]
    X["distinct_contacts"] = df["Distinct_Called_Numbers"]
    X["complaints"] = df["Complains"]
    X["service_status"] = df["Status"]
    X["customer_value"] = df["Customer_Value"]
    X["age"] = df["Age"]
    X["segment_code"] = df["Age_Group"] * 10 + df["Tariff_Plan"]
    X["dataset_code"] = 3
    y = df["Churn"].astype(int)
    return X[COMMON_FEATURES], y


def load_bigml() -> tuple[pd.DataFrame, pd.Series]:
    paths = [REPO_ROOT / "churn-bigml-80.csv", REPO_ROOT / "churn-bigml-20.csv"]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame(columns=COMMON_FEATURES), pd.Series(dtype=int)

    df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    X = blank_frame(df.index)
    day_minutes = pd.to_numeric(df["Total day minutes"], errors="coerce")
    eve_minutes = pd.to_numeric(df["Total eve minutes"], errors="coerce")
    night_minutes = pd.to_numeric(df["Total night minutes"], errors="coerce")
    intl_minutes = pd.to_numeric(df["Total intl minutes"], errors="coerce")
    day_calls = pd.to_numeric(df["Total day calls"], errors="coerce")
    eve_calls = pd.to_numeric(df["Total eve calls"], errors="coerce")
    night_calls = pd.to_numeric(df["Total night calls"], errors="coerce")
    intl_calls = pd.to_numeric(df["Total intl calls"], errors="coerce")
    day_charge = pd.to_numeric(df["Total day charge"], errors="coerce")
    eve_charge = pd.to_numeric(df["Total eve charge"], errors="coerce")
    night_charge = pd.to_numeric(df["Total night charge"], errors="coerce")
    intl_charge = pd.to_numeric(df["Total intl charge"], errors="coerce")

    X["monthly_revenue"] = day_charge + eve_charge + night_charge + intl_charge
    X["mobile_revenue"] = X["monthly_revenue"]
    X["total_revenue"] = X["monthly_revenue"]
    X["arpu"] = X["monthly_revenue"]
    X["tenure"] = pd.to_numeric(df["Account length"], errors="coerce")
    X["usage_minutes"] = day_minutes + eve_minutes + night_minutes + intl_minutes
    X["usage_frequency"] = day_calls + eve_calls + night_calls + intl_calls
    X["customer_care_calls"] = pd.to_numeric(df["Customer service calls"], errors="coerce")
    X["complaints"] = X["customer_care_calls"]
    X["contract_risk"] = (df["International plan"] == "Yes").astype(int)
    X["service_status"] = (df["Voice mail plan"] == "Yes").astype(int)
    X["customer_value"] = X["monthly_revenue"] * X["tenure"]
    X["zip_code"] = pd.to_numeric(df["Area code"], errors="coerce")
    X["segment_code"] = pd.factorize(df["State"].astype(str))[0]
    X["dataset_code"] = 4
    y = df["Churn"].astype(int)
    return X[COMMON_FEATURES], y


def load_india_monthly(high_value_only: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    path = raw_data_path("telecom_churn_data.csv")
    if not path.exists():
        return pd.DataFrame(columns=COMMON_FEATURES), pd.Series(dtype=int)

    df = pd.read_csv(path)
    usage_9_cols = ["total_ic_mou_9", "total_og_mou_9", "vol_2g_mb_9", "vol_3g_mb_9"]
    churn = df[usage_9_cols].fillna(0).sum(axis=1).eq(0).astype(int)

    if high_value_only:
        good_month_recharge = df[["total_rech_amt_6", "total_rech_amt_7"]].mean(axis=1)
        df = df.loc[good_month_recharge >= good_month_recharge.quantile(0.7)].copy()
        churn = churn.loc[df.index]

    df = df.reset_index(drop=True)
    churn = churn.reset_index(drop=True)
    X = blank_frame(df.index)

    arpu_6 = pd.to_numeric(df["arpu_6"], errors="coerce")
    arpu_7 = pd.to_numeric(df["arpu_7"], errors="coerce")
    arpu_8 = pd.to_numeric(df["arpu_8"], errors="coerce")
    recharge_6 = pd.to_numeric(df["total_rech_amt_6"], errors="coerce")
    recharge_7 = pd.to_numeric(df["total_rech_amt_7"], errors="coerce")
    recharge_8 = pd.to_numeric(df["total_rech_amt_8"], errors="coerce")
    data_recharge_8 = pd.to_numeric(df["av_rech_amt_data_8"], errors="coerce")
    voice_6 = pd.to_numeric(df["total_ic_mou_6"], errors="coerce") + pd.to_numeric(df["total_og_mou_6"], errors="coerce")
    voice_7 = pd.to_numeric(df["total_ic_mou_7"], errors="coerce") + pd.to_numeric(df["total_og_mou_7"], errors="coerce")
    voice_8 = pd.to_numeric(df["total_ic_mou_8"], errors="coerce") + pd.to_numeric(df["total_og_mou_8"], errors="coerce")
    data_8 = pd.to_numeric(df["vol_2g_mb_8"], errors="coerce") + pd.to_numeric(df["vol_3g_mb_8"], errors="coerce")
    recharge_count_8 = pd.to_numeric(df["total_rech_num_8"], errors="coerce")
    data_recharge_count_8 = (
        pd.to_numeric(df["count_rech_2g_8"], errors="coerce")
        + pd.to_numeric(df["count_rech_3g_8"], errors="coerce")
    )
    active_8 = (voice_8.fillna(0) + data_8.fillna(0) + recharge_8.fillna(0)).gt(0).astype(float)

    X["monthly_revenue"] = arpu_8
    X["mobile_revenue"] = recharge_8.fillna(0) + data_recharge_8.fillna(0)
    X["total_revenue"] = recharge_6.fillna(0) + recharge_7.fillna(0) + recharge_8.fillna(0)
    X["arpu"] = arpu_8
    X["tenure"] = pd.to_numeric(df["aon"], errors="coerce") / 30.0
    X["total_subs"] = 1.0
    X["active_subs"] = active_8
    X["inactive_subs"] = 1.0 - active_8
    X["active_ratio"] = active_8
    X["inactive_ratio"] = 1.0 - active_8
    X["usage_minutes"] = voice_8
    X["usage_frequency"] = recharge_count_8.fillna(0) + data_recharge_count_8.fillna(0)
    X["revenue_change"] = arpu_8 - pd.concat([arpu_6, arpu_7], axis=1).mean(axis=1)
    X["minutes_change"] = voice_8 - pd.concat([voice_6, voice_7], axis=1).mean(axis=1)
    X["service_status"] = pd.to_numeric(df["fb_user_8"], errors="coerce").fillna(0)
    X["customer_value"] = X["total_revenue"]
    X["zip_code"] = pd.to_numeric(df["circle_id"], errors="coerce")
    X["segment_code"] = pd.to_numeric(df["circle_id"], errors="coerce")
    X["dataset_code"] = 5 if not high_value_only else 6
    return X[COMMON_FEATURES], churn.astype(int)


def load_kdd_orange_small(max_rows: int = 50_000) -> tuple[pd.DataFrame, pd.Series]:
    data_path = raw_data_path("kdd2009_orange_small.data")
    label_path = raw_data_path("kdd2009_orange_small_churn.labels")
    if not data_path.exists() or not label_path.exists():
        return pd.DataFrame(columns=COMMON_FEATURES), pd.Series(dtype=int)

    df = pd.read_csv(data_path, sep="\t", na_values=[""])
    labels = pd.read_csv(label_path, header=None, names=["label"])["label"].map({1: 1, -1: 0})
    if len(df) != len(labels):
        raise ValueError(f"KDD Orange row mismatch: data={len(df)} labels={len(labels)}")

    valid = labels.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = labels.loc[valid].astype(int).reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        sampled_idx = (
            y.to_frame("churn")
            .groupby("churn", group_keys=False)
            .sample(frac=max_rows / len(df), random_state=RANDOM_STATE)
            .index
        )
        df = df.loc[sampled_idx].reset_index(drop=True)
        y = y.loc[sampled_idx].reset_index(drop=True)

    numeric = df.iloc[:, :190].apply(pd.to_numeric, errors="coerce")
    categorical = df.iloc[:, 190:].fillna("__missing__").astype(str)
    numeric_filled = numeric.fillna(0.0)
    numeric_missing = numeric.isna()

    first_half_mean = numeric.iloc[:, :95].mean(axis=1).fillna(0.0)
    second_half_mean = numeric.iloc[:, 95:].mean(axis=1).fillna(0.0)
    early_usage = numeric.iloc[:, 100:145].sum(axis=1).fillna(0.0)
    late_usage = numeric.iloc[:, 145:190].sum(axis=1).fillna(0.0)
    row_nonzero = numeric_filled.ne(0).sum(axis=1).astype(float)
    row_missing = numeric_missing.sum(axis=1).astype(float)
    row_missing_ratio = row_missing / max(numeric.shape[1], 1)
    cat_missing = categorical.eq("__missing__").sum(axis=1).astype(float)
    cat_unique = categorical.nunique(axis=1).astype(float)

    X = blank_frame(df.index)
    X["monthly_revenue"] = numeric.mean(axis=1).fillna(0.0)
    X["mobile_revenue"] = numeric.iloc[:, :50].mean(axis=1).fillna(0.0)
    X["fixed_revenue"] = numeric.iloc[:, 50:100].mean(axis=1).fillna(0.0)
    X["total_revenue"] = numeric_filled.sum(axis=1)
    X["arpu"] = numeric.median(axis=1).fillna(0.0)
    X["tenure"] = row_nonzero
    X["total_subs"] = 1.0
    X["active_subs"] = (row_missing_ratio < 0.5).astype(float)
    X["inactive_subs"] = (row_missing_ratio >= 0.5).astype(float)
    X["suspended_subs"] = 0.0
    X["active_ratio"] = 1.0 - row_missing_ratio
    X["inactive_ratio"] = row_missing_ratio
    X["suspended_ratio"] = 0.0
    X["usage_minutes"] = late_usage
    X["usage_seconds"] = late_usage * 60.0
    X["usage_frequency"] = row_nonzero
    X["sms_frequency"] = cat_unique
    X["distinct_contacts"] = cat_unique
    X["dropped_calls"] = row_missing
    X["blocked_calls"] = cat_missing
    X["unanswered_calls"] = numeric.iloc[:, 145:190].fillna(0.0).eq(0).sum(axis=1).astype(float)
    X["customer_care_calls"] = cat_missing
    X["revenue_change"] = second_half_mean - first_half_mean
    X["minutes_change"] = late_usage - early_usage
    X["complaints"] = cat_missing
    X["retention_calls"] = 0.0
    X["contract_risk"] = (row_missing_ratio > row_missing_ratio.median()).astype(float)
    X["service_status"] = 1.0 - row_missing_ratio
    X["customer_value"] = numeric_filled.abs().sum(axis=1)
    X["age"] = 0.0
    X["zip_code"] = pd.factorize(categorical.iloc[:, 0])[0]
    X["segment_code"] = pd.factorize(
        categorical.iloc[:, 0] + "_" + categorical.iloc[:, 1] + "_" + categorical.iloc[:, 2]
    )[0]
    X["dataset_code"] = 8
    return X[COMMON_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0), y


def load_orange_uplift() -> tuple[pd.DataFrame, pd.Series]:
    path = raw_data_path("churn_uplift_mlg.parquet")
    if not path.exists():
        return pd.DataFrame(columns=COMMON_FEATURES), pd.Series(dtype=int)

    df = pd.read_parquet(path)
    if "y" not in df.columns:
        raise ValueError(f"Orange uplift target column y not found: {path}")

    pc_cols = [col for col in df.columns if col.startswith("PC")]
    factor_cols = [col for col in df.columns if col.startswith("FACTOR")]
    pc = df[pc_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    pc_filled = pc.fillna(0.0)

    def block_mean(start: int, stop: int) -> pd.Series:
        block = pc.iloc[:, start:stop]
        if block.shape[1] == 0:
            return pd.Series(0.0, index=df.index)
        return block.mean(axis=1).fillna(0.0)

    def block_sum(start: int, stop: int) -> pd.Series:
        block = pc_filled.iloc[:, start:stop]
        if block.shape[1] == 0:
            return pd.Series(0.0, index=df.index)
        return block.sum(axis=1)

    def sigmoid(values: pd.Series) -> pd.Series:
        return pd.Series(1.0 / (1.0 + np.exp(-np.clip(values, -20, 20))), index=df.index)

    factor_frame = (
        df[factor_cols].fillna("__missing__").astype(str)
        if factor_cols
        else pd.DataFrame(index=df.index)
    )
    if factor_cols:
        segment_values = factor_frame.iloc[:, : min(3, len(factor_cols))].agg("_".join, axis=1)
        zip_code = pd.factorize(factor_frame.iloc[:, 0])[0]
        segment_code = pd.factorize(segment_values)[0]
        factor_missing = factor_frame.eq("__missing__").sum(axis=1).astype(float)
    else:
        zip_code = np.zeros(len(df), dtype=float)
        segment_code = np.zeros(len(df), dtype=float)
        factor_missing = pd.Series(0.0, index=df.index)

    row_activity = pc_filled.abs().gt(1e-6).sum(axis=1).astype(float)
    active_ratio = sigmoid(block_mean(40, 80)).clip(0, 1)
    usage_minutes = block_sum(40, 80)
    late_usage = block_sum(120, 160)
    early_value = block_mean(0, 40)
    late_value = block_mean(80, 120)

    X = blank_frame(df.index)
    X["monthly_revenue"] = block_mean(0, 10)
    X["mobile_revenue"] = block_mean(10, 20)
    X["fixed_revenue"] = block_mean(20, 30)
    X["total_revenue"] = block_sum(0, 40)
    X["arpu"] = early_value
    X["tenure"] = row_activity
    X["total_subs"] = 1.0
    X["active_subs"] = active_ratio
    X["inactive_subs"] = 1.0 - active_ratio
    X["suspended_subs"] = 0.0
    X["active_ratio"] = active_ratio
    X["inactive_ratio"] = 1.0 - active_ratio
    X["suspended_ratio"] = 0.0
    X["usage_minutes"] = usage_minutes
    X["usage_seconds"] = usage_minutes * 60.0
    X["usage_frequency"] = block_sum(80, 120).abs()
    X["sms_frequency"] = block_sum(120, 135).abs()
    X["distinct_contacts"] = row_activity
    X["dropped_calls"] = pc_filled.iloc[:, 120:160].lt(0).sum(axis=1).astype(float)
    X["blocked_calls"] = block_mean(135, 145).abs()
    X["unanswered_calls"] = block_mean(145, 160).abs()
    X["customer_care_calls"] = factor_missing
    X["revenue_change"] = late_value - early_value
    X["minutes_change"] = late_usage - usage_minutes
    X["complaints"] = factor_missing
    X["retention_calls"] = 0.0
    X["contract_risk"] = (active_ratio < 0.5).astype(float)
    X["service_status"] = active_ratio
    X["customer_value"] = pc_filled.abs().sum(axis=1)
    X["age"] = block_mean(30, 40).abs()
    X["zip_code"] = zip_code
    X["segment_code"] = segment_code
    X["dataset_code"] = 9

    y = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)
    return X[COMMON_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0), y


def load_kkbox_activity(max_rows: int = 100_000) -> tuple[pd.DataFrame, pd.Series]:
    path = processed_data_path("kkbox_activity_common_features.csv")
    if not path.exists():
        return pd.DataFrame(columns=COMMON_FEATURES), pd.Series(dtype=int)

    df = pd.read_csv(path)
    if max_rows and len(df) > max_rows:
        frac = max_rows / len(df)
        df = (
            df.groupby("churn", group_keys=False)
            .sample(frac=frac, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
    y = df["churn"].astype(int)
    return df[COMMON_FEATURES], y


def build_source_groups() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    sources = {
        "ibm": load_ibm(),
        "hf_telco": load_hf_telco(),
        "cell": load_cell2cell(),
        "iran": load_iranian(),
        "bigml": load_bigml(),
        "india_monthly": load_india_monthly(high_value_only=False),
        "india_monthly_hv": load_india_monthly(high_value_only=True),
        "kdd_orange": load_kdd_orange_small(),
        "orange_uplift": load_orange_uplift(),
        "kkbox_activity": load_kkbox_activity(),
    }
    groups: dict[str, tuple[pd.DataFrame, pd.Series]] = {
        "baza_only": (pd.DataFrame(columns=COMMON_FEATURES), pd.Series(dtype=int)),
    }
    for name, loaded in sources.items():
        if len(loaded[0]) > 0:
            groups[name] = loaded
    combos = {
        "ibm_cell": ["ibm", "cell"],
        "all": ["ibm", "cell", "iran"],
        "all_bigml": ["ibm", "cell", "iran", "bigml"],
        "cell_bigml": ["cell", "bigml"],
        "ibm_cell_bigml": ["ibm", "cell", "bigml"],
        "cell_india": ["cell", "india_monthly"],
        "cell_india_hv": ["cell", "india_monthly_hv"],
        "ibm_cell_india": ["ibm", "cell", "india_monthly"],
        "ibm_cell_india_hv": ["ibm", "cell", "india_monthly_hv"],
        "all_india": ["ibm", "cell", "iran", "bigml", "india_monthly"],
        "all_india_hv": ["ibm", "cell", "iran", "bigml", "india_monthly_hv"],
        "cell_kdd": ["cell", "kdd_orange"],
        "ibm_cell_kdd": ["ibm", "cell", "kdd_orange"],
        "all_kdd": ["ibm", "cell", "iran", "bigml", "kdd_orange"],
        "cell_orange": ["cell", "orange_uplift"],
        "ibm_cell_orange": ["ibm", "cell", "orange_uplift"],
        "all_orange": ["ibm", "cell", "iran", "bigml", "orange_uplift"],
        "hf_cell": ["hf_telco", "cell"],
        "hf_orange": ["hf_telco", "orange_uplift"],
        "hf_cell_orange": ["hf_telco", "cell", "orange_uplift"],
        "hf_all": ["hf_telco", "ibm", "cell", "iran", "bigml"],
        "hf_all_orange": ["hf_telco", "ibm", "cell", "iran", "bigml", "orange_uplift"],
        "cell_kkbox": ["cell", "kkbox_activity"],
        "ibm_cell_kkbox": ["ibm", "cell", "kkbox_activity"],
        "all_kkbox": ["ibm", "cell", "iran", "bigml", "kkbox_activity"],
        "orange_kkbox": ["orange_uplift", "kkbox_activity"],
        "all_orange_kkbox": ["ibm", "cell", "iran", "bigml", "orange_uplift", "kkbox_activity"],
    }
    for combo_name, names in combos.items():
        names = [name for name in names if name in sources and len(sources[name][0]) > 0]
        if not names:
            continue
        groups[combo_name] = (
            pd.concat([sources[name][0] for name in names], ignore_index=True),
            pd.concat([sources[name][1] for name in names], ignore_index=True),
        )
    return groups


def build_candidates() -> list[TransferCandidate]:
    source_groups = [
        "baza_only",
        "cell",
        "bigml",
        "india_monthly_hv",
        "india_monthly",
        "kdd_orange",
        "orange_uplift",
        "hf_telco",
        "hf_cell",
        "hf_orange",
        "hf_cell_orange",
        "hf_all",
        "hf_all_orange",
        "kkbox_activity",
        "cell_kdd",
        "ibm_cell_kdd",
        "all_kdd",
        "cell_orange",
        "ibm_cell_orange",
        "all_orange",
        "cell_kkbox",
        "ibm_cell_kkbox",
        "all_kkbox",
        "orange_kkbox",
        "all_orange_kkbox",
        "cell_india_hv",
        "cell_india",
        "ibm_cell_india_hv",
        "ibm_cell_india",
        "all_india_hv",
        "all_india",
        "cell_bigml",
        "all_bigml",
        "ibm_cell_bigml",
        "all",
        "ibm_cell",
        "ibm",
        "iran",
    ]
    source_weights = [0.05, 0.1, 0.25, 0.5, 1.0]
    models: dict[str, Any] = {
        "LR_balanced": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        "LGBM_balanced": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            class_weight="balanced",
            num_leaves=15,
            min_child_samples=20,
            verbose=-1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "LGBM_regularized": LGBMClassifier(
            n_estimators=700,
            learning_rate=0.02,
            class_weight="balanced",
            num_leaves=7,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            verbose=-1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGB_balanced": HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.03,
            class_weight="balanced",
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
        "XGB_weighted": XGBClassifier(
            n_estimators=500,
            max_depth=2,
            learning_rate=0.03,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "RF_balanced": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    candidates: list[TransferCandidate] = []
    for group in source_groups:
        for weight in source_weights:
            if group == "baza_only" and weight != source_weights[0]:
                continue
            for model_name, estimator in models.items():
                candidates.append(TransferCandidate(group, weight, model_name, estimator))
    return candidates


def parse_csv_arg(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def parse_float_csv_arg(value: str | None) -> set[float] | None:
    items = parse_csv_arg(value)
    if items is None:
        return None
    return {float(item) for item in items}


def make_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", clone(estimator)),
    ])


def best_f1_threshold(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1_values))
    return float(thresholds[idx]), float(f1_values[idx])


def evaluate_candidate(
    candidate: TransferCandidate,
    source_X: pd.DataFrame,
    source_y: pd.Series,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, Any], ThresholdClassifier]:
    train_X = pd.concat([X_train, source_X], ignore_index=True)
    train_y = pd.concat([y_train.reset_index(drop=True), source_y.reset_index(drop=True)], ignore_index=True)
    sample_weight = np.concatenate([
        np.ones(len(X_train)),
        np.full(len(source_X), candidate.source_weight),
    ])

    pipeline = make_pipeline(candidate.estimator)
    if candidate.model_name == "XGB_weighted":
        pipeline.set_params(model__scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)))

    started = time.perf_counter()
    fit_kwargs = {"model__sample_weight": sample_weight} if len(source_X) else {}
    pipeline.fit(train_X, train_y, **fit_kwargs)

    val_scores = pipeline.predict_proba(X_val)[:, 1]
    test_scores = pipeline.predict_proba(X_test)[:, 1]
    threshold, val_best_f1 = best_f1_threshold(y_val, val_scores)
    tuned = ThresholdClassifier(pipeline, threshold)
    tuned_pred = tuned.predict(X_test)
    default_pred = pipeline.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, tuned_pred, labels=[0, 1]).ravel()

    row = {
        "source_group": candidate.source_group,
        "source_weight": candidate.source_weight,
        "model": candidate.model_name,
        "source_rows": int(len(source_X)),
        "threshold": threshold,
        "val_best_f1": val_best_f1,
        "test_tuned_f1": float(f1_score(y_test, tuned_pred, zero_division=0)),
        "test_default_f1": float(f1_score(y_test, default_pred, zero_division=0)),
        "test_precision": float(precision_score(y_test, tuned_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, tuned_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "test_average_precision": float(average_precision_score(y_test, test_scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "train_seconds": round(time.perf_counter() - started, 3),
    }
    return row, tuned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-f1", type=float, default=0.6)
    parser.add_argument("--table-out", type=Path, default=result_path("baza_transfer_benchmark.csv"))
    parser.add_argument("--json-out", type=Path, default=result_path("baza_transfer_benchmark_summary.json"))
    parser.add_argument("--model-out", type=Path, default=model_path("baza_transfer_best_model.joblib"))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--groups", type=str, default=None, help="Comma-separated source groups to benchmark")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names to benchmark")
    parser.add_argument("--weights", type=str, default=None, help="Comma-separated auxiliary source weights")
    args = parser.parse_args()

    X_baza, y_baza = load_baza()
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_baza, y_baza, test_size=0.2, random_state=RANDOM_STATE, stratify=y_baza
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
    )

    source_groups = build_source_groups()
    candidates = build_candidates()
    allowed_groups = parse_csv_arg(args.groups)
    allowed_models = parse_csv_arg(args.models)
    allowed_weights = parse_float_csv_arg(args.weights)
    if allowed_groups is not None:
        candidates = [candidate for candidate in candidates if candidate.source_group in allowed_groups]
    if allowed_models is not None:
        candidates = [candidate for candidate in candidates if candidate.model_name in allowed_models]
    if allowed_weights is not None:
        candidates = [candidate for candidate in candidates if candidate.source_weight in allowed_weights]
    if not candidates:
        raise ValueError("No transfer candidates matched the requested filters.")

    print(
        f"Baza fixed split: train={len(X_train)} val={len(X_val)} test={len(X_test)} "
        f"churn_rate={y_baza.mean():.2%}",
        flush=True,
    )
    print("Source groups:", flush=True)
    for name, (source_X, source_y) in source_groups.items():
        if allowed_groups is not None and name not in allowed_groups:
            continue
        churn_rate = float(source_y.mean()) if len(source_y) else 0.0
        print(f"  {name}: rows={len(source_X)} churn_rate={churn_rate:.2%}", flush=True)

    rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, float, str], ThresholdClassifier] = {}
    for idx, candidate in enumerate(candidates, start=1):
        source_X, source_y = source_groups[candidate.source_group]
        print(
            f"[{idx:03d}/{len(candidates)}] {candidate.source_group} "
            f"w={candidate.source_weight} {candidate.model_name}",
            flush=True,
        )
        row, tuned = evaluate_candidate(
            candidate,
            source_X,
            source_y,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )
        rows.append(row)
        fitted[(candidate.source_group, candidate.source_weight, candidate.model_name)] = tuned
        print(
            f"    f1={row['test_tuned_f1']:.4f} default={row['test_default_f1']:.4f} "
            f"precision={row['test_precision']:.4f} recall={row['test_recall']:.4f} "
            f"auc={row['test_roc_auc']:.4f}",
            flush=True,
        )

    table = pd.DataFrame(rows).sort_values(["test_tuned_f1", "test_roc_auc"], ascending=[False, False])
    args.table_out = args.table_out if args.table_out.is_absolute() else REPO_ROOT / args.table_out
    args.json_out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    args.model_out = args.model_out if args.model_out.is_absolute() else REPO_ROOT / args.model_out
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(args.table_out, index=False, encoding="utf-8-sig")
    best = table.iloc[0].to_dict()
    best_key = (best["source_group"], float(best["source_weight"]), best["model"])
    joblib.dump(fitted[best_key], args.model_out)

    baseline_rows = table[table["source_group"] == "baza_only"]
    baseline = baseline_rows.iloc[0].to_dict() if len(baseline_rows) else None
    absolute_f1_gain = (
        float(best["test_tuned_f1"] - baseline["test_tuned_f1"])
        if baseline is not None
        else None
    )
    relative_f1_gain_pct = (
        float(100 * (best["test_tuned_f1"] - baseline["test_tuned_f1"]) / max(baseline["test_tuned_f1"], 1e-12))
        if baseline is not None
        else None
    )
    summary = {
        "target_f1": args.target_f1,
        "target_reached": bool((table["test_tuned_f1"] >= args.target_f1).any()),
        "target_reached_count": int((table["test_tuned_f1"] >= args.target_f1).sum()),
        "baza_rows": int(len(X_baza)),
        "baza_churn_rate": float(y_baza.mean()),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "features": COMMON_FEATURES,
        "baseline": baseline,
        "best": best,
        "absolute_f1_gain": absolute_f1_gain,
        "relative_f1_gain_pct": relative_f1_gain_pct,
        "table_path": str(args.table_out),
        "best_model_path": str(args.model_out),
    }
    args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    cols = [
        "source_group",
        "source_weight",
        "model",
        "test_tuned_f1",
        "test_default_f1",
        "test_precision",
        "test_recall",
        "test_roc_auc",
        "threshold",
    ]
    print("\n=== Baza Transfer Top Models ===", flush=True)
    print(table[cols].head(args.top).to_string(index=False), flush=True)
    if baseline is not None:
        print(f"\nBaseline Baza-only F1: {baseline['test_tuned_f1']:.4f}", flush=True)
    print(f"Best transfer F1: {best['test_tuned_f1']:.4f}", flush=True)
    if absolute_f1_gain is not None:
        print(f"Absolute gain: {absolute_f1_gain:+.4f}", flush=True)
    print(f"Saved table: {args.table_out}", flush=True)
    print(f"Saved summary: {args.json_out}", flush=True)
    print(f"Saved best Baza transfer model: {args.model_out}", flush=True)


if __name__ == "__main__":
    main()
