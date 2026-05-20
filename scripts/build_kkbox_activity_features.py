#!/usr/bin/env python3
"""Build KKBox activity features for auxiliary Baza transfer training.

The KKBox files use ``msno`` identifiers and cannot be joined to Baza ``PID``
rows. This script therefore creates a separate source-domain dataset mapped
into the common telco feature schema used by ``train_baza_transfer.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_baza_transfer import COMMON_FEATURES
from src.utils.helpers import processed_data_path, raw_data_path

RANDOM_STATE = 42
CUTOFF_DATE = 20170331
LAST_7_START = 20170325
LAST_14_START = 20170318


def combine_grouped(
    current: pd.DataFrame | None,
    update: pd.DataFrame,
    sum_cols: Iterable[str],
    min_cols: Iterable[str] = (),
    max_cols: Iterable[str] = (),
) -> pd.DataFrame:
    if current is None:
        return update
    combined = pd.concat([current, update], axis=0, copy=False)
    agg: dict[str, str] = {}
    agg.update({col: "sum" for col in sum_cols})
    agg.update({col: "min" for col in min_cols})
    agg.update({col: "max" for col in max_cols})
    return combined.groupby(level=0, sort=False).agg(agg)


def aggregate_user_logs(path: Path, labeled_msnos: set[str], chunksize: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    sum_cols = [
        "log_days",
        "total_secs",
        "last7_secs",
        "last14_secs",
        "total_plays",
        "complete_plays",
        "short_plays",
        "num_unq_sum",
        "skip_rate_sum",
        "completion_rate_sum",
        "diversity_score_sum",
    ]
    min_cols = ["first_log_date"]
    max_cols = ["last_log_date"]
    aggregated: pd.DataFrame | None = None
    usecols = ["msno", "date", "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"]

    for idx, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), start=1):
        chunk = chunk[chunk["msno"].isin(labeled_msnos)]
        chunk = chunk[chunk["date"].le(CUTOFF_DATE)]
        if chunk.empty:
            continue
        plays = chunk[["num_25", "num_50", "num_75", "num_985", "num_100"]].sum(axis=1)
        safe_unq = chunk["num_unq"].replace(0, np.nan)
        safe_plays = plays.replace(0, np.nan)
        chunk["skip_rate_sum"] = (chunk["num_25"] / safe_unq).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        chunk["completion_rate_sum"] = (chunk["num_100"] / safe_unq).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        chunk["diversity_score_sum"] = (chunk["num_unq"] / safe_plays).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        chunk["log_days"] = 1
        chunk["total_plays"] = plays
        chunk["complete_plays"] = chunk["num_100"]
        chunk["short_plays"] = chunk["num_25"]
        chunk["last7_secs"] = np.where(chunk["date"].ge(LAST_7_START), chunk["total_secs"], 0.0)
        chunk["last14_secs"] = np.where(chunk["date"].ge(LAST_14_START), chunk["total_secs"], 0.0)
        chunk["first_log_date"] = chunk["date"]
        chunk["last_log_date"] = chunk["date"]
        chunk["num_unq_sum"] = chunk["num_unq"]
        grouped = chunk.groupby("msno", sort=False)[sum_cols + min_cols + max_cols].agg(
            {**{col: "sum" for col in sum_cols}, **{col: "min" for col in min_cols}, **{col: "max" for col in max_cols}}
        )
        aggregated = combine_grouped(aggregated, grouped, sum_cols, min_cols, max_cols)
        print(f"[logs] chunk={idx} groups={len(grouped)} total_groups={len(aggregated)}", flush=True)

    return aggregated if aggregated is not None else pd.DataFrame()


def aggregate_transactions(paths: list[Path], labeled_msnos: set[str], chunksize: int) -> pd.DataFrame:
    sum_cols = [
        "txn_count",
        "plan_days_sum",
        "plan_price_sum",
        "actual_paid_sum",
        "auto_renew_sum",
        "cancel_sum",
    ]
    min_cols = ["first_txn_date"]
    max_cols = ["last_txn_date", "max_expire_date"]
    aggregated: pd.DataFrame | None = None
    usecols = [
        "msno",
        "payment_plan_days",
        "plan_list_price",
        "actual_amount_paid",
        "is_auto_renew",
        "transaction_date",
        "membership_expire_date",
        "is_cancel",
    ]

    for path in paths:
        if not path.exists():
            continue
        for idx, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), start=1):
            chunk = chunk[chunk["msno"].isin(labeled_msnos)]
            chunk = chunk[chunk["transaction_date"].le(CUTOFF_DATE)]
            if chunk.empty:
                continue
            chunk["txn_count"] = 1
            chunk["plan_days_sum"] = chunk["payment_plan_days"]
            chunk["plan_price_sum"] = chunk["plan_list_price"]
            chunk["actual_paid_sum"] = chunk["actual_amount_paid"]
            chunk["auto_renew_sum"] = chunk["is_auto_renew"]
            chunk["cancel_sum"] = chunk["is_cancel"]
            chunk["first_txn_date"] = chunk["transaction_date"]
            chunk["last_txn_date"] = chunk["transaction_date"]
            chunk["max_expire_date"] = chunk["membership_expire_date"]
            grouped = chunk.groupby("msno", sort=False)[sum_cols + min_cols + max_cols].agg(
                {**{col: "sum" for col in sum_cols}, **{col: "min" for col in min_cols}, **{col: "max" for col in max_cols}}
            )
            aggregated = combine_grouped(aggregated, grouped, sum_cols, min_cols, max_cols)
            print(
                f"[transactions:{path.name}] chunk={idx} groups={len(grouped)} "
                f"total_groups={len(aggregated)}",
                flush=True,
            )

    return aggregated if aggregated is not None else pd.DataFrame()


def load_members(path: Path, labeled_msnos: set[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    members = pd.read_csv(path, usecols=["msno", "city", "bd", "gender", "registered_via", "registration_init_time"])
    members = members[members["msno"].isin(labeled_msnos)].drop_duplicates("msno").set_index("msno")
    members["bd"] = pd.to_numeric(members["bd"], errors="coerce")
    members.loc[~members["bd"].between(1, 100), "bd"] = np.nan
    members["gender_code"] = pd.factorize(members["gender"].fillna("unknown").astype(str))[0]
    return members


def yyyymmdd_to_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def build_common_features(labels: pd.DataFrame, logs: pd.DataFrame, txns: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    base = labels.drop_duplicates("msno").set_index("msno")
    df = base.join(logs, how="left").join(txns, how="left").join(members, how="left")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    txn_count = df["txn_count"].replace(0, np.nan) if "txn_count" in df else pd.Series(np.nan, index=df.index)
    log_days = df["log_days"].replace(0, np.nan) if "log_days" in df else pd.Series(np.nan, index=df.index)
    monthly_revenue = df.get("actual_paid_sum", 0.0) / txn_count
    auto_renew_rate = df.get("auto_renew_sum", 0.0) / txn_count
    cancel_rate = df.get("cancel_sum", 0.0) / txn_count

    reg_date = yyyymmdd_to_datetime(df.get("registration_init_time", pd.Series(0, index=df.index)))
    cutoff = pd.Timestamp("2017-03-31")
    tenure_months = ((cutoff - reg_date).dt.days / 30.0).clip(lower=0)
    fallback_tenure = df.get("plan_days_sum", 0.0) / 30.0
    tenure_months = tenure_months.fillna(fallback_tenure)

    prior_secs = (df.get("total_secs", 0.0) - df.get("last7_secs", 0.0)).clip(lower=0)
    prior_days = (df.get("log_days", 0.0) - 7).clip(lower=1)
    recent_daily_secs = df.get("last7_secs", 0.0) / 7.0
    prior_daily_secs = prior_secs / prior_days
    active = df.get("log_days", 0.0).gt(0).astype(float)
    skip_rate_mean = (df.get("skip_rate_sum", 0.0) / log_days).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    completion_rate_mean = (df.get("completion_rate_sum", 0.0) / log_days).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    diversity_score_mean = (df.get("diversity_score_sum", 0.0) / log_days).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X = pd.DataFrame(index=df.index, columns=COMMON_FEATURES, dtype=float)
    X["monthly_revenue"] = monthly_revenue.fillna(0.0)
    X["mobile_revenue"] = monthly_revenue.fillna(0.0)
    X["fixed_revenue"] = 0.0
    X["total_revenue"] = df.get("actual_paid_sum", 0.0)
    X["arpu"] = monthly_revenue.fillna(0.0)
    X["tenure"] = tenure_months.fillna(0.0)
    X["total_subs"] = 1.0
    X["active_subs"] = active
    X["inactive_subs"] = 1.0 - active
    X["suspended_subs"] = 0.0
    X["active_ratio"] = active
    X["inactive_ratio"] = 1.0 - active
    X["suspended_ratio"] = 0.0
    X["revenue_per_active_sub"] = monthly_revenue.fillna(0.0)
    X["inactive_x_revenue"] = X["inactive_ratio"] * df.get("actual_paid_sum", 0.0)
    X["revenue_balance"] = 0.0
    X["usage_minutes"] = df.get("total_secs", 0.0) / 60.0
    X["usage_seconds"] = df.get("total_secs", 0.0)
    X["usage_frequency"] = df.get("total_plays", 0.0)
    X["sms_frequency"] = 0.0
    X["distinct_contacts"] = df.get("num_unq_sum", 0.0)
    X["skip_rate"] = skip_rate_mean
    X["completion_rate"] = completion_rate_mean
    X["diversity_score"] = diversity_score_mean
    X["dropped_calls"] = df.get("short_plays", 0.0)
    X["blocked_calls"] = 0.0
    X["unanswered_calls"] = 0.0
    X["customer_care_calls"] = 0.0
    X["revenue_change"] = X["monthly_revenue"] - (df.get("plan_price_sum", 0.0) / txn_count).fillna(0.0)
    X["minutes_change"] = (recent_daily_secs - prior_daily_secs).fillna(0.0) / 60.0
    X["complaints"] = 0.0
    X["retention_calls"] = df.get("cancel_sum", 0.0)
    X["contract_risk"] = ((cancel_rate.fillna(0.0) > 0) | (auto_renew_rate.fillna(0.0) < 0.5)).astype(float)
    X["service_status"] = auto_renew_rate.fillna(0.0)
    X["customer_value"] = df.get("actual_paid_sum", 0.0)
    X["age"] = df.get("bd", 0.0)
    X["zip_code"] = df.get("city", 0.0)
    X["segment_code"] = df.get("registered_via", 0.0) * 10 + df.get("gender_code", 0.0)
    X["dataset_code"] = 7
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X["churn"] = df["is_churn"].astype(int).values
    return X.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=raw_data_path("train_v2.csv"))
    parser.add_argument("--members", type=Path, default=raw_data_path("members_v3.csv"))
    parser.add_argument("--user-logs", type=Path, default=raw_data_path("user_logs_v2.csv"))
    parser.add_argument("--transactions", type=Path, nargs="*", default=[raw_data_path("transactions.csv"), raw_data_path("transactions_v2.csv")])
    parser.add_argument("--output", type=Path, default=processed_data_path("kkbox_activity_common_features.csv"))
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-output-rows", type=int, default=250_000)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    labeled_msnos = set(labels["msno"].astype(str))
    print(f"[labels] rows={len(labels)} churn_rate={labels['is_churn'].mean():.2%}", flush=True)

    logs = aggregate_user_logs(args.user_logs, labeled_msnos, args.chunksize)
    txns = aggregate_transactions(args.transactions, labeled_msnos, args.chunksize)
    members = load_members(args.members, labeled_msnos)
    features = build_common_features(labels, logs, txns, members)

    if args.max_output_rows and len(features) > args.max_output_rows:
        features = (
            features.groupby("churn", group_keys=False)
            .sample(frac=args.max_output_rows / len(features), random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

    args.output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        f"[done] rows={len(features)} churn_rate={features['churn'].mean():.2%} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
