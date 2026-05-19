#!/usr/bin/env python3
"""Build a KKBox customer-month churn dataset from raw behavior logs.

The output is a public-data substitute for the kind of internal telco dataset
needed for stronger churn modeling: usage windows, billing/payment history,
service cancellation signals, member attributes, and a next-window churn label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.helpers import processed_data_path, raw_data_path, result_path

RANDOM_STATE = 42
CUTOFF_DATE = 20170331
CUTOFF_TS = pd.Timestamp("2017-03-31")

WINDOW_STARTS = {
    "7d": 20170325,
    "30d": 20170302,
    "60d": 20170131,
    "90d": 20170101,
}


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


def yyyymmdd_to_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def aggregate_user_logs(path: Path, labeled_msnos: set[str], chunksize: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    usecols = ["msno", "date", "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"]
    sum_cols = ["log_days_total", "total_secs_all", "total_plays_all", "complete_plays_all", "short_plays_all", "num_unq_all"]
    for suffix in WINDOW_STARTS:
        sum_cols.extend([
            f"log_days_{suffix}",
            f"total_secs_{suffix}",
            f"total_plays_{suffix}",
            f"complete_plays_{suffix}",
            f"short_plays_{suffix}",
            f"num_unq_{suffix}",
        ])
    min_cols = ["first_log_date"]
    max_cols = ["last_log_date"]
    aggregated: pd.DataFrame | None = None

    for idx, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), start=1):
        chunk = chunk[chunk["msno"].isin(labeled_msnos)]
        chunk = chunk[chunk["date"].le(CUTOFF_DATE)]
        if chunk.empty:
            continue

        plays = chunk[["num_25", "num_50", "num_75", "num_985", "num_100"]].sum(axis=1)
        chunk["log_days_total"] = 1
        chunk["total_secs_all"] = pd.to_numeric(chunk["total_secs"], errors="coerce").fillna(0.0)
        chunk["total_plays_all"] = plays
        chunk["complete_plays_all"] = chunk["num_100"]
        chunk["short_plays_all"] = chunk["num_25"]
        chunk["num_unq_all"] = chunk["num_unq"]
        for suffix, start_date in WINDOW_STARTS.items():
            in_window = chunk["date"].ge(start_date)
            chunk[f"log_days_{suffix}"] = in_window.astype(int)
            chunk[f"total_secs_{suffix}"] = np.where(in_window, chunk["total_secs_all"], 0.0)
            chunk[f"total_plays_{suffix}"] = np.where(in_window, chunk["total_plays_all"], 0.0)
            chunk[f"complete_plays_{suffix}"] = np.where(in_window, chunk["complete_plays_all"], 0.0)
            chunk[f"short_plays_{suffix}"] = np.where(in_window, chunk["short_plays_all"], 0.0)
            chunk[f"num_unq_{suffix}"] = np.where(in_window, chunk["num_unq_all"], 0.0)
        chunk["first_log_date"] = chunk["date"]
        chunk["last_log_date"] = chunk["date"]

        grouped = chunk.groupby("msno", sort=False)[sum_cols + min_cols + max_cols].agg(
            {**{col: "sum" for col in sum_cols}, **{col: "min" for col in min_cols}, **{col: "max" for col in max_cols}}
        )
        aggregated = combine_grouped(aggregated, grouped, sum_cols, min_cols, max_cols)
        print(f"[logs] chunk={idx} groups={len(grouped)} total_groups={len(aggregated)}", flush=True)

    return aggregated if aggregated is not None else pd.DataFrame()


def combine_latest(current: pd.DataFrame | None, update: pd.DataFrame) -> pd.DataFrame:
    if current is None:
        return update
    combined = pd.concat([current, update], axis=0, copy=False)
    combined = combined.sort_values(["msno", "transaction_date"], kind="mergesort")
    return combined.groupby("msno", sort=False).tail(1).set_index("msno")


def aggregate_transactions(paths: list[Path], labeled_msnos: set[str], chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "msno",
        "payment_method_id",
        "payment_plan_days",
        "plan_list_price",
        "actual_amount_paid",
        "is_auto_renew",
        "transaction_date",
        "membership_expire_date",
        "is_cancel",
    ]
    sum_cols = ["txn_count_all", "plan_days_all", "plan_price_all", "paid_amount_all", "auto_renew_all", "cancel_all"]
    for suffix in ["30d", "60d", "90d"]:
        sum_cols.extend([
            f"txn_count_{suffix}",
            f"plan_days_{suffix}",
            f"plan_price_{suffix}",
            f"paid_amount_{suffix}",
            f"auto_renew_{suffix}",
            f"cancel_{suffix}",
        ])
    min_cols = ["first_txn_date"]
    max_cols = ["last_txn_date", "max_membership_expire_date"]
    aggregated: pd.DataFrame | None = None
    latest: pd.DataFrame | None = None

    for path in paths:
        if not path.exists():
            continue
        for idx, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), start=1):
            chunk = chunk[chunk["msno"].isin(labeled_msnos)]
            chunk = chunk[chunk["transaction_date"].le(CUTOFF_DATE)]
            if chunk.empty:
                continue
            for col in usecols:
                if col != "msno":
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)
            chunk["txn_count_all"] = 1
            chunk["plan_days_all"] = chunk["payment_plan_days"]
            chunk["plan_price_all"] = chunk["plan_list_price"]
            chunk["paid_amount_all"] = chunk["actual_amount_paid"]
            chunk["auto_renew_all"] = chunk["is_auto_renew"]
            chunk["cancel_all"] = chunk["is_cancel"]
            for suffix, start_date in {k: v for k, v in WINDOW_STARTS.items() if k != "7d"}.items():
                in_window = chunk["transaction_date"].ge(start_date)
                chunk[f"txn_count_{suffix}"] = in_window.astype(int)
                chunk[f"plan_days_{suffix}"] = np.where(in_window, chunk["payment_plan_days"], 0.0)
                chunk[f"plan_price_{suffix}"] = np.where(in_window, chunk["plan_list_price"], 0.0)
                chunk[f"paid_amount_{suffix}"] = np.where(in_window, chunk["actual_amount_paid"], 0.0)
                chunk[f"auto_renew_{suffix}"] = np.where(in_window, chunk["is_auto_renew"], 0.0)
                chunk[f"cancel_{suffix}"] = np.where(in_window, chunk["is_cancel"], 0.0)
            chunk["first_txn_date"] = chunk["transaction_date"]
            chunk["last_txn_date"] = chunk["transaction_date"]
            chunk["max_membership_expire_date"] = chunk["membership_expire_date"]

            grouped = chunk.groupby("msno", sort=False)[sum_cols + min_cols + max_cols].agg(
                {**{col: "sum" for col in sum_cols}, **{col: "min" for col in min_cols}, **{col: "max" for col in max_cols}}
            )
            latest_chunk = chunk.sort_values(["msno", "transaction_date"], kind="mergesort").groupby("msno", sort=False).tail(1)
            latest_chunk = latest_chunk[[
                "msno",
                "payment_method_id",
                "payment_plan_days",
                "plan_list_price",
                "actual_amount_paid",
                "is_auto_renew",
                "is_cancel",
                "transaction_date",
                "membership_expire_date",
            ]]
            aggregated = combine_grouped(aggregated, grouped, sum_cols, min_cols, max_cols)
            latest = combine_latest(latest, latest_chunk)
            print(
                f"[transactions:{path.name}] chunk={idx} groups={len(grouped)} total_groups={len(aggregated)}",
                flush=True,
            )

    if aggregated is None:
        aggregated = pd.DataFrame()
    if latest is None:
        latest = pd.DataFrame()
    latest = latest.add_prefix("last_")
    return aggregated, latest


def load_members(path: Path, labeled_msnos: set[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    members = pd.read_csv(path, usecols=["msno", "city", "bd", "gender", "registered_via", "registration_init_time"])
    members = members[members["msno"].isin(labeled_msnos)].drop_duplicates("msno").set_index("msno")
    members["age"] = pd.to_numeric(members["bd"], errors="coerce")
    members.loc[~members["age"].between(1, 100), "age"] = np.nan
    members["gender_code"] = pd.factorize(members["gender"].fillna("unknown").astype(str))[0]
    members["registration_date"] = yyyymmdd_to_datetime(members["registration_init_time"])
    members["member_tenure_days"] = (CUTOFF_TS - members["registration_date"]).dt.days.clip(lower=0)
    return members.drop(columns=["bd", "gender", "registration_date"])


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    for suffix in ["7d", "30d", "60d", "90d"]:
        if f"total_secs_{suffix}" in df:
            days = int(suffix.replace("d", ""))
            df[f"daily_secs_{suffix}"] = df[f"total_secs_{suffix}"] / days
            df[f"daily_plays_{suffix}"] = df[f"total_plays_{suffix}"] / days
            df[f"completion_rate_{suffix}"] = df[f"complete_plays_{suffix}"] / df[f"total_plays_{suffix}"].replace(0, np.nan)
            df[f"short_play_rate_{suffix}"] = df[f"short_plays_{suffix}"] / df[f"total_plays_{suffix}"].replace(0, np.nan)
    if "daily_secs_30d" in df and "daily_secs_60d" in df:
        df["usage_secs_change_30_vs_60"] = df["daily_secs_30d"] - df["daily_secs_60d"]
        df["usage_secs_ratio_30_vs_60"] = df["daily_secs_30d"] / df["daily_secs_60d"].replace(0, np.nan)
    if "daily_secs_30d" in df and "daily_secs_90d" in df:
        df["usage_secs_change_30_vs_90"] = df["daily_secs_30d"] - df["daily_secs_90d"]
        df["usage_secs_ratio_30_vs_90"] = df["daily_secs_30d"] / df["daily_secs_90d"].replace(0, np.nan)
    for suffix in ["30d", "60d", "90d"]:
        count = df.get(f"txn_count_{suffix}", pd.Series(0.0, index=df.index)).replace(0, np.nan)
        df[f"avg_paid_amount_{suffix}"] = df.get(f"paid_amount_{suffix}", 0.0) / count
        df[f"avg_plan_price_{suffix}"] = df.get(f"plan_price_{suffix}", 0.0) / count
        df[f"auto_renew_rate_{suffix}"] = df.get(f"auto_renew_{suffix}", 0.0) / count
        df[f"cancel_rate_{suffix}"] = df.get(f"cancel_{suffix}", 0.0) / count
    if "avg_paid_amount_30d" in df and "avg_paid_amount_90d" in df:
        df["paid_amount_change_30_vs_90"] = df["avg_paid_amount_30d"] - df["avg_paid_amount_90d"]
    return df


def build_dataset(labels: pd.DataFrame, logs: pd.DataFrame, txns: pd.DataFrame, latest_txns: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    labels = labels.drop_duplicates("msno").set_index("msno")
    df = labels.join(logs, how="left").join(txns, how="left").join(latest_txns, how="left").join(members, how="left")
    df["snapshot_month"] = "2017-03"
    df["snapshot_date"] = CUTOFF_DATE
    df["churn_next_30d"] = df["is_churn"].astype(int)

    for date_col in ["first_log_date", "last_log_date", "first_txn_date", "last_txn_date", "max_membership_expire_date", "last_membership_expire_date"]:
        if date_col in df:
            date_values = yyyymmdd_to_datetime(df[date_col])
            df[f"days_since_{date_col}"] = (CUTOFF_TS - date_values).dt.days
    if "max_membership_expire_date" in df:
        expire_dt = yyyymmdd_to_datetime(df["max_membership_expire_date"])
        df["days_until_membership_expire"] = (expire_dt - CUTOFF_TS).dt.days
        df["membership_expired_by_cutoff"] = df["days_until_membership_expire"].lt(0).astype(float)

    df = add_ratios(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    object_cols = [col for col in df.columns if df[col].dtype == object and col not in {"snapshot_month"}]
    for col in object_cols:
        if col != "msno":
            df[col] = df[col].fillna("unknown").astype(str)
    ordered = [
        "churn_next_30d",
        "snapshot_month",
        "snapshot_date",
        *[col for col in df.columns if col not in {"is_churn", "churn_next_30d", "snapshot_month", "snapshot_date"}],
    ]
    return df[ordered].reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=raw_data_path("train_v2.csv"))
    parser.add_argument("--members", type=Path, default=raw_data_path("members_v3.csv"))
    parser.add_argument("--user-logs", type=Path, default=raw_data_path("user_logs_v2.csv"))
    parser.add_argument("--transactions", type=Path, nargs="*", default=[raw_data_path("transactions.csv"), raw_data_path("transactions_v2.csv")])
    parser.add_argument("--output", type=Path, default=processed_data_path("kkbox_customer_month_churn.csv"))
    parser.add_argument("--metadata-out", type=Path, default=result_path("kkbox_customer_month_dataset_metadata.json"))
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    labels["msno"] = labels["msno"].astype(str)
    labeled_msnos = set(labels["msno"])
    print(f"[labels] rows={len(labels)} churn_rate={labels['is_churn'].mean():.2%}", flush=True)

    logs = aggregate_user_logs(args.user_logs, labeled_msnos, args.chunksize)
    txns, latest_txns = aggregate_transactions(args.transactions, labeled_msnos, args.chunksize)
    members = load_members(args.members, labeled_msnos)
    dataset = build_dataset(labels, logs, txns, latest_txns, members)

    args.output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    args.metadata_out = args.metadata_out if args.metadata_out.is_absolute() else REPO_ROOT / args.metadata_out
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False, encoding="utf-8-sig")

    metadata = {
        "source": "WSDM/Kaggle KKBox churn public competition files",
        "snapshot_date": str(CUTOFF_TS.date()),
        "label": "churn_next_30d",
        "rows": int(len(dataset)),
        "churn_rate": float(dataset["churn_next_30d"].mean()),
        "columns": list(dataset.columns),
        "available_signal_groups": {
            "usage": "daily listening logs aggregated into 7/30/60/90 day windows where available",
            "billing": "subscription transaction amount, plan, auto-renewal, cancellation, and expiry history",
            "service_events": "cancellation and expiry-derived service status signals",
            "member_profile": "city, age, gender code, registration channel, tenure",
        },
        "not_available_in_public_kkbox": [
            "call center logs",
            "complaint text",
            "retention offer received/accepted",
            "telecom voice/SMS network CDR fields",
        ],
    }
    args.metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[done] rows={len(dataset)} churn_rate={dataset['churn_next_30d'].mean():.2%} "
        f"output={args.output}",
        flush=True,
    )
    print(f"[metadata] {args.metadata_out}", flush=True)


if __name__ == "__main__":
    main()
