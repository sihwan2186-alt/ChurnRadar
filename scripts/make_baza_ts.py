#!/usr/bin/env python3
"""Create a Baza Telecom time-series parquet file.

The source Baza data is a static snapshot, so this script creates a cautious
simulation for sequence-model experiments. Unlike the old iid-noise version,
the generated sequence now contains a label-aware direction:

- churn rows: activity energy and active momentum gradually decline
- non-churn rows: activity energy and active momentum stay stable or rise mildly
- churn rows: inactive acceleration gradually increases

This is still an assumption-based simulation and should be reported as such.

Feature mapping:
  Activity_Energy  = ARPU-like usage/revenue energy
  Momentum         = Active_subscribers / Total_SUBs
  Acceleration     = Not_Active_subscribers / Total_SUBs

Output: data/processed/baza_ts.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.utils.helpers import processed_data_path, raw_data_path, resolve_input_path

DEFAULT_CSV = raw_data_path("baza_telecom_v2.csv")
DEFAULT_OUT = processed_data_path("baza_ts.parquet")

NOISE_STD = 0.04
TIME_STEPS = 30
RANDOM_SEED = 42

CHURN_ENERGY_END = (0.60, 0.85)
STABLE_ENERGY_END = (0.98, 1.05)
CHURN_MOMENTUM_END = (0.70, 0.90)
STABLE_MOMENTUM_END = (0.98, 1.05)
CHURN_INACTIVE_END = (1.05, 1.35)
STABLE_INACTIVE_END = (0.90, 1.05)


def _trend(
    rng: np.random.Generator,
    is_churn: bool,
    churn_range: tuple[float, float],
    stable_range: tuple[float, float],
    time_steps: int,
) -> np.ndarray:
    end = rng.uniform(*(churn_range if is_churn else stable_range))
    return np.linspace(1.0, end, time_steps)


def simulate_baza_sequence(
    base_energy: float,
    base_momentum: float,
    base_accel: float,
    is_churn: bool,
    rng: np.random.Generator,
    time_steps: int = TIME_STEPS,
    noise_std: float = NOISE_STD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one simulated 30-step Baza sequence."""

    energy_trend = _trend(rng, is_churn, CHURN_ENERGY_END, STABLE_ENERGY_END, time_steps)
    momentum_trend = _trend(rng, is_churn, CHURN_MOMENTUM_END, STABLE_MOMENTUM_END, time_steps)
    inactive_trend = _trend(rng, is_churn, CHURN_INACTIVE_END, STABLE_INACTIVE_END, time_steps)

    noise_e = rng.normal(1.0, noise_std, size=time_steps)
    noise_m = rng.normal(1.0, noise_std, size=time_steps)
    noise_a = rng.normal(1.0, noise_std, size=time_steps)

    energy_seq = np.clip(base_energy * energy_trend * noise_e, 0.0, None)
    momentum_seq = np.clip(base_momentum * momentum_trend * noise_m, 0.0, 1.0)
    accel_seq = np.clip(base_accel * inactive_trend * noise_a, 0.0, 1.0)
    return energy_seq, momentum_seq, accel_seq


def make_baza_ts(csv_path: Path, out_path: Path, time_steps: int = TIME_STEPS) -> None:
    import polars as pl

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"[Load] {csv_path}")
    df = pl.read_csv(csv_path, infer_schema_length=0)
    str_cols = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Utf8 or t == pl.String]
    df = df.with_columns([pl.col(c).str.strip_chars() for c in str_cols])

    numeric_cols = [
        "Total_SUBs",
        "Active_subscribers",
        "Not_Active_subscribers",
        "TotalRevenue",
        "ARPU",
    ]
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in numeric_cols])

    df = df.with_columns(
        pl.col("CHURN")
        .str.to_lowercase()
        .map_elements(lambda v: 1 if v == "yes" else 0, return_dtype=pl.Int32)
        .alias("is_churn")
    )

    total_subs = pl.col("Total_SUBs").cast(pl.Float64)
    safe_total = pl.when((total_subs <= 0) | total_subs.is_null()).then(1.0).otherwise(total_subs)
    arpu_fallback = pl.col("TotalRevenue").cast(pl.Float64) / safe_total

    df = df.with_columns(
        [
            pl.when(pl.col("ARPU").is_null())
            .then(arpu_fallback)
            .otherwise(pl.col("ARPU"))
            .fill_null(0.0)
            .alias("base_energy"),
            (pl.col("Active_subscribers").fill_null(0.0) / safe_total)
            .clip(0.0, 1.0)
            .alias("base_momentum"),
            (pl.col("Not_Active_subscribers").fill_null(0.0) / safe_total)
            .clip(0.0, 1.0)
            .alias("base_accel"),
        ]
    )

    churn_rate = float(df["is_churn"].mean() or 0.0)
    print(f"[Info] customers={len(df):,} churn_rate={churn_rate:.2%}")
    print(f"[Info] timesteps={time_steps} trend_noise={NOISE_STD:.2f}")

    rows: list[dict] = []
    for row in df.iter_rows(named=True):
        pid = str(row["PID"])
        is_churn = bool(int(row["is_churn"]))

        energy_seq, momentum_seq, accel_seq = simulate_baza_sequence(
            base_energy=float(row["base_energy"]),
            base_momentum=float(row["base_momentum"]),
            base_accel=float(row["base_accel"]),
            is_churn=is_churn,
            rng=rng,
            time_steps=time_steps,
        )

        for t in range(time_steps):
            rows.append(
                {
                    "Entity_ID": pid,
                    "Event_Time": t,
                    "Activity_Energy": float(energy_seq[t]),
                    "Momentum": float(momentum_seq[t]),
                    "Acceleration": float(accel_seq[t]),
                    "is_churn": int(is_churn),
                }
            )

    result = pl.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(out_path)

    n_users = result["Entity_ID"].n_unique()
    n_churn_users = result.filter(pl.col("is_churn") == 1)["Entity_ID"].n_unique()
    print(f"[Done] output={out_path}")
    print(f"[Done] rows={len(result):,} users={n_users:,} timesteps={time_steps}")
    print(f"[Done] churn_users={n_churn_users:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timesteps", type=int, default=TIME_STEPS)
    args = parser.parse_args()

    args.csv = resolve_input_path(args.csv, DEFAULT_CSV)
    if not args.out.is_absolute():
        args.out = REPO_ROOT / args.out

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    make_baza_ts(args.csv, args.out, args.timesteps)
