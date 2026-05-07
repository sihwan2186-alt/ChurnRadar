#!/usr/bin/env python3
# scripts/make_baza_ts.py
"""
Baza Telecom v2 스냅샷 CSV → 시계열 Parquet 변환 스크립트.

1행 = 1고객 스냅샷 데이터를 ±5% 정규분포 노이즈로 30 timestep 시계열로 시뮬레이션.

피처 매핑:
  Activity_Energy  = ARPU
  Momentum         = Active_subscribers / Total_SUBs
  Acceleration     = Not_Active_subscribers / Total_SUBs

출력: data/processed/baza_ts.parquet

Usage:
    python scripts/make_baza_ts.py
    python scripts/make_baza_ts.py --csv data/raw/baza_telecom_v2.csv --out data/processed/baza_ts.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "baza_telecom_v2.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "baza_ts.parquet"

NOISE_STD = 0.05   # ±5% 노이즈 표준편차
TIME_STEPS = 30
RANDOM_SEED = 42


def make_baza_ts(csv_path: Path, out_path: Path, time_steps: int = TIME_STEPS) -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"[Load] {csv_path}")
    df = pl.read_csv(csv_path, infer_schema_length=0)  # 전체 행 스캔으로 타입 오류 방지
    str_cols = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Utf8 or t == pl.String]
    df = df.with_columns([pl.col(c).str.strip_chars() for c in str_cols])

    # CHURN → is_churn (0/1)
    df = df.with_columns(
        pl.col("CHURN").str.to_lowercase().map_elements(
            lambda v: 1 if v == "yes" else 0, return_dtype=pl.Int32
        ).alias("is_churn")
    )

    # 기반 피처 계산
    total_subs = df["Total_SUBs"].cast(pl.Float64)
    safe_total = pl.when(total_subs == 0).then(1.0).otherwise(total_subs)

    df = df.with_columns([
        pl.col("ARPU").cast(pl.Float64).fill_null(0.0).alias("base_energy"),
        (pl.col("Active_subscribers").cast(pl.Float64) / safe_total).clip(0.0, 1.0).alias("base_momentum"),
        (pl.col("Not_Active_subscribers").cast(pl.Float64).fill_null(0.0) / safe_total).clip(0.0, 1.0).alias("base_accel"),
    ])

    print(f"[Info] 총 고객: {len(df)}명 | 이탈률: {df['is_churn'].mean():.2%}")
    print(f"[Info] 시계열 생성: {time_steps} timestep, 노이즈 ±{NOISE_STD*100:.0f}%")

    rows: list[dict] = []
    for row in df.iter_rows(named=True):
        pid = str(row["PID"])
        is_churn = int(row["is_churn"])

        base_e = float(row["base_energy"])
        base_m = float(row["base_momentum"])
        base_a = float(row["base_accel"])

        # ±5% 정규분포 노이즈로 30 timestep 시뮬레이션
        noise_e = rng.normal(1.0, NOISE_STD, size=time_steps)
        noise_m = rng.normal(1.0, NOISE_STD, size=time_steps)
        noise_a = rng.normal(1.0, NOISE_STD, size=time_steps)

        energy_seq     = np.clip(base_e * noise_e, 0, None)
        momentum_seq   = np.clip(base_m * noise_m, 0.0, 1.0)
        accel_seq      = np.clip(base_a * noise_a, 0.0, 1.0)

        for t in range(time_steps):
            rows.append({
                "Entity_ID":       pid,
                "Event_Time":      t,
                "Activity_Energy": float(energy_seq[t]),
                "Momentum":        float(momentum_seq[t]),
                "Acceleration":    float(accel_seq[t]),
                "is_churn":        is_churn,
            })

    result = pl.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(out_path)

    n_users = result["Entity_ID"].n_unique()
    print(f"\n✅ 저장 완료: {out_path}")
    print(f"   총 행 수: {len(result):,}행 ({n_users}명 × {time_steps} timestep)")
    print(f"   이탈 유저: {result.filter(pl.col('is_churn') == 1)['Entity_ID'].n_unique()}명")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timesteps", type=int, default=TIME_STEPS)
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV 없음: {args.csv}")

    make_baza_ts(args.csv, args.out, args.timesteps)
