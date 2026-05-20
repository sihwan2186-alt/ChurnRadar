import argparse
from pathlib import Path

import polars as pl


def map_kkbox_to_energy(input_csv: Path, output_parquet: Path) -> None:
    print(f"[KKBox] Loading lazily: {input_csv}")

    q = pl.scan_csv(input_csv)

    total_plays = (
        pl.col("num_25")
        + pl.col("num_50")
        + pl.col("num_75")
        + pl.col("num_985")
        + pl.col("num_100")
    )

    q = (
        q.rename({"msno": "Entity_ID", "date": "Event_Time", "total_secs": "Activity_Energy"})
        .with_columns(
            [
                total_plays.alias("Total_Plays"),
                (pl.col("num_25") / (pl.col("num_unq") + 1e-5)).clip(0.0, 1.0).alias("Skip_Rate"),
                (pl.col("num_100") / (pl.col("num_unq") + 1e-5)).clip(0.0, 1.0).alias("Completion_Rate"),
                (pl.col("num_unq") / (total_plays + 1e-5)).clip(0.0, 1.0).alias("Diversity_Score"),
            ]
        )
        .select(
            [
                "Entity_ID",
                "Event_Time",
                "Activity_Energy",
                "Skip_Rate",
                "Completion_Rate",
                "Diversity_Score",
            ]
        )
        .sort(["Entity_ID", "Event_Time"])
    )

    q = q.with_columns(
        [pl.col("Activity_Energy").shift(1).over("Entity_ID").alias("Prev_Energy")]
    )
    q = q.with_columns(
        [(pl.col("Activity_Energy") / (pl.col("Prev_Energy") + 1e-5)).alias("Momentum")]
    )
    q = q.with_columns(
        [pl.col("Momentum").shift(1).over("Entity_ID").alias("Prev_Momentum")]
    )
    q = q.with_columns(
        [(pl.col("Momentum") - pl.col("Prev_Momentum")).alias("Acceleration")]
    )

    q = q.fill_null(0.0).select(
        [
            "Entity_ID",
            "Event_Time",
            "Activity_Energy",
            "Momentum",
            "Acceleration",
            "Skip_Rate",
            "Completion_Rate",
            "Diversity_Score",
        ]
    )

    print("[KKBox] Collecting and writing parquet...")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_result = q.collect()
    df_result.write_parquet(output_parquet)
    print(f"[KKBox] Done. shape={df_result.shape} output={output_parquet}")


def map_hm_to_energy(input_csv: Path, output_parquet: Path) -> None:
    print(f"[H&M] Loading lazily: {input_csv}")

    q = pl.scan_csv(input_csv)
    q = (
        q.rename({"customer_id": "Entity_ID", "t_dat": "Event_Time", "price": "Activity_Energy"})
        .group_by(["Entity_ID", "Event_Time"])
        .agg(pl.col("Activity_Energy").sum())
        .sort(["Entity_ID", "Event_Time"])
    )

    q = q.with_columns(
        [pl.col("Activity_Energy").shift(1).over("Entity_ID").alias("Prev_Energy")]
    )
    q = q.with_columns(
        [(pl.col("Activity_Energy") / (pl.col("Prev_Energy") + 1e-5)).alias("Momentum")]
    )
    q = q.with_columns(
        [pl.col("Momentum").shift(1).over("Entity_ID").alias("Prev_Momentum")]
    )
    q = q.with_columns(
        [(pl.col("Momentum") - pl.col("Prev_Momentum")).alias("Acceleration")]
    )
    q = q.fill_null(0.0).select(["Entity_ID", "Event_Time", "Activity_Energy", "Momentum", "Acceleration"])

    print("[H&M] Collecting and writing parquet...")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_result = q.collect()
    df_result.write_parquet(output_parquet)
    print(f"[H&M] Done. shape={df_result.shape} output={output_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, choices=["kkbox", "hm"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    if args.domain == "kkbox":
        map_kkbox_to_energy(args.input, args.output)
    elif args.domain == "hm":
        map_hm_to_energy(args.input, args.output)
