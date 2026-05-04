import argparse
from pathlib import Path
import polars as pl

def map_kkbox_to_energy(input_csv: Path, output_parquet: Path) -> None:
    print(f"[KKBox] 로드 중 (Lazy): {input_csv}")
    
    # 1. Lazy load with Polars
    # Columns: msno,date,num_25,num_50,num_75,num_985,num_100,num_unq,total_secs
    q = pl.scan_csv(input_csv)
    
    # 2. Activity Energy 매핑 (total_secs 활용)
    #    날짜(date) 기준으로 오름차순 정렬 후, 고객(msno)별로 Momentum/Acceleration 계산
    q = (
        q.rename({"msno": "Entity_ID", "date": "Event_Time", "total_secs": "Activity_Energy"})
        .select(["Entity_ID", "Event_Time", "Activity_Energy"])
        .sort(["Entity_ID", "Event_Time"])
    )
    
    # 3. Momentum & Acceleration 계산
    # Momentum: 현재 에너지 / 과거(직전) 에너지
    # Acceleration: 현재 모멘텀 - 과거 모멘텀
    q = q.with_columns([
        pl.col("Activity_Energy").shift(1).over("Entity_ID").alias("Prev_Energy")
    ])
    
    q = q.with_columns([
        (pl.col("Activity_Energy") / (pl.col("Prev_Energy") + 1e-5)).alias("Momentum")
    ])
    
    q = q.with_columns([
        pl.col("Momentum").shift(1).over("Entity_ID").alias("Prev_Momentum")
    ])
    
    q = q.with_columns([
        (pl.col("Momentum") - pl.col("Prev_Momentum")).alias("Acceleration")
    ])
    
    # 결측치(초기값)는 0으로 처리
    q = q.fill_null(0.0)
    
    print("[KKBox] 연산(Collect) 및 Parquet 저장 중...")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect executes the lazy query
    df_result = q.collect()
    df_result.write_parquet(output_parquet)
    print(f"[KKBox] 완료. 저장 크기: {df_result.shape}, 경로: {output_parquet}")

def map_hm_to_energy(input_csv: Path, output_parquet: Path) -> None:
    print(f"[H&M] 로드 중 (Lazy): {input_csv}")
    
    # 1. Lazy load
    # Columns: t_dat,customer_id,article_id,price,sales_channel_id
    q = pl.scan_csv(input_csv)
    
    # 2. 매핑 및 집계 (동일 날짜의 구매 가격 합산)
    q = (
        q.rename({"customer_id": "Entity_ID", "t_dat": "Event_Time", "price": "Activity_Energy"})
        .group_by(["Entity_ID", "Event_Time"])
        .agg(pl.col("Activity_Energy").sum())
        .sort(["Entity_ID", "Event_Time"])
    )
    
    # 3. Momentum & Acceleration 계산
    q = q.with_columns([
        pl.col("Activity_Energy").shift(1).over("Entity_ID").alias("Prev_Energy")
    ])
    
    q = q.with_columns([
        (pl.col("Activity_Energy") / (pl.col("Prev_Energy") + 1e-5)).alias("Momentum")
    ])
    
    q = q.with_columns([
        pl.col("Momentum").shift(1).over("Entity_ID").alias("Prev_Momentum")
    ])
    
    q = q.with_columns([
        (pl.col("Momentum") - pl.col("Prev_Momentum")).alias("Acceleration")
    ])
    
    q = q.fill_null(0.0)
    
    print("[H&M] 연산(Collect) 및 Parquet 저장 중...")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    
    df_result = q.collect()
    df_result.write_parquet(output_parquet)
    print(f"[H&M] 완료. 저장 크기: {df_result.shape}, 경로: {output_parquet}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, choices=["kkbox", "hm"], required=True, help="처리할 도메인 지정")
    parser.add_argument("--input", type=Path, required=True, help="입력 CSV 경로")
    parser.add_argument("--output", type=Path, required=True, help="출력 Parquet 경로")
    args = parser.parse_args()
    
    if not args.input.is_file():
        raise SystemExit(f"입력 파일 없음: {args.input}")
        
    if args.domain == "kkbox":
        map_kkbox_to_energy(args.input, args.output)
    elif args.domain == "hm":
        map_hm_to_energy(args.input, args.output)
