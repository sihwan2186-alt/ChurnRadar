import polars as pl
from pathlib import Path

def create_real_gold_dataset():
    print("🚀 [Real Gold] 시계열 로그와 정답지(Label) 및 프로필 병합 시작")
    
    # 1. 기존에 만든 로그 파티션(Parquet) 로드
    # 저장 경로가 data/interim/kkbox_user_logs.parquet 였습니다.
    logs_path = Path("data/interim/kkbox_user_logs.parquet")
    if not logs_path.exists():
        raise FileNotFoundError(f"로그 파일이 없습니다. 먼저 mapper를 실행하세요: {logs_path}")
        
    print(f"  - 로그 데이터 로드: {logs_path}")
    logs = pl.read_parquet(logs_path)
    
    # 여기서 Entity_ID가 존재하므로 명시적 String 캐스팅 보장
    logs = logs.with_columns(pl.col("Entity_ID").cast(pl.Utf8))
    
    # 2. 정답지 및 멤버 데이터 로드
    label_path = Path("data/raw/train_v2.csv")
    members_path = Path("data/raw/members_v3.csv")
    
    print(f"  - 정답지 로드: {label_path}")
    # msno 컬럼을 일관성을 위해 Entity_ID로 이름 변경 후 String 캐스팅
    train_label = pl.read_csv(label_path).rename({"msno": "Entity_ID"}).with_columns(pl.col("Entity_ID").cast(pl.Utf8))
    
    print(f"  - 멤버 프로필 로드: {members_path}")
    members = pl.read_csv(members_path).rename({"msno": "Entity_ID"}).with_columns(pl.col("Entity_ID").cast(pl.Utf8))

    # 3. Inner Join으로 정답이 있는 유저만 남기기
    print("  - 정답이 존재하는 훈련 유저만 Inner Join 필터링 중...")
    df = logs.join(train_label, on="Entity_ID", how="inner")
    
    # 4. 멤버 정보 결합 (나이, 성별 등) Left Join
    print("  - 멤버 프로필 속성 Left Join 병합 중...")
    df = df.join(members, on="Entity_ID", how="left")
    
    # 5. 최종 '진짜' 데이터셋 저장
    out_path = Path("data/processed/kkbox_real_gold_v1.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("  - 최종 병합 결과 Parquet 저장 중...")
    df.write_parquet(out_path)
    
    print(f"✅ 진짜 데이터셋 완성! 총 {df.height}개의 시계열 레코드가 확보되었습니다.")
    print(f"   -> 저장 경로: {out_path}")

if __name__ == "__main__":
    create_real_gold_dataset()
