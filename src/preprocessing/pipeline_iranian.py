import argparse
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.preprocessing.scaler import scale_features

def run_iranian_pipeline(input_csv: Path, output_csv: Path) -> None:
    print(f"[Iranian] 로드 중: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 1. 컬럼명 정리 (더블 스페이스 등 제거)
    df.columns = df.columns.str.replace('  ', ' ').str.replace(' ', '_')
    
    # 2. Target
    target_col = "Churn"
    if target_col in df.columns:
        y = df[target_col].astype(int)
        df = df.drop(columns=[target_col])
    else:
        y = None
        
    # 3. All remaining features are numeric/ordinal, treat as numeric
    numeric_features = list(df.columns)
    
    num_pipe = scale_features()
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_features)
    ])
    
    print("[Iranian] 전처리(Scaling) 적용 중...")
    X_transformed = preprocessor.fit_transform(df)
    
    processed_df = pd.DataFrame(X_transformed, columns=numeric_features)
    
    if y is not None:
        processed_df["CHURN"] = y.values
        
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_csv, index=False)
    print(f"[Iranian] 완료: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/iranian_churn.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/iranian_churn_processed.csv"))
    args = parser.parse_args()
    
    if not args.input.is_file():
        raise SystemExit(f"입력 파일이 없습니다: {args.input}")
        
    run_iranian_pipeline(args.input, args.output)
