import argparse
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.preprocessing.encoder import encode_data
from src.preprocessing.scaler import scale_features

def run_cell2cell_pipeline(train_csv: Path, holdout_csv: Path, out_train_csv: Path, out_holdout_csv: Path) -> None:
    print(f"[Cell2Cell] 로드 중: {train_csv.name}, {holdout_csv.name}")
    
    df_train = pd.read_csv(train_csv)
    df_holdout = pd.read_csv(holdout_csv)
    
    # 1. Drop CustomerID
    for df in (df_train, df_holdout):
        if "CustomerID" in df.columns:
            df.drop(columns=["CustomerID"], inplace=True)
            
    # 2. Target mapping
    target_col = "Churn"
    y_train, y_holdout = None, None
    
    if target_col in df_train.columns:
        y_train = df_train[target_col].map({"Yes": 1, "No": 0, "1": 1, "0": 0})
        # If any mapped values are NaN but original wasn't, let's keep original numeric if already 0/1
        if df_train[target_col].dtype in ['int64', 'float64'] and y_train.isna().all():
             y_train = df_train[target_col]
        df_train.drop(columns=[target_col], inplace=True)
        
    if target_col in df_holdout.columns:
        y_holdout = df_holdout[target_col].map({"Yes": 1, "No": 0, "1": 1, "0": 0})
        if df_holdout[target_col].dtype in ['int64', 'float64'] and y_holdout.isna().all():
             y_holdout = df_holdout[target_col]
        df_holdout.drop(columns=[target_col], inplace=True)
        
    # 3. Detect numeric vs categorical
    numeric_features = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 4. Preprocessor
    num_pipe = scale_features()
    cat_pipe = encode_data()
    
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, cat_features),
    ])
    
    print("[Cell2Cell] 전처리(Scaling & Encoding) 적용 중...")
    
    # Fit on train, transform both
    X_train_transformed = preprocessor.fit_transform(df_train)
    X_holdout_transformed = preprocessor.transform(df_holdout)
    
    out_columns = numeric_features + cat_features
    processed_train = pd.DataFrame(X_train_transformed, columns=out_columns)
    processed_holdout = pd.DataFrame(X_holdout_transformed, columns=out_columns)
    
    if y_train is not None:
        processed_train["CHURN"] = y_train.values
    if y_holdout is not None:
        processed_holdout["CHURN"] = y_holdout.values
        
    out_train_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_train.to_csv(out_train_csv, index=False)
    processed_holdout.to_csv(out_holdout_csv, index=False)
    
    print(f"[Cell2Cell] 완료: {out_train_csv.name}, {out_holdout_csv.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/cell2cell_train.csv"))
    parser.add_argument("--holdout", type=Path, default=Path("data/raw/cell2cell_holdout.csv"))
    parser.add_argument("--out_train", type=Path, default=Path("data/processed/cell2cell_train_processed.csv"))
    parser.add_argument("--out_holdout", type=Path, default=Path("data/processed/cell2cell_holdout_processed.csv"))
    args = parser.parse_args()
    
    if not args.train.is_file():
        raise SystemExit(f"입력 파일이 없습니다: {args.train}")
        
    run_cell2cell_pipeline(args.train, args.holdout, args.out_train, args.out_holdout)
