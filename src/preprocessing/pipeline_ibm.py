import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.preprocessing.encoder import encode_data
from src.preprocessing.scaler import scale_features
from src.utils.helpers import processed_data_path, raw_data_path, resolve_input_path

def run_ibm_pipeline(input_csv: Path, output_csv: Path) -> None:
    print(f"[IBM] 로드 중: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 1. Drop customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        
    # 2. TotalCharges is object, convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    
    # 3. Target mapping
    if "Churn" in df.columns:
        y = df["Churn"].map({"Yes": 1, "No": 0})
        df = df.drop(columns=["Churn"])
    else:
        y = None
        
    # 4. Feature separation
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    # SeniorCitizen is 0/1 but conceptually categorical, though numeric works. 
    # Let's keep it numeric since it's already encoded as 0/1, or we can encode it.
    if "SeniorCitizen" in df.columns:
        numeric_features.append("SeniorCitizen")
        
    cat_features = [col for col in df.columns if col not in numeric_features]
    
    # 5. Preprocessor
    num_pipe = scale_features()
    cat_pipe = encode_data()
    
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, cat_features),
    ])
    
    print("[IBM] 전처리(Scaling & Encoding) 적용 중...")
    X_transformed = preprocessor.fit_transform(df)
    
    out_columns = numeric_features + cat_features
    processed_df = pd.DataFrame(X_transformed, columns=out_columns)
    
    if y is not None:
        processed_df["CHURN"] = y.values
        
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_csv, index=False)
    print(f"[IBM] 완료: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=raw_data_path("ibm_telco_churn.csv"))
    parser.add_argument("--output", type=Path, default=processed_data_path("ibm_telco_churn_processed.csv"))
    args = parser.parse_args()

    args.input = resolve_input_path(args.input, raw_data_path("ibm_telco_churn.csv"))
    if not args.input.is_file():
        raise SystemExit(f"입력 파일이 없습니다: {args.input}")
        
    run_ibm_pipeline(args.input, args.output)
