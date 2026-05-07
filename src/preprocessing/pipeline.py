import argparse
from pathlib import Path
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.preprocessing.cleaner import clean_data, NUMERIC_FEATURES, CAT_FEATURES, TARGET
from src.preprocessing.encoder import encode_data
from src.preprocessing.scaler import scale_features
from src.utils.helpers import processed_data_path, raw_data_path, resolve_input_path

def make_preprocessor() -> ColumnTransformer:
    """
    수치형 및 범주형 전처리 파이프라인을 결합한 ColumnTransformer를 반환합니다.
    """
    num_pipe = scale_features()
    cat_pipe = encode_data()
    
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES),
    ])

def run_preprocessing_pipeline(input_csv: Path, output_csv: Path) -> None:
    """
    데이터를 로드하고 클리닝, 파생변수 생성 후 전처리(Scaling & Encoding) 파이프라인을 
    적용하여 결과를 저장합니다.
    """
    print(f"[1/4] 데이터 로드 중: {input_csv}")
    raw_df = pd.read_csv(input_csv)
    
    print(f"[2/4] 결측치 보간, 파생변수 생성, 타겟 이진화 중...")
    X, y = clean_data(raw_df)
    
    print(f"[3/4] 수치형(Scaling) 및 범주형(Encoding) 변환 중...")
    preprocessor = make_preprocessor()
    
    # ColumnTransformer는 NumPy Array를 반환하므로 DataFrame으로 재구성
    X_transformed = preprocessor.fit_transform(X)
    
    # 반환되는 컬럼 순서는 num_pipe에 정의된 피처 후 cat_pipe 피처
    out_columns = NUMERIC_FEATURES + CAT_FEATURES
    
    processed_df = pd.DataFrame(X_transformed, columns=out_columns)
    
    # 타겟(y) 결합
    processed_df[TARGET] = y
    
    print(f"[4/4] 결과 저장 중: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_csv, index=False)
    print("완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=raw_data_path("baza_telecom_v2.csv"))
    parser.add_argument("--output", type=Path, default=processed_data_path("baza_telecom_v2_processed.csv"))
    args = parser.parse_args()

    args.input = resolve_input_path(args.input, raw_data_path("baza_telecom_v2.csv"))
    if not args.input.is_file():
        raise SystemExit(f"입력 파일이 없습니다: {args.input}")
        
    run_preprocessing_pipeline(args.input, args.output)
