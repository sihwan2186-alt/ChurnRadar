import argparse
import sys
import logging
from pathlib import Path

# 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

def train_xgb(csv_path: str):
    logger.info("XGBoost 훈련 파이프라인 시작...")
    try:
        from scripts.train_ensemble import main as xgb_main
        # sys.argv 덮어쓰기 (argparse 충돌 방지)
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]
        xgb_main()
        sys.argv = original_argv
    except Exception as e:
        logger.warning(f"scripts.train_ensemble을 직접 import 실행하는데 실패했습니다 ({e}). subprocess로 대체 실행합니다.")
        import subprocess
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "train_ensemble.py")], check=True)

def train_transformer(csv_path: str, epochs: int):
    logger.info(f"TS-Transformer 훈련 파이프라인 시작... (epochs: {epochs})")
    try:
        from scripts.train_ts_engine import train_engine
        train_engine(epochs=epochs)
    except Exception as e:
        logger.warning(f"scripts.train_ts_engine을 직접 import 실행하는데 실패했습니다 ({e}). subprocess로 대체 실행합니다.")
        import subprocess
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "train_ts_engine.py")], check=True)

def main():
    parser = argparse.ArgumentParser(description="ChurnRadar 통합 학습 파이프라인")
    parser.add_argument("--model", choices=["xgb", "transformer", "all"], default="all", help="학습할 모델 선택")
    parser.add_argument("--csv", type=str, default="data/raw/baza.csv", help="데이터 CSV 경로")
    parser.add_argument("--parquet", type=str, default="data/raw/baza.parquet", help="데이터 Parquet 경로")
    parser.add_argument("--epochs", type=int, default=50, help="Transformer 학습 에폭 수")
    
    args = parser.parse_args()
    
    logger.info("=== MLOps Data Leakage 방지 원칙 ===")
    logger.info("1. 모든 모델 학습 시 Scaler의 fit은 Train 데이터에만 독립적으로 적용됩니다.")
    logger.info("2. Test/Validation 데이터는 오직 transform만 수행합니다.")
    logger.info("3. TS-SMOTE 오버샘플링 역시 Train 데이터에만 적용됩니다.\n")
    
    logger.info("저장 경로:")
    logger.info("- XGBoost: models/model.joblib")
    logger.info("- Transformer: models/transformer_churn_v1.pth\n")
    
    if args.model in ["xgb", "all"]:
        train_xgb(args.csv)
        
    if args.model in ["transformer", "all"]:
        train_transformer(args.csv, args.epochs)
        
    logger.info("통합 학습 파이프라인(train_pipeline.py) 구동 종료")

if __name__ == "__main__":
    main()
