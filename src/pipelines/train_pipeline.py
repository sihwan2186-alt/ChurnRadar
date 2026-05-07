import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.utils.helpers import model_path, processed_data_path, raw_data_path, resolve_input_path


def train_xgb(csv_path: Path) -> None:
    logger.info("XGBoost 훈련 파이프라인 시작...")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_ensemble.py"),
            "--csv",
            str(csv_path),
            "--out",
            str(model_path("model.joblib")),
        ],
        check=True,
    )


def train_transformer(parquet_path: Path, epochs: int) -> None:
    logger.info(f"TS-Transformer 훈련 파이프라인 시작... (epochs: {epochs})")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_ts_engine.py"),
            "--input",
            str(parquet_path),
            "--output",
            str(model_path("transformer_churn_v1.pth")),
            "--epochs",
            str(epochs),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ChurnRadar 통합 학습 파이프라인")
    parser.add_argument("--model", choices=["xgb", "transformer", "all"], default="all", help="학습할 모델 선택")
    parser.add_argument("--csv", type=Path, default=raw_data_path("baza_telecom_v2.csv"), help="데이터 CSV 경로")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=processed_data_path("kkbox_real_gold_v1.parquet"),
        help="데이터 Parquet 경로",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Transformer 학습 에폭 수")
    args = parser.parse_args()

    args.csv = resolve_input_path(args.csv, raw_data_path("baza_telecom_v2.csv"))
    args.parquet = resolve_input_path(args.parquet, processed_data_path("kkbox_real_gold_v1.parquet"))

    logger.info("=== MLOps Data Leakage 방지 원칙 ===")
    logger.info("1. 모든 모델 학습 시 Scaler의 fit은 Train 데이터에만 독립적으로 적용됩니다.")
    logger.info("2. Test/Validation 데이터는 오직 transform만 수행합니다.")
    logger.info("3. TS-SMOTE 오버샘플링 역시 Train 데이터에만 적용됩니다.\n")

    logger.info("실행 경로:")
    logger.info(f"- Raw CSV: {args.csv}")
    logger.info(f"- TS Parquet: {args.parquet}")
    logger.info(f"- XGBoost: {model_path('model.joblib')}")
    logger.info(f"- Transformer: {model_path('transformer_churn_v1.pth')}\n")

    if args.model in ["xgb", "all"]:
        train_xgb(args.csv)

    if args.model in ["transformer", "all"]:
        train_transformer(args.parquet, args.epochs)

    logger.info("통합 학습 파이프라인(train_pipeline.py) 구동 종료")


if __name__ == "__main__":
    main()
