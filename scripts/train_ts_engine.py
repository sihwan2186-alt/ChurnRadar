import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score
import numpy as np

# 모듈 로드
from src.data.ts_dataset import ChurnTimeSeriesDataset
from src.models.ts_model import ChurnLSTM
# from src.models.ts_smote import TSSMOTE

def train_engine(parquet_path: Path, model_out: Path, epochs: int = 5, batch_size: int = 64, lr: float = 1e-3):
    print("=" * 60)
    print("🚀 ChurnRadar Pro - Time-Series LSTM Engine Training (Baseline)")
    print("=" * 60)

    # 1. Device 설정 (Apple Silicon MPS 최적화)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 하드웨어 가속기 감지됨: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ 하드웨어 가속기 감지됨: NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("⚠️ 가속기 없음. CPU 모드로 동작합니다.")

    # 2. 데이터 로드
    # 병합된 'Real Gold' 데이터셋 로드 (is_churn 포함)
    dataset = ChurnTimeSeriesDataset(parquet_path, max_seq_len=30, target_col="is_churn")
    X_np = dataset.X
    y_np = dataset.y

    print(f"\n[Dataset] 총 추출된 유저 수: {len(y_np)}명, 전체 이탈자 비율: {y_np.mean():.2%}")

    # 3. 데이터 분할 (Train / Validation)
    print("[Split] 훈련(Train) 및 검증(Val) 80:20 분할 중...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
    )

    # TS-SMOTE 주석 처리 (Baseline First 전략)
    # print("\n[TS-SMOTE] 데이터 증식(Augmentation) 수행 중...")
    # ts_smote = TSSMOTE(random_state=42)
    # X_resampled, y_resampled = ts_smote.fit_resample(X_train, y_train)

    # Random Oversampling (단순 복제)
    print("\n[Oversampling] 소수 클래스(이탈자) 단순 복제 증식 적용 중...")
    minority_idx = np.where(y_train == 1)[0]
    majority_idx = np.where(y_train == 0)[0]

    # 이탈자를 2배 복제 추가 (원본 1배 + 복제 2배 = 총 3배)
    minority_idx_oversampled = np.random.choice(minority_idx, size=len(minority_idx) * 2, replace=True)

    # 병합 후 셔플
    resampled_idx = np.concatenate([majority_idx, minority_idx, minority_idx_oversampled])
    np.random.shuffle(resampled_idx)

    X_resampled = X_train[resampled_idx]
    y_resampled = y_train[resampled_idx]

    # TensorDataset 및 DataLoader로 변환
    train_dataset = TensorDataset(torch.tensor(X_resampled), torch.tensor(y_resampled).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. 모델 생성 및 디바이스 이동
    model = ChurnLSTM(input_size=3, hidden_size=64, num_layers=2).to(device)

    # 불균형 처리를 위한 Weighted Loss 계산 (pos_weight)
    n_pos = y_resampled.sum()
    n_neg = len(y_resampled) - n_pos
    pos_weight_val = n_neg / (n_pos + 1e-5)
    print(f"\n[Weighted Loss] 소수 클래스 가중치(pos_weight) 적용: {pos_weight_val:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning Rate 스케줄러 추가 (3에포크마다 절반으로 감소)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # 5. 훈련 루프
    print(f"\n🔥 학습 시작 (Total Epochs: {epochs}, Batch Size: {batch_size})")

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # 검증(Validation) 루프
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                # 시그모이드 후 0.5 기준으로 클래스 판별
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        recall = recall_score(val_targets, val_preds, zero_division=0)
        f1 = f1_score(val_targets, val_preds, zero_division=0)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  [Epoch {epoch+1}/{epochs}] LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f} ({time.time() - start_time:.1f}초)")

        # 스케줄러 스텝 적용
        scheduler.step()

    print(f"\n✅ 전체 학습 완료!")

    # 6. 모델 저장
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"저장 완료: {model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 병합된 진짜 데이터 경로 지정
    parser.add_argument("--input", type=Path, default=Path("data/processed/kkbox_real_gold_v1.parquet"))
    parser.add_argument("--output", type=Path, default=Path("models/churn_pro_engine.pth"))
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"입력 파일 없음: {args.input}")

    train_engine(args.input, args.output, epochs=args.epochs)
