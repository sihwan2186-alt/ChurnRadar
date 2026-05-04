import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.data.ts_dataset import ChurnTimeSeriesDataset
from src.models.ts_transformer import ChurnTransformer

def train_engine(parquet_path: Path, model_out: Path, epochs: int = 5, batch_size: int = 64, lr: float = 1e-3):
    print("=" * 60)
    print("🚀 ChurnRadar Pro - Transformer Engine Training (Attention)")
    print("=" * 60)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 하드웨어 가속기 감지됨: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 1. 데이터 로드
    dataset = ChurnTimeSeriesDataset(parquet_path, max_seq_len=30, target_col="is_churn")
    X_np = dataset.X
    y_np = dataset.y
    l_np = dataset.lens
    
    print(f"\n[Dataset] 총 유저: {len(y_np)}명, 이탈률: {y_np.mean():.2%}")

    # 2. 물리적 분리 (Train / Validation) - 누수 원천 차단
    print("[Split] 훈련(Train) 및 검증(Val) 80:20 분할 중...")
    X_train, X_val, y_train, y_val, l_train, l_val = train_test_split(
        X_np, y_np, l_np, test_size=0.2, random_state=42, stratify=y_np
    )
    
    # 3. 올바른 스케일링 (Train에서 획득한 통계량만 사용)
    print("[Scaling] 훈련 세트 기준으로 StandardScaler 적용 (Leakage 차단 완수)...")
    scaler = StandardScaler()
    
    N_tr, T, F = X_train.shape
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N_tr, T, F)
    
    N_val = X_val.shape[0]
    X_val_scaled = scaler.transform(X_val.reshape(-1, F)).reshape(N_val, T, F)
    
    # DataLoader 변환
    train_dataset = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train).unsqueeze(1), torch.tensor(l_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.tensor(X_val_scaled), torch.tensor(y_val).unsqueeze(1), torch.tensor(l_val))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. 모델 생성 및 디바이스 이동 (Transformer로 교체)
    model = ChurnTransformer(input_size=3, d_model=64, nhead=4, num_layers=2).to(device)
    
    # 편향 완화: Random Oversampling 제거 후 보수적인 pos_weight 계산
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight_val = n_neg / (n_pos + 1e-5)
    print(f"\n[Weighted Loss] 소수 클래스 가중치(pos_weight) 적용: {pos_weight_val:.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # 5. 훈련 루프
    print(f"\n🔥 학습 시작 (Total Epochs: {epochs}, Batch Size: {batch_size})")
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        epoch_loss = 0.0
        
        for batch_X, batch_y, batch_lengths in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Transformer를 위한 Padding Mask 생성 (True = Padding)
            b_size, seq_len = batch_X.size(0), batch_X.size(1)
            padding_mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(b_size, seq_len) >= batch_lengths.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X, padding_mask=padding_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        
        # 검증 루프
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_X, batch_y, batch_lengths in val_loader:
                batch_X = batch_X.to(device)
                
                b_size, seq_len = batch_X.size(0), batch_X.size(1)
                padding_mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(b_size, seq_len) >= batch_lengths.unsqueeze(1).to(device)
                
                logits = model(batch_X, padding_mask=padding_mask)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.numpy())
                
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        recall = recall_score(val_targets, val_preds, zero_division=0)
        f1 = f1_score(val_targets, val_preds, zero_division=0)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  [Epoch {epoch+1}/{epochs}] LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f} ({time.time() - start_time:.1f}초)")
        scheduler.step()
        
    print(f"\n✅ 전체 학습 완료!")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"저장 완료: {model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/kkbox_real_gold_v1.parquet"))
    parser.add_argument("--output", type=Path, default=Path("models/churn_pro_engine.pth"))
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    if not args.input.is_file():
        raise SystemExit(f"입력 파일 없음: {args.input}")
    train_engine(args.input, args.output, epochs=args.epochs)
