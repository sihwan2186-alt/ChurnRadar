# scripts/train_ts_engine.py
"""
ChurnRadar Pro — TS-Transformer + TS-SMOTE 학습 파이프라인 (Baza 전용)

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. python scripts/train_ts_engine.py \
        --epochs 50 \
        --weight_scale 0.9 \
        --batch_size 512
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

# 현재 스크립트 위치(scripts/)의 부모 폴더(프로젝트 루트)를 경로에 추가
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.utils.helpers import model_path, processed_data_path, resolve_input_path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.ts_dataset import ChurnTimeSeriesDataset
from src.models.ts_smote import TSSMOTE
from src.models.ts_transformer import ChurnTransformer


def train_engine(
    parquet_path: Path,
    model_out: Path,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_scale: float = 0.9,
    use_smote: bool = True,
) -> None:
    print("=" * 60)
    print("🚀 ChurnRadar Pro - Transformer Engine Training (Baza)")
    print("=" * 60)

    # ── 디바이스 설정 ────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 하드웨어 가속기 감지됨: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ CUDA GPU 감지됨")
    else:
        device = torch.device("cpu")
        print("⚠️  CPU 모드로 실행")

    # ── 1. 데이터 로드 ───────────────────────────────────────────────────
    dataset = ChurnTimeSeriesDataset(parquet_path, max_seq_len=30, target_col="is_churn")
    X_np = dataset.X          # (N, 30, 3) float32
    y_np = dataset.y          # (N,)       float32
    l_np = dataset.lens       # (N,)       int64

    print(f"\n[Dataset] 총 유저: {len(y_np):,}명 | 이탈률: {y_np.mean():.2%}")

    # ── 2. Train / Validation 분리 (Leakage 원천 차단) ──────────────────
    print("[Split] 훈련(Train) 및 검증(Val) 80:20 분할 중...")
    X_train, X_val, y_train, y_val, l_train, l_val = train_test_split(
        X_np, y_np, l_np, test_size=0.2, random_state=42, stratify=y_np
    )
    print(f"  Train: {len(y_train):,}명 | Val: {len(y_val):,}명")
    print(f"  Train 이탈: {int(y_train.sum())}명 ({y_train.mean():.2%})")

    # ── 3. TS-SMOTE (Train Only) ─────────────────────────────────────────
    if use_smote:
        print("\n[TS-SMOTE] 소수 클래스 오버샘플링 시작...")
        print(f"  적용 전 → 0: {int((y_train == 0).sum())}, 1: {int((y_train == 1).sum())}")
        smote = TSSMOTE(k_neighbors=5, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train.astype(int))
        y_train = y_train.astype(np.float32)

        # SMOTE 후 lengths는 30으로 패딩 (합성 샘플은 전체 길이)
        n_synthetic = len(y_train) - len(l_train)
        l_synthetic = np.full(n_synthetic, 30, dtype=np.int64)
        l_train = np.concatenate([l_train, l_synthetic], axis=0)
        print(f"  적용 후 → 0: {int((y_train == 0).sum())}, 1: {int((y_train == 1).sum())}")

    # ── 4. 스케일링 (Train 통계량만 사용) ───────────────────────────────
    print("\n[Scaling] StandardScaler 적용 (Leakage 차단)...")
    scaler = StandardScaler()
    N_tr, T, F = X_train.shape
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N_tr, T, F)

    N_val = X_val.shape[0]
    X_val_scaled = scaler.transform(X_val.reshape(-1, F)).reshape(N_val, T, F)

    # DataLoader
    train_ds = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(l_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(l_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── 5. 모델 / 손실함수 / 옵티마이저 ─────────────────────────────────
    model = ChurnTransformer(input_size=3, d_model=64, nhead=4, num_layers=2).to(device)
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train)) - n_pos
    pos_weight_val = (n_neg / (n_pos + 1e-5)) * weight_scale
    print(f"\n[Loss] pos_weight = {pos_weight_val:.3f} "
          f"(neg/pos={n_neg/n_pos:.1f} × scale={weight_scale})")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # ── 6. 학습 루프 ─────────────────────────────────────────────────────
    print(f"\n🔥 학습 시작 (Epochs: {epochs}, Batch: {batch_size})")
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        epoch_loss = 0.0

        for batch_X, batch_y, batch_lengths in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            b_size, seq_len = batch_X.size(0), batch_X.size(1)
            padding_mask = (
                torch.arange(seq_len, device=device)
                .unsqueeze(0)
                .expand(b_size, seq_len)
                >= batch_lengths.unsqueeze(1).to(device)
            )

            optimizer.zero_grad()
            logits = model(batch_X, padding_mask=padding_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # ── 검증 루프 ────────────────────────────────────────────────────
        model.eval()
        val_probs: list[float] = []
        val_targets: list[float] = []

        with torch.no_grad():
            for batch_X, batch_y, batch_lengths in val_loader:
                batch_X = batch_X.to(device)

                b_size, seq_len = batch_X.size(0), batch_X.size(1)
                padding_mask = (
                    torch.arange(seq_len, device=device)
                    .unsqueeze(0)
                    .expand(b_size, seq_len)
                    >= batch_lengths.unsqueeze(1).to(device)
                )

                logits = model(batch_X, padding_mask=padding_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs.flatten().tolist())
                val_targets.extend(batch_y.numpy().flatten().tolist())

        val_probs_arr = np.array(val_probs)
        val_targets_arr = np.array(val_targets)

        # threshold 자동 탐색 (0.20 ~ 0.65)
        best_f1, best_thresh = 0.0, 0.5
        for th in np.arange(0.20, 0.65, 0.05):
            preds = (val_probs_arr >= th).astype(int)
            f = f1_score(val_targets_arr, preds, zero_division=0)
            if f > best_f1:
                best_f1, best_thresh = f, float(th)

        recall = recall_score(
            val_targets_arr,
            (val_probs_arr >= best_thresh).astype(int),
            zero_division=0,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        print(
            f"  [Epoch {epoch+1:>3}/{epochs}] "
            f"LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | "
            f"Thresh: {best_thresh:.2f} | Recall: {recall:.4f} | F1: {best_f1:.4f} "
            f"({elapsed:.1f}초)"
        )
        scheduler.step()

    print(f"\n✅ 전체 학습 완료!")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"저장 완료: {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChurnRadar TS-Transformer 학습")
    parser.add_argument(
        "--input", type=Path,
        default=processed_data_path("baza_ts.parquet"),
        help="학습용 Parquet 경로"
    )
    parser.add_argument(
        "--output", type=Path,
        default=model_path("churn_pro_engine.pth"),
        help="모델 저장 경로"
    )
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=512)
    parser.add_argument("--weight_scale", type=float, default=0.9)
    parser.add_argument("--no_smote",     action="store_true", help="TS-SMOTE 비활성화")
    args = parser.parse_args()

    args.input = resolve_input_path(args.input, processed_data_path("baza_ts.parquet"))
    if not args.output.is_absolute():
        args.output = REPO_ROOT / args.output

    if not args.input.is_file():
        raise SystemExit(f"입력 파일 없음: {args.input}")
    train_engine(
        parquet_path=args.input,
        model_out=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weight_scale=args.weight_scale,
        use_smote=not args.no_smote,
    )
