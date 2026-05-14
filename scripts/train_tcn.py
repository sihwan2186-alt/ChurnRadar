from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.models.tcn_model import ChurnTCN, count_parameters, normalize_channels
from src.utils.helpers import model_path, processed_data_path, raw_data_path, resolve_input_path, result_path


RANDOM_SEED = 42
NOISE_STD = 0.05


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_padding_mask(lengths: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    return (
        torch.arange(seq_len, device=device)
        .unsqueeze(0)
        .expand(lengths.size(0), seq_len)
        >= lengths.to(device).unsqueeze(1)
    )


def load_parquet_timeseries(parquet_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from src.data.ts_dataset import ChurnTimeSeriesDataset

    dataset = ChurnTimeSeriesDataset(parquet_path, max_seq_len=30, target_col="is_churn")
    return dataset.X, dataset.y, dataset.lens


def load_baza_csv_as_timeseries(
    csv_path: Path,
    time_steps: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    churn = (
        df["CHURN"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
    valid = churn.notna()
    df = df.loc[valid].copy()
    y = churn.loc[valid].astype(np.float32).to_numpy()

    total_subs = pd.to_numeric(df["Total_SUBs"], errors="coerce").fillna(0.0)
    safe_total = total_subs.replace(0, 1.0)
    arpu = pd.to_numeric(df["ARPU"], errors="coerce")
    total_revenue = pd.to_numeric(df["TotalRevenue"], errors="coerce").fillna(0.0)
    arpu = arpu.fillna(total_revenue / safe_total).fillna(0.0).clip(lower=0.0)
    active = pd.to_numeric(df["Active_subscribers"], errors="coerce").fillna(0.0)
    inactive = pd.to_numeric(df["Not_Active_subscribers"], errors="coerce").fillna(0.0)

    base_energy = arpu.to_numpy(dtype=np.float32)
    base_momentum = (active / safe_total).clip(0.0, 1.0).to_numpy(dtype=np.float32)
    base_acceleration = (inactive / safe_total).clip(0.0, 1.0).to_numpy(dtype=np.float32)

    x = np.zeros((len(df), time_steps, 3), dtype=np.float32)
    for idx, (energy, momentum, acceleration) in enumerate(
        zip(base_energy, base_momentum, base_acceleration)
    ):
        noise = rng.normal(1.0, NOISE_STD, size=(time_steps, 3)).astype(np.float32)
        x[idx, :, 0] = np.clip(energy * noise[:, 0], 0.0, None)
        x[idx, :, 1] = np.clip(momentum * noise[:, 1], 0.0, 1.0)
        x[idx, :, 2] = np.clip(acceleration * noise[:, 2], 0.0, 1.0)

    lengths = np.full(len(df), time_steps, dtype=np.int64)
    return x, y, lengths


def load_timeseries(input_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if input_path.suffix.lower() == ".parquet":
        x, y, lengths = load_parquet_timeseries(input_path)
        return x, y, lengths, "parquet"
    if input_path.suffix.lower() == ".csv":
        x, y, lengths = load_baza_csv_as_timeseries(input_path)
        return x, y, lengths, "csv_synthetic_30day"
    raise ValueError(f"Unsupported TCN input format: {input_path}")


def evaluate_model(
    model: ChurnTCN,
    loader: DataLoader,
    device: torch.device,
    threshold_grid: np.ndarray,
) -> dict[str, Any]:
    model.eval()
    probs: list[float] = []
    targets: list[float] = []

    with torch.no_grad():
        for batch_x, batch_y, batch_lengths in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x, lengths=batch_lengths.to(device))
            batch_probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs.extend(batch_probs.tolist())
            targets.extend(batch_y.numpy().reshape(-1).tolist())

    probs_arr = np.asarray(probs)
    targets_arr = np.asarray(targets)

    best = {
        "threshold": 0.5,
        "f1": 0.0,
        "recall": 0.0,
        "precision": 0.0,
    }
    for threshold in threshold_grid:
        pred = (probs_arr >= threshold).astype(int)
        f1 = f1_score(targets_arr, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "recall": float(recall_score(targets_arr, pred, zero_division=0)),
                "precision": float(precision_score(targets_arr, pred, zero_division=0)),
            }

    return best


def train_tcn(
    parquet_path: Path,
    model_out: Path,
    summary_out: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_scale: float,
    channels: tuple[int, ...],
    kernel_size: int,
    dropout: float,
    use_smote: bool,
) -> dict[str, Any]:
    device = select_device()
    start = time.time()
    print("=" * 60)
    print("ChurnRadar - TCN Training")
    print("=" * 60)
    print(f"Device: {device}")

    x_np, y_np, lengths_np, input_format = load_timeseries(parquet_path)
    print(f"Dataset: {len(y_np):,} customers | churn rate: {y_np.mean():.2%}")
    print(f"Input format: {input_format}")

    x_train, x_val, y_train, y_val, l_train, l_val = train_test_split(
        x_np,
        y_np,
        lengths_np,
        test_size=0.2,
        random_state=42,
        stratify=y_np,
    )

    if use_smote:
        from src.models.ts_smote import TSSMOTE

        smote = TSSMOTE(k_neighbors=5, random_state=42)
        original_train_size = len(y_train)
        x_train, y_train = smote.fit_resample(x_train, y_train.astype(int))
        y_train = y_train.astype(np.float32)
        synthetic_count = len(y_train) - original_train_size
        l_train = np.concatenate(
            [l_train, np.full(synthetic_count, x_train.shape[1], dtype=np.int64)]
        )
    else:
        synthetic_count = 0

    scaler = StandardScaler()
    n_train, seq_len, feature_count = x_train.shape
    x_train_scaled = scaler.fit_transform(
        x_train.reshape(-1, feature_count)
    ).reshape(n_train, seq_len, feature_count)
    x_val_scaled = scaler.transform(
        x_val.reshape(-1, feature_count)
    ).reshape(x_val.shape[0], seq_len, feature_count)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
            torch.tensor(l_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
            torch.tensor(l_val, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ChurnTCN(
        input_size=feature_count,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)
    print(f"TCN channels: {channels} | parameters: {count_parameters(model):,}")

    positive_count = float(y_train.sum())
    negative_count = float(len(y_train)) - positive_count
    pos_weight = (negative_count / max(positive_count, 1.0)) * weight_scale
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    threshold_grid = np.arange(0.20, 0.70, 0.05)

    best_metrics: dict[str, Any] = {
        "threshold": 0.5,
        "f1": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "epoch": 0,
    }
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y, batch_lengths in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x, lengths=batch_lengths.to(device))
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        metrics = evaluate_model(model, val_loader, device, threshold_grid)
        if metrics["f1"] >= best_metrics["f1"]:
            best_metrics = {**metrics, "epoch": epoch}
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss={epoch_loss / max(len(train_loader), 1):.4f} | "
            f"val_f1={metrics['f1']:.4f} | "
            f"val_recall={metrics['recall']:.4f} | "
            f"threshold={metrics['threshold']:.2f}"
        )

    checkpoint = {
        "model_state_dict": best_state_dict,
        "config": {
            "input_size": feature_count,
            "channels": list(channels),
            "kernel_size": kernel_size,
            "dropout": dropout,
        },
        "threshold": best_metrics["threshold"],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "max_seq_len": seq_len,
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_out)

    summary = {
        "model": "TCN",
        "input_format": input_format,
        "input_shape": [int(len(y_np)), int(seq_len), int(feature_count)],
        "channels": list(channels),
        "kernel_size": kernel_size,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_scale": weight_scale,
        "use_smote": use_smote,
        "synthetic_samples": int(synthetic_count),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "parameter_count": int(count_parameters(model)),
        "best_validation": best_metrics,
        "elapsed_seconds": round(time.time() - start, 3),
        "model_path": str(model_out),
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved model: {model_out}")
    print(f"Saved summary: {summary_out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TCN churn model")
    parser.add_argument("--input", type=Path, default=processed_data_path("baza_ts.parquet"))
    parser.add_argument("--output", type=Path, default=model_path("churn_tcn.pth"))
    parser.add_argument("--summary", type=Path, default=result_path("tcn_training_summary.json"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_scale", type=float, default=0.9)
    parser.add_argument("--channels", type=str, default="32,64")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no_smote", action="store_true")
    args = parser.parse_args()

    input_path = resolve_input_path(
        args.input,
        processed_data_path("baza_ts.parquet"),
        raw_data_path("baza_telecom_v2.csv"),
    )
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    summary_path = args.summary if args.summary.is_absolute() else REPO_ROOT / args.summary

    if not input_path.is_file():
        raise SystemExit(f"Input parquet not found: {input_path}")

    train_tcn(
        parquet_path=input_path,
        model_out=output_path,
        summary_out=summary_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_scale=args.weight_scale,
        channels=normalize_channels(args.channels),
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_smote=not args.no_smote,
    )


if __name__ == "__main__":
    main()
