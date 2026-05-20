from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.train_tcn import load_baza_csv_as_timeseries
from src.models.tcn_model import ChurnTCN
from src.utils.helpers import model_path, raw_data_path, resolve_input_path, result_path
from src.utils.threshold_optimizer import evaluate_threshold, find_best_threshold


FEATURES = [
    "Total_SUBs",
    "AvgMobileRevenue",
    "AvgFIXRevenue",
    "TotalRevenue",
    "ARPU",
    "Active_Ratio",
    "Not_Active_subscribers",
    "Mobile_Revenue_Ratio",
    "Inactive_Ratio",
    "Suspended_Ratio",
    "Revenue_per_Active_Sub",
    "Inactive_x_Revenue",
    "Revenue_Balance",
    "CRM_PID_Value_Segment",
    "EffectiveSegment",
]


def load_xgb_frame(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    mask = df["ARPU"].isna() & df["Total_SUBs"].gt(0)
    df.loc[mask, "ARPU"] = df.loc[mask, "TotalRevenue"] / df.loc[mask, "Total_SUBs"]

    df["Active_Ratio"] = df["Active_subscribers"] / df["Total_SUBs"].replace(0, np.nan)
    df["Active_Ratio"] = df["Active_Ratio"].fillna(0.0).clip(0.0, 1.0)
    df["Not_Active_subscribers"] = df["Not_Active_subscribers"].fillna(0.0)
    df["Mobile_Revenue_Ratio"] = df["AvgMobileRevenue"] / df["TotalRevenue"].replace(0, np.nan)
    df["Mobile_Revenue_Ratio"] = df["Mobile_Revenue_Ratio"].fillna(0.0).clip(0.0, 1.0)
    df["Inactive_Ratio"] = df["Not_Active_subscribers"] / df["Total_SUBs"].replace(0, np.nan)
    df["Inactive_Ratio"] = df["Inactive_Ratio"].fillna(0.0).clip(0.0, 1.0)
    if "Suspended_subscribers" not in df.columns:
        df["Suspended_subscribers"] = 0.0
    df["Suspended_subscribers"] = df["Suspended_subscribers"].fillna(0.0)
    df["Suspended_Ratio"] = df["Suspended_subscribers"] / df["Total_SUBs"].replace(0, np.nan)
    df["Suspended_Ratio"] = df["Suspended_Ratio"].fillna(0.0).clip(0.0, 1.0)
    df["Revenue_per_Active_Sub"] = df["TotalRevenue"] / df["Active_subscribers"].replace(0, np.nan)
    df["Revenue_per_Active_Sub"] = df["Revenue_per_Active_Sub"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["Inactive_x_Revenue"] = df["Inactive_Ratio"] * df["TotalRevenue"].fillna(0.0)
    revenue_pair = df[["AvgMobileRevenue", "AvgFIXRevenue"]].fillna(0.0)
    df["Revenue_Balance"] = revenue_pair.min(axis=1) / (revenue_pair.max(axis=1) + 1e-5)
    df["Revenue_Balance"] = df["Revenue_Balance"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    for col in ["CRM_PID_Value_Segment", "EffectiveSegment"]:
        df[col] = df[col].fillna("Unknown")

    y = df["CHURN"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    return df.loc[valid, FEATURES].reset_index(drop=True), y[valid].astype(int).to_numpy()


def predict_tcn_batch(
    csv_path: Path,
    validation_indices: np.ndarray,
    tcn_path: Path,
) -> np.ndarray | None:
    if not tcn_path.is_file():
        return None

    checkpoint = torch.load(tcn_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        return None

    x_np, _, lengths_np = load_baza_csv_as_timeseries(csv_path)
    x_val = x_np[validation_indices]
    l_val = lengths_np[validation_indices]

    scaler_mean = checkpoint.get("scaler_mean")
    scaler_scale = checkpoint.get("scaler_scale")
    if scaler_mean is not None and scaler_scale is not None:
        mean = np.asarray(scaler_mean, dtype=np.float32).reshape(1, 1, -1)
        scale = np.asarray(scaler_scale, dtype=np.float32).reshape(1, 1, -1)
        scale = np.where(scale == 0, 1.0, scale)
        x_val = (x_val - mean) / scale

    config = checkpoint.get("config", {})
    model = ChurnTCN(
        input_size=int(config.get("input_size", 3)),
        channels=tuple(config.get("channels", [32, 64])),
        kernel_size=int(config.get("kernel_size", 3)),
        dropout=float(config.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(
            torch.tensor(x_val, dtype=torch.float32),
            lengths=torch.tensor(l_val, dtype=torch.long),
        )
        return torch.sigmoid(logits).numpy().reshape(-1)


def optimize_threshold(
    csv_path: Path,
    xgb_model_path: Path,
    tcn_model_path: Path,
    output_path: Path,
    min_threshold: float,
    max_threshold: float,
    step: float,
    min_recall: float | None,
) -> dict[str, Any]:
    xgb_model = joblib.load(xgb_model_path)
    xgb_frame, y = load_xgb_frame(csv_path)
    indices = np.arange(len(y))
    _, validation_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    xgb_prob = xgb_model.predict_proba(xgb_frame.iloc[validation_indices])[:, 1]
    tcn_prob = predict_tcn_batch(csv_path, validation_indices, tcn_model_path)
    ts_prob = np.full_like(xgb_prob, 0.20)

    if tcn_prob is None:
        ensemble_name = "xgb_ts"
        ensemble_prob = xgb_prob * 0.8 + ts_prob * 0.2
    else:
        ensemble_name = "xgb_tcn_ts"
        ensemble_prob = xgb_prob * 0.7 + tcn_prob * 0.2 + ts_prob * 0.1

    thresholds = np.round(
        np.arange(min_threshold, max_threshold + step / 2, step),
        6,
    ).tolist()
    best = find_best_threshold(
        y[validation_indices].tolist(),
        ensemble_prob.tolist(),
        thresholds,
        min_recall=min_recall,
    )
    default_metrics = evaluate_threshold(
        y[validation_indices].tolist(),
        ensemble_prob.tolist(),
        0.5,
    )
    threshold_table = [
        evaluate_threshold(y[validation_indices].tolist(), ensemble_prob.tolist(), threshold).to_dict()
        for threshold in thresholds
    ]

    summary = {
        "model": ensemble_name,
        "selection_data": "validation",
        "validation_rows": int(len(validation_indices)),
        "positive_rows": int(y[validation_indices].sum()),
        "threshold_grid": {
            "min": min_threshold,
            "max": max_threshold,
            "step": step,
            "min_recall": min_recall,
        },
        "selected_threshold": best.threshold,
        "best": best.to_dict(),
        "default_0_5": default_metrics.to_dict(),
        "delta_best_vs_default": {
            "f1": best.f1 - default_metrics.f1,
            "recall": best.recall - default_metrics.recall,
            "precision": best.precision - default_metrics.precision,
            "tp": best.tp - default_metrics.tp,
            "fp": best.fp - default_metrics.fp,
            "fn": best.fn - default_metrics.fn,
            "tn": best.tn - default_metrics.tn,
        },
        "threshold_table": threshold_table,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize churn classification threshold")
    parser.add_argument("--csv", type=Path, default=raw_data_path("baza_telecom_v2.csv"))
    parser.add_argument("--xgb-model", type=Path, default=model_path("model.joblib"))
    parser.add_argument("--tcn-model", type=Path, default=model_path("churn_tcn.pth"))
    parser.add_argument("--output", type=Path, default=result_path("threshold_optimization_summary.json"))
    parser.add_argument("--min-threshold", type=float, default=0.20)
    parser.add_argument("--max-threshold", type=float, default=0.70)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--min-recall", type=float, default=None)
    args = parser.parse_args()

    csv_path = resolve_input_path(args.csv, raw_data_path("baza_telecom_v2.csv"))
    xgb_model_path = resolve_input_path(args.xgb_model, model_path("model.joblib"))
    tcn_model_path = resolve_input_path(args.tcn_model, model_path("churn_tcn.pth"))
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    summary = optimize_threshold(
        csv_path=csv_path,
        xgb_model_path=xgb_model_path,
        tcn_model_path=tcn_model_path,
        output_path=output_path,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
        min_recall=args.min_recall,
    )
    print("=== Threshold Optimization ===")
    print(f"Model: {summary['model']}")
    print(f"Selected threshold: {summary['selected_threshold']:.2f}")
    print(
        "Best: "
        f"F1={summary['best']['f1']:.4f}, "
        f"Recall={summary['best']['recall']:.4f}, "
        f"Precision={summary['best']['precision']:.4f}"
    )
    print(
        "Default 0.5: "
        f"F1={summary['default_0_5']['f1']:.4f}, "
        f"Recall={summary['default_0_5']['recall']:.4f}, "
        f"Precision={summary['default_0_5']['precision']:.4f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
