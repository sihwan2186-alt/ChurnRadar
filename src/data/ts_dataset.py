from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


DEFAULT_FEATURE_COLUMNS = [
    "Activity_Energy",
    "Momentum",
    "Acceleration",
    "Skip_Rate",
    "Completion_Rate",
    "Diversity_Score",
]


class ChurnTimeSeriesDataset(Dataset):
    """Load customer time-series parquet as a padded PyTorch dataset.

    The loader keeps the original 3-channel tensor path working, while also
    accepting the added KKBox behavior channels when they are present.
    """

    def __init__(
        self,
        parquet_path: Path,
        max_seq_len: int = 30,
        target_col: str = "is_churn",
        feature_cols: list[str] | None = None,
    ):
        super().__init__()
        print(f"[Dataset] Loading: {parquet_path}")
        df = pl.read_parquet(parquet_path)

        requested_cols = feature_cols or DEFAULT_FEATURE_COLUMNS
        self.feature_cols = [col for col in requested_cols if col in df.columns]
        if not self.feature_cols:
            raise ValueError(f"No time-series feature columns found in {parquet_path}")

        grouped = df.sort("Event_Time").group_by("Entity_ID", maintain_order=True).agg(
            [pl.col(col) for col in self.feature_cols]
        )

        self.max_seq_len = max_seq_len
        self.sequences = []
        self.targets = []
        self.lengths = []

        has_target = target_col is not None and target_col in df.columns
        if has_target:
            target_dict = dict(zip(df["Entity_ID"].to_list(), df[target_col].to_list()))

        print("[Dataset] Building padded tensors...")
        for row in grouped.iter_rows(named=True):
            entity_id = row["Entity_ID"]
            seq = np.column_stack([row[col] for col in self.feature_cols])

            actual_len = min(len(seq), self.max_seq_len)
            if len(seq) >= self.max_seq_len:
                seq = seq[-self.max_seq_len:]
            else:
                pad_len = self.max_seq_len - len(seq)
                pad_matrix = np.zeros((pad_len, len(self.feature_cols)))
                seq = np.vstack((seq, pad_matrix))

            self.sequences.append(seq)
            self.lengths.append(actual_len)

            if has_target:
                self.targets.append(target_dict.get(entity_id, 0))
            else:
                self.targets.append(0)

        self.X = np.array(self.sequences, dtype=np.float32)
        self.y = np.array(self.targets, dtype=np.float32)
        self.lens = np.array(self.lengths, dtype=np.int64)

        print(f"[Dataset] Ready. tensor_shape={self.X.shape} features={self.feature_cols}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx]), torch.tensor(self.lens[idx])
