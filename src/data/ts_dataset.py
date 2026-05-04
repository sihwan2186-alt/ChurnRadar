import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from pathlib import Path

class ChurnTimeSeriesDataset(Dataset):
    """
    Parquet로 저장된 3D 텐서 메타데이터를 읽어 PyTorch Dataset으로 변환합니다.
    (Samples, Time_Steps, Features)
    - Data Leakage 방지를 위해 스케일링 로직을 외부로 분리했습니다.
    - LSTM pack_padded_sequence를 위해 Post-padding(뒤에 0 채우기)을 적용합니다.
    """
    def __init__(self, parquet_path: Path, max_seq_len: int = 30, target_col: str = "is_churn"):
        super().__init__()
        print(f"[Dataset] 로드 중: {parquet_path}")
        df = pl.read_parquet(parquet_path)
        
        grouped = df.group_by("Entity_ID").agg([
            pl.col("Activity_Energy"),
            pl.col("Momentum"),
            pl.col("Acceleration")
        ])
        
        self.max_seq_len = max_seq_len
        self.sequences = []
        self.targets = []
        self.lengths = []
        
        has_target = target_col is not None and target_col in df.columns
        if has_target:
            target_dict = dict(zip(df["Entity_ID"].to_list(), df[target_col].to_list()))
            
        print("[Dataset] 시퀀스 패딩(Post-Padding) 및 텐서 변환 중...")
        for row in grouped.iter_rows(named=True):
            entity_id = row["Entity_ID"]
            energy = row["Activity_Energy"]
            momentum = row["Momentum"]
            accel = row["Acceleration"]
            
            # (Time_Steps, 3) 
            seq = np.column_stack((energy, momentum, accel))
            
            actual_len = min(len(seq), self.max_seq_len)
            
            # 길이 맞추기 (Post-Padding)
            if len(seq) >= self.max_seq_len:
                seq = seq[-self.max_seq_len:]  # 최근 기록 우선
            else:
                pad_len = self.max_seq_len - len(seq)
                pad_matrix = np.zeros((pad_len, 3))
                seq = np.vstack((seq, pad_matrix))  # 뒤에 0을 채움
                
            self.sequences.append(seq)
            self.lengths.append(actual_len)
            
            if has_target:
                self.targets.append(target_dict.get(entity_id, 0))
            else:
                self.targets.append(0)
                
        self.X = np.array(self.sequences, dtype=np.float32)
        self.y = np.array(self.targets, dtype=np.float32)
        self.lens = np.array(self.lengths, dtype=np.int64)
        
        print(f"[Dataset] 준비 완료. 텐서 형태: {self.X.shape}")

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx]), torch.tensor(self.lens[idx])
