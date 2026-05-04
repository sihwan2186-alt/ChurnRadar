import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class ChurnTimeSeriesDataset(Dataset):
    """
    Parquet로 저장된 3D 텐서 메타데이터를 읽어 PyTorch Dataset으로 변환합니다.
    (Samples, Time_Steps, Features)
    """
    def __init__(self, parquet_path: Path, max_seq_len: int = 30, target_col: str = "is_churn"):
        super().__init__()
        print(f"[Dataset] 로드 중: {parquet_path}")
        df = pl.read_parquet(parquet_path)
        
        # Entity_ID 별로 그룹화하여 시퀀스 리스트 생성
        # 피처: Activity_Energy, Momentum, Acceleration
        grouped = df.group_by("Entity_ID").agg([
            pl.col("Activity_Energy"),
            pl.col("Momentum"),
            pl.col("Acceleration")
        ])
        
        self.max_seq_len = max_seq_len
        self.sequences = []
        self.targets = []
        
        # 임시 라벨 할당 (타겟 컬럼이 명시되지 않은 KKBox, H&M은 우선 전부 0으로 취급, 혹은 향후 매핑)
        # Baza 데이터를 활용할 때는 target_col을 명시하여 활용
        has_target = target_col is not None and target_col in df.columns
        if has_target:
            target_dict = dict(zip(df["Entity_ID"].to_list(), df[target_col].to_list()))
            
        print("[Dataset] 시퀀스 패딩(Padding) 및 텐서 변환 중...")
        for row in grouped.iter_rows(named=True):
            entity_id = row["Entity_ID"]
            energy = row["Activity_Energy"]
            momentum = row["Momentum"]
            accel = row["Acceleration"]
            
            # (Time_Steps, 3) 
            seq = np.column_stack((energy, momentum, accel))
            
            # 길이 맞추기 (Truncate or Pad)
            if len(seq) >= self.max_seq_len:
                seq = seq[-self.max_seq_len:]  # 최근 기록 우선
            else:
                pad_len = self.max_seq_len - len(seq)
                pad_matrix = np.zeros((pad_len, 3))
                seq = np.vstack((pad_matrix, seq))
                
            self.sequences.append(seq)
            if has_target:
                self.targets.append(target_dict.get(entity_id, 0))
            else:
                # 외부 타겟 파일 조인이 필요하지만 우선 0 (더미)
                self.targets.append(0)
                
        self.X = np.array(self.sequences, dtype=np.float32)
        self.y = np.array(self.targets, dtype=np.float32)
        
        # X의 스케일링 (전체 데이터에 대해 z-score 정규화)
        # shape: (N, Time_Steps, Features) -> (N*Time_Steps, Features) -> (N, Time_Steps, Features)
        N, T, F = self.X.shape
        self.scaler = StandardScaler()
        X_flat = self.X.reshape(-1, F)
        X_flat_scaled = self.scaler.fit_transform(X_flat)
        self.X = X_flat_scaled.reshape(N, T, F)
        
        print(f"[Dataset] 준비 완료. 텐서 형태: {self.X.shape}")

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def get_dataloader(parquet_path: Path, batch_size: int = 64, max_seq_len: int = 30) -> DataLoader:
    dataset = ChurnTimeSeriesDataset(parquet_path, max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
