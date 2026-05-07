# src/models/ts_smote.py
"""
Time-Series SMOTE (TS-SMOTE)
다변량 시계열 데이터(3D Tensor: [Samples, Time_Steps, Features])에 대하여
DTW(Dynamic Time Warping) 기반의 보간을 수행하여 소수 클래스를 오버샘플링합니다.

v2: 랜덤 선택 → KNN(NearestNeighbors) 기반으로 교체.
    idx1의 k_neighbors 이웃 중에서 idx2를 선택해 합성 노이즈를 줄임.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tslearn.metrics import dtw_path


class TSSMOTE:
    """
    KNN 기반 TS-SMOTE.

    Args:
        k_neighbors (int): 이웃 수 (기본값 5). 소수 클래스 수보다 작아야 함.
        random_state (int): 재현성 시드.
    """

    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        np.random.seed(self.random_state)

    def _dtw_interpolate(self, seq1: np.ndarray, seq2: np.ndarray, alpha: float) -> np.ndarray:
        """
        두 시계열 시퀀스(seq1, seq2) 사이를 DTW 경로를 따라 보간합니다.
        seq1, seq2: (Time_Steps, Features)
        반환: 보간된 시퀀스 (Time_Steps, Features)
        """
        path, _ = dtw_path(seq1, seq2)
        new_seq = np.zeros_like(seq1, dtype=np.float32)

        for i, j in path:
            val = seq1[i] + alpha * (seq2[j] - seq1[i])
            if i < len(new_seq) and np.all(new_seq[i] == 0):
                new_seq[i] = val

        # 비어있는 timestep → Forward Fill
        for t in range(1, len(new_seq)):
            if np.all(new_seq[t] == 0) and not np.all(new_seq[t - 1] == 0):
                new_seq[t] = new_seq[t - 1]

        return new_seq

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        X: (N, Time_Steps, Features)  — 3D numpy array
        y: (N,)                        — 1D 레이블 배열
        반환: (X_resampled, y_resampled) — 소수 클래스 오버샘플링 완료
        """
        if X.ndim != 3:
            raise ValueError(
                f"입력 데이터는 3차원(Samples, Time_Steps, Features)이어야 합니다. 현재: {X.shape}"
            )

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return X, y

        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]

        max_count = int(np.max(counts))
        min_count = int(np.min(counts))
        n_synthetic = max_count - min_count

        print(f"[TS-SMOTE] 소수 클래스: {minority_class} ({min_count}개) → "
              f"다수 클래스: {majority_class} ({max_count}개) | 생성 목표: {n_synthetic}개")

        minority_X = X[y == minority_class]

        # ── KNN 학습 (2D flatten) ─────────────────────────────────────────
        k = min(self.k_neighbors + 1, len(minority_X))  # 자기 자신 포함
        minority_X_flat = minority_X.reshape(len(minority_X), -1)

        knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        knn.fit(minority_X_flat)
        # neighbors[i] = [자기 자신, 이웃1, 이웃2, ...]
        neighbors = knn.kneighbors(minority_X_flat, return_distance=False)

        # ── 합성 샘플 생성 ────────────────────────────────────────────────
        synthetic_X: list[np.ndarray] = []
        synthetic_y: list[int] = []

        for _ in range(n_synthetic):
            idx1 = np.random.randint(0, len(minority_X))
            # 자기 자신(neighbors[idx1][0]) 제외하고 이웃 중 랜덤 선택
            neighbor_indices = neighbors[idx1][1:]  # 이웃만 (자기 자신 제외)
            idx2 = int(np.random.choice(neighbor_indices))

            seq1 = minority_X[idx1]
            seq2 = minority_X[idx2]
            alpha = np.random.uniform(0, 1)

            new_seq = self._dtw_interpolate(seq1, seq2, alpha)
            synthetic_X.append(new_seq)
            synthetic_y.append(int(minority_class))

        if synthetic_X:
            X_syn = np.array(synthetic_X, dtype=np.float32)
            y_syn = np.array(synthetic_y, dtype=y.dtype)
            X_resampled = np.concatenate([X, X_syn], axis=0)
            y_resampled = np.concatenate([y, y_syn], axis=0)
            print(f"[TS-SMOTE] 완료 → 총 {len(y_resampled)}개 "
                  f"(0: {np.sum(y_resampled == 0)}, 1: {np.sum(y_resampled == 1)})")
            return X_resampled, y_resampled

        return X, y


if __name__ == "__main__":
    # 작동 검증 테스트
    dummy_X = np.random.rand(12, 5, 3).astype(np.float32)
    dummy_y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

    print(f"Original: {dummy_X.shape}, counts: 0→{np.sum(dummy_y==0)}, 1→{np.sum(dummy_y==1)}")
    ts_smote = TSSMOTE(k_neighbors=2)
    rx, ry = ts_smote.fit_resample(dummy_X, dummy_y)
    print(f"Resampled: {rx.shape}, counts: 0→{np.sum(ry==0)}, 1→{np.sum(ry==1)}")
