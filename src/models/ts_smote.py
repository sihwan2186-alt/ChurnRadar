import numpy as np
from tslearn.metrics import dtw_path
import random

class TSSMOTE:
    """
    Time-Series SMOTE (TS-SMOTE)
    다변량 시계열 데이터(3D Tensor: [Samples, Time_Steps, Features])에 대하여
    DTW(Dynamic Time Warping) 기반의 보간을 수행하여 소수 클래스를 오버샘플링합니다.
    """
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
    def _dtw_interpolate(self, seq1: np.ndarray, seq2: np.ndarray, alpha: float) -> np.ndarray:
        """
        두 시계열 시퀀스(seq1, seq2) 사이를 DTW 경로를 따라 보간합니다.
        seq1, seq2: (Time_Steps, Features)
        """
        path, _ = dtw_path(seq1, seq2)
        new_seq = np.zeros_like(seq1)
        
        # DTW 경로 상의 점들을 매핑하여 길이 복원 (단순화 버전)
        # 실제 구현에서는 경로의 평균적 길이나 고정 길이로 리샘플링
        for i, j in path:
            # 보간: alpha * seq1 + (1-alpha) * seq2
            val = seq1[i] + alpha * (seq2[j] - seq1[i])
            # 결과 배열 길이가 원본(seq1)과 같도록 간략히 매핑 (가장 첫 매칭 우선)
            if i < len(new_seq) and np.all(new_seq[i] == 0):
                new_seq[i] = val
                
        # 비어있는 점들은 이전 값으로 채움 (Forward Fill)
        for t in range(1, len(new_seq)):
            if np.all(new_seq[t] == 0) and not np.all(new_seq[t-1] == 0):
                new_seq[t] = new_seq[t-1]
                
        return new_seq

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        X: (N, Time_Steps, Features)
        y: (N,)
        """
        if len(X.shape) != 3:
            raise ValueError(f"입력 데이터는 3차원(Samples, Time_Steps, Features)이어야 합니다. 현재: {X.shape}")
            
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return X, y
            
        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]
        
        max_count = np.max(counts)
        min_count = np.min(counts)
        n_synthetic = max_count - min_count
        
        print(f"[TS-SMOTE] 생성할 합성 데이터 개수: {n_synthetic} (다수: {max_count}, 소수: {min_count})")
        
        minority_X = X[y == minority_class]
        synthetic_X = []
        synthetic_y = []
        
        for _ in range(n_synthetic):
            # 랜덤하게 두 소수 클래스 샘플 선택 (실제로는 KNN 기반 선택 권장)
            idx1, idx2 = np.random.choice(len(minority_X), 2, replace=False)
            seq1 = minority_X[idx1]
            seq2 = minority_X[idx2]
            
            # 난수 alpha로 보간율 결정
            alpha = np.random.uniform(0, 1)
            
            # DTW 기반 보간
            new_seq = self._dtw_interpolate(seq1, seq2, alpha)
            synthetic_X.append(new_seq)
            synthetic_y.append(minority_class)
            
        if synthetic_X:
            synthetic_X = np.array(synthetic_X)
            synthetic_y = np.array(synthetic_y)
            X_resampled = np.concatenate([X, synthetic_X], axis=0)
            y_resampled = np.concatenate([y, synthetic_y], axis=0)
            return X_resampled, y_resampled
        else:
            return X, y

if __name__ == "__main__":
    # 간단한 작동 검증 테스트
    dummy_X = np.random.rand(10, 5, 2)  # 10개 샘플, 5스텝, 2피처
    dummy_y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 0이 다수, 1이 소수
    
    print(f"Original shape: {dummy_X.shape}")
    ts_smote = TSSMOTE()
    resampled_X, resampled_y = ts_smote.fit_resample(dummy_X, dummy_y)
    print(f"Resampled shape: {resampled_X.shape}")
    print(f"Resampled counts: 0 -> {np.sum(resampled_y==0)}, 1 -> {np.sum(resampled_y==1)}")
