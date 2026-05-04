import torch
import torch.nn as nn

class ChurnLSTM(nn.Module):
    """
    Time-Series LSTM Model for ChurnRadar Pro.
    이탈 가속도와 모멘텀의 시간적 흐름을 기억하여 최종 이탈 여부를 예측합니다.
    """
    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(ChurnLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어 (batch_first=True 설정으로 입력 형태를 [Batch, Time_Steps, Features]로 받음)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Classifier 헤드
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # 이진 분류를 위한 출력 (BCEWithLogitsLoss 사용 예정이므로 Sigmoid 생략)
        )
        
    def forward(self, x):
        # x shape: (Batch, Time_Steps, Features)
        
        # LSTM 통과 -> out shape: (Batch, Time_Steps, Hidden_Size)
        # h_n, c_n shape: (Num_Layers, Batch, Hidden_Size)
        out, (h_n, c_n) = self.lstm(x)
        
        # 마지막 시점(Time_Step)의 Hidden State를 추출하여 Classifier에 전달
        # out[:, -1, :] 대신 모델의 가장 마지막 레이어의 h_n을 사용해도 됨
        last_hidden = out[:, -1, :]
        
        # 예측 (Logits)
        logits = self.classifier(last_hidden)
        return logits

if __name__ == "__main__":
    # 간단한 작동 검증 테스트
    dummy_input = torch.randn(16, 30, 3) # Batch 16, 30 Time Steps, 3 Features
    model = ChurnLSTM(input_size=3, hidden_size=64, num_layers=2)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
