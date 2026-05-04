import torch
import torch.nn as nn

class ChurnLSTM(nn.Module):
    """
    Time-Series LSTM Model for ChurnRadar Pro.
    Apple Silicon(MPS)에서 pack_padded_sequence 버그/병목 현상을 피하기 위해,
    일반 연산 후 각 유저별 '실제 마지막 시점'의 Hidden State를 직접 인덱싱하여 추출합니다.
    """
    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(ChurnLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, lengths):
        # x shape: (Batch, Time_Steps, Features)
        out, (h_n, c_n) = self.lstm(x)
        
        # MPS 버그를 회피하면서 정확히 '패딩되기 직전 실제 데이터의 마지막 시점' 추출
        batch_size = x.size(0)
        batch_indices = torch.arange(batch_size, device=x.device)
        
        # 길이가 0인 예외 방지
        valid_lengths = torch.clamp(lengths.to(x.device), min=1)
        last_valid_indices = valid_lengths - 1
        
        # out shape: (Batch, Time_Steps, Hidden_Size)
        # 각 배치별 실제 데이터 마지막 시점의 Hidden State 추출
        last_hidden = out[batch_indices, last_valid_indices, :]
        
        logits = self.classifier(last_hidden)
        return logits

if __name__ == "__main__":
    dummy_input = torch.randn(16, 30, 3)
    dummy_lengths = torch.randint(5, 30, (16,))
    model = ChurnLSTM(input_size=3, hidden_size=64, num_layers=2)
    output = model(dummy_input, dummy_lengths)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
