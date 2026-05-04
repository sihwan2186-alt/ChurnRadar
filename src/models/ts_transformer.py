import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ChurnTransformer(nn.Module):
    """
    Time-Series Transformer Model for ChurnRadar Pro.
    과거 30일치 데이터 전체를 내려다보며 특정 행동의 맥락과 가중치를 깨우칩니다.
    """
    def __init__(self, input_size: int = 3, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super(ChurnTransformer, self).__init__()
        
        # 1. Input Projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer Encoder (batch_first=True)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. Global Pooling & Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, padding_mask=None):
        # x shape: (Batch, Seq_Len, Features)
        # padding_mask shape: (Batch, Seq_Len) -> True for padding elements
        
        # Projection & Positional Encoding
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        out = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Global Average Pooling (패딩된 부분 무시)
        if padding_mask is not None:
            # ~padding_mask = True for valid elements
            valid_mask = (~padding_mask).unsqueeze(-1).float()  # (Batch, Seq_Len, 1)
            sum_out = (out * valid_mask).sum(dim=1)  # (Batch, d_model)
            valid_lengths = valid_mask.sum(dim=1).clamp(min=1)  # (Batch, 1)
            pooled_out = sum_out / valid_lengths
        else:
            pooled_out = out.mean(dim=1)
            
        logits = self.classifier(pooled_out)
        return logits

if __name__ == "__main__":
    dummy_input = torch.randn(16, 30, 3)
    # 패딩 마스크 생성 (True가 패딩)
    dummy_lengths = torch.randint(5, 30, (16,))
    batch_size, seq_len = dummy_input.size(0), dummy_input.size(1)
    mask = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len) >= dummy_lengths.unsqueeze(1)
    
    model = ChurnTransformer()
    output = model(dummy_input, padding_mask=mask)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {output.shape}")
