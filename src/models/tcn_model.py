from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Remove right-side padding so dilated Conv1d stays causal."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class ChurnTCN(nn.Module):
    """
    Lightweight Temporal Convolutional Network for 30-day churn signals.

    Input shape follows the existing time-series pipeline:
    (batch, time_steps, features) = (N, 30, 3).
    """

    def __init__(
        self,
        input_size: int = 3,
        channels: Sequence[int] = (32, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2 for temporal patterns")
        if len(channels) == 0:
            raise ValueError("channels must contain at least one layer size")

        layers: list[nn.Module] = []
        current_channels = input_size
        for layer_idx, out_channels in enumerate(channels):
            dilation = 2**layer_idx
            layers.append(
                TemporalBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            current_channels = out_channels

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(current_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Conv1d expects (batch, features, time).
        out = self.network(x.transpose(1, 2)).transpose(1, 2)

        if padding_mask is None and lengths is not None:
            seq_len = x.size(1)
            padding_mask = (
                torch.arange(seq_len, device=x.device)
                .unsqueeze(0)
                .expand(x.size(0), seq_len)
                >= lengths.to(x.device).unsqueeze(1)
            )

        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            pooled = (out * valid_mask).sum(dim=1)
            denom = valid_mask.sum(dim=1).clamp(min=1.0)
            pooled = pooled / denom
        else:
            pooled = out.mean(dim=1)

        return self.classifier(pooled)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def normalize_channels(channels: Iterable[int] | str) -> tuple[int, ...]:
    if isinstance(channels, str):
        return tuple(int(value.strip()) for value in channels.split(",") if value.strip())
    return tuple(int(value) for value in channels)


if __name__ == "__main__":
    dummy_input = torch.randn(16, 30, 3)
    dummy_lengths = torch.randint(5, 31, (16,))
    model = ChurnTCN(input_size=3, channels=(32, 64))
    output = model(dummy_input, lengths=dummy_lengths)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {count_parameters(model):,}")
