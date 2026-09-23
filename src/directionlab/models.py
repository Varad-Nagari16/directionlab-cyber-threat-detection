"""Small causal TCN with static encoder and gated fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        # Pad only on the left: output at t cannot depend on x[t+1:].
        x = nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size=3, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size=3, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        y = self.net(x)
        y = (y + residual).transpose(1, 2)
        return self.norm(y).transpose(1, 2)


class CausalTCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dilations: tuple[int, ...] = (1, 2, 4, 8), dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(*(TemporalBlock(hidden_dim, dilation, dropout) for dilation in dilations))

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, time, features]
        y = self.blocks(self.projection(x.transpose(1, 2)))
        return y[:, :, -1]


class DirectionalThreatModel(nn.Module):
    """Static MLP + causal TCN + gated fusion.

    The model consumes current-flow features and a causal history window. It
    has no pathway for reverse-flow or future records.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_encoder = CausalTCN(input_dim, hidden_dim, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.binary_head = nn.Linear(hidden_dim, 1)
        self.multiclass_head = nn.Linear(hidden_dim, classes)
        self.severity_head = nn.Linear(hidden_dim, 1)

    def forward(self, current: Tensor, history: Tensor) -> dict[str, Tensor]:
        static = self.static_encoder(current)
        temporal = self.temporal_encoder(history)
        joined = torch.cat([static, temporal], dim=-1)
        gate = self.gate(joined)
        fused = self.fusion(torch.cat([static, temporal * gate], dim=-1))
        return {
            "embedding": fused,
            "binary_logit": self.binary_head(fused).squeeze(-1),
            "multiclass_logits": self.multiclass_head(fused),
            "severity": self.severity_head(fused).squeeze(-1),
            "gate": gate,
        }
