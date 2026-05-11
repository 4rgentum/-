"""CNN-LSTM: proposed architecture.

Stack:  Conv1D x N  ->  LSTM  ->  MLP head  ->  sigmoid logit

Math: see Stage1 §1.4.4 and Stage2 §2.1.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import DLDetector, ModelOutput, register


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "leakyrelu": nn.LeakyReLU(0.1)}[name.lower()]


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        pool_size: int,
        dropout: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = _act(activation)
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        return self.drop(self.pool(self.act(self.bn(self.conv(x)))))


@register("cnn_lstm")
class CNNLSTM(DLDetector):
    """Hybrid CNN-LSTM.

    Input:  (B, W, F) — window of W flow records, F features each.
    Output: ModelOutput with ``logits`` shape (B,) and ``probs`` shape (B,).
    """

    expects_windows = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        arch = config["architecture"]
        in_features = int(arch["input_features"])

        # Conv stack
        conv_blocks: list[nn.Module] = []
        in_ch = in_features
        for blk in arch["conv_blocks"]:
            conv_blocks.append(
                _ConvBlock(
                    in_ch=in_ch,
                    out_ch=int(blk["out_channels"]),
                    kernel_size=int(blk["kernel_size"]),
                    pool_size=int(blk.get("pool_size", 1)),
                    dropout=float(blk.get("dropout", 0.0)),
                    activation=str(blk.get("activation", "relu")),
                )
            )
            in_ch = int(blk["out_channels"])
        self.conv = nn.Sequential(*conv_blocks)

        # LSTM
        lstm_cfg = arch["lstm"]
        bidirectional = bool(lstm_cfg.get("bidirectional", False))
        hidden = int(lstm_cfg["hidden_size"])
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden,
            num_layers=int(lstm_cfg.get("num_layers", 1)),
            dropout=float(lstm_cfg.get("dropout", 0.0)) if int(lstm_cfg.get("num_layers", 1)) > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_out_dim = hidden * (2 if bidirectional else 1)

        # MLP head
        head_cfg = arch["head"]
        head_layers: list[nn.Module] = []
        prev = lstm_out_dim
        for dim in head_cfg["hidden_dims"]:
            head_layers.append(nn.Linear(prev, int(dim)))
            head_layers.append(_act(str(head_cfg.get("activation", "relu"))))
            head_layers.append(nn.Dropout(float(head_cfg.get("dropout", 0.0))))
            prev = int(dim)
        self.head = nn.Sequential(*head_layers)
        self.classifier = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> ModelOutput:  # (B, W, F)
        # Conv1D expects (B, C=F, T=W)
        z = self.conv(x.transpose(1, 2))      # (B, C', T')
        z = z.transpose(1, 2)                  # (B, T', C')
        out, _ = self.lstm(z)                  # (B, T', H)
        h = out[:, -1, :]                      # last time step
        h = self.head(h)
        logit = self.classifier(h).squeeze(-1) # (B,)
        prob = torch.sigmoid(logit)
        return ModelOutput(logits=logit, probs=prob)
