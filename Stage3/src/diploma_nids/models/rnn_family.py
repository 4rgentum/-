"""LSTM, GRU and BiLSTM (with optional self-attention) detectors."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .base import DLDetector, ModelOutput, register


def _make_head(in_dim: int, hidden_dims: list[int], dropout: float) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    prev = in_dim
    for d in hidden_dims:
        layers += [nn.Linear(prev, int(d)), nn.ReLU(), nn.Dropout(dropout)]
        prev = int(d)
    return nn.Sequential(*layers), prev


@register("lstm")
class LSTMDetector(DLDetector):
    expects_windows = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        arch = config["architecture"]
        lstm = arch["lstm"]
        hidden = int(lstm["hidden_size"])
        nlayers = int(lstm.get("num_layers", 1))
        bi = bool(lstm.get("bidirectional", False))
        self.lstm = nn.LSTM(
            int(arch["input_features"]),
            hidden,
            num_layers=nlayers,
            dropout=float(lstm.get("dropout", 0.0)) if nlayers > 1 else 0.0,
            bidirectional=bi,
            batch_first=True,
        )
        out_dim = hidden * (2 if bi else 1)
        head = arch["head"]
        self.head, prev = _make_head(out_dim, head["hidden_dims"], float(head.get("dropout", 0.0)))
        self.classifier = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        logit = self.classifier(self.head(h)).squeeze(-1)
        return ModelOutput(logits=logit, probs=torch.sigmoid(logit))


@register("gru")
class GRUDetector(DLDetector):
    expects_windows = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        arch = config["architecture"]
        gru = arch["gru"]
        hidden = int(gru["hidden_size"])
        nlayers = int(gru.get("num_layers", 1))
        bi = bool(gru.get("bidirectional", False))
        self.gru = nn.GRU(
            int(arch["input_features"]),
            hidden,
            num_layers=nlayers,
            dropout=float(gru.get("dropout", 0.0)) if nlayers > 1 else 0.0,
            bidirectional=bi,
            batch_first=True,
        )
        out_dim = hidden * (2 if bi else 1)
        head = arch["head"]
        self.head, prev = _make_head(out_dim, head["hidden_dims"], float(head.get("dropout", 0.0)))
        self.classifier = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        out, _ = self.gru(x)
        h = out[:, -1, :]
        logit = self.classifier(self.head(h)).squeeze(-1)
        return ModelOutput(logits=logit, probs=torch.sigmoid(logit))


@register("bilstm")
class BiLSTMAttn(DLDetector):
    """BiLSTM with optional dot-product self-attention pooling."""

    expects_windows = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        arch = config["architecture"]
        lstm = arch["lstm"]
        hidden = int(lstm["hidden_size"])
        nlayers = int(lstm.get("num_layers", 1))
        self.lstm = nn.LSTM(
            int(arch["input_features"]),
            hidden,
            num_layers=nlayers,
            dropout=float(lstm.get("dropout", 0.0)) if nlayers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        out_dim = hidden * 2
        attn_cfg = arch.get("attention", {"enabled": False})
        self.use_attn = bool(attn_cfg.get("enabled", False))
        if self.use_attn:
            self.attn_q = nn.Linear(out_dim, out_dim, bias=False)
            self.attn_k = nn.Linear(out_dim, out_dim, bias=False)
            self.scale = 1.0 / math.sqrt(out_dim)
        head = arch["head"]
        self.head, prev = _make_head(out_dim, head["hidden_dims"], float(head.get("dropout", 0.0)))
        self.classifier = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        out, _ = self.lstm(x)                  # (B, T, 2H)
        if self.use_attn:
            q = self.attn_q(out[:, -1:, :])    # query = last step
            k = self.attn_k(out)
            attn = torch.softmax((q @ k.transpose(1, 2)) * self.scale, dim=-1)
            ctx = (attn @ out).squeeze(1)
        else:
            ctx = out[:, -1, :]
        logit = self.classifier(self.head(ctx)).squeeze(-1)
        return ModelOutput(logits=logit, probs=torch.sigmoid(logit))
