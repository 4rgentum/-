"""Transformer-encoder for flow-windows."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .base import DLDetector, ModelOutput, register


class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


@register("transformer")
class TransformerEncoder(DLDetector):
    expects_windows = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        arch = config["architecture"]
        d_model = int(arch["embedding_dim"])
        self.input_proj = nn.Linear(int(arch["input_features"]), d_model)
        self.pe = _SinusoidalPE(d_model, max_len=int(arch["window_size"]) + 32)

        enc = arch["encoder"]
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(enc["num_heads"]),
            dim_feedforward=int(enc["ff_dim"]),
            dropout=float(enc.get("dropout", 0.1)),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(enc["num_layers"]))
        self.pool = arch.get("pooling", "mean")

        head = arch["head"]
        layers: list[nn.Module] = []
        prev = d_model
        for d in head["hidden_dims"]:
            layers += [nn.Linear(prev, int(d)), nn.ReLU(), nn.Dropout(float(head.get("dropout", 0.0)))]
            prev = int(d)
        self.head = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        z = self.input_proj(x)
        z = self.pe(z)
        z = self.encoder(z)
        if self.pool == "mean":
            h = z.mean(dim=1)
        elif self.pool == "last":
            h = z[:, -1, :]
        else:
            h = z[:, 0, :]
        logit = self.classifier(self.head(h)).squeeze(-1)
        return ModelOutput(logits=logit, probs=torch.sigmoid(logit))
