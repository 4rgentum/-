"""Temporal Convolutional Network (Bai et al., 2018) for flow-windows."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils import weight_norm

from .base import DLDetector, ModelOutput, register


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : -self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, k: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(n_in, n_out, k, padding=pad, dilation=dilation)),
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(n_out, n_out, k, padding=pad, dilation=dilation)),
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.proj(x))


@register("tcn")
class TCN(DLDetector):
    expects_windows = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        arch = config["architecture"]
        ch = [int(c) for c in arch["channels"]]
        k = int(arch["kernel_size"])
        drop = float(arch.get("dropout", 0.2))

        layers: list[nn.Module] = []
        in_ch = int(arch["input_features"])
        for i, out_ch in enumerate(ch):
            layers.append(_TemporalBlock(in_ch, out_ch, k, dilation=2**i, dropout=drop))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)

        head = arch["head"]
        layers_h: list[nn.Module] = []
        prev = in_ch
        for d in head["hidden_dims"]:
            layers_h += [nn.Linear(prev, int(d)), nn.ReLU(), nn.Dropout(float(head.get("dropout", 0.0)))]
            prev = int(d)
        self.head = nn.Sequential(*layers_h)
        self.classifier = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        z = self.tcn(x.transpose(1, 2))        # (B, C, T)
        h = z[..., -1]                          # last time step
        logit = self.classifier(self.head(h)).squeeze(-1)
        return ModelOutput(logits=logit, probs=torch.sigmoid(logit))
