"""1D-CNN over flow-windows."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import DLDetector, ModelOutput, register


@register("cnn1d")
class CNN1D(DLDetector):
    expects_windows = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        arch = config["architecture"]
        in_ch = int(arch["input_features"])
        blocks: list[nn.Module] = []
        for blk in arch["conv_blocks"]:
            out_ch = int(blk["out_channels"])
            ks = int(blk["kernel_size"])
            pool = int(blk.get("pool_size", 1))
            drop = float(blk.get("dropout", 0.0))
            blocks += [
                nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(pool) if pool > 1 else nn.Identity(),
                nn.Dropout(drop) if drop > 0 else nn.Identity(),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1) if arch.get("global_pool", "mean") == "mean" else nn.AdaptiveMaxPool1d(1)

        head = arch["head"]
        layers: list[nn.Module] = []
        prev = in_ch
        for d in head["hidden_dims"]:
            layers += [nn.Linear(prev, int(d)), nn.ReLU(), nn.Dropout(float(head.get("dropout", 0.0)))]
            prev = int(d)
        self.head = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> ModelOutput:  # (B, W, F)
        z = self.conv(x.transpose(1, 2))      # (B, C, T)
        z = self.global_pool(z).squeeze(-1)    # (B, C)
        h = self.head(z)
        logit = self.classifier(h).squeeze(-1)
        return ModelOutput(logits=logit, probs=torch.sigmoid(logit))
