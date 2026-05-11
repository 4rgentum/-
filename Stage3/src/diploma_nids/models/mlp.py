"""MLP baseline: tabular flow-vector classifier (no windowing)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import DLDetector, ModelOutput, register


@register("mlp")
class MLP(DLDetector):
    expects_windows = False

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        arch = config["architecture"]
        layers: list[nn.Module] = []
        prev = int(arch["input_features"])
        for dim in arch["hidden_dims"]:
            layers += [nn.Linear(prev, int(dim)), nn.ReLU(), nn.Dropout(float(arch.get("dropout", 0.0)))]
            prev = int(dim)
        self.body = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> ModelOutput:  # (B, F)
        if x.ndim == 3:                       # accept windows by mean-pooling
            x = x.mean(dim=1)
        h = self.body(x)
        logit = self.classifier(h).squeeze(-1)
        return ModelOutput(logits=logit, probs=torch.sigmoid(logit))
