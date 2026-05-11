"""Threshold selection and probability calibration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class TemperatureScaler:
    """Single-parameter temperature scaling (Guo et al., 2017)."""

    temperature: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray, lr: float = 0.01, max_iter: int = 200) -> TemperatureScaler:
        z = torch.from_numpy(logits.astype(np.float32))
        y = torch.from_numpy(y_true.astype(np.float32))
        T = nn.Parameter(torch.tensor(1.0))
        opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)
        loss_fn = nn.BCEWithLogitsLoss()

        def _closure():
            opt.zero_grad()
            scaled = z / T.clamp_min(1e-3)
            loss = loss_fn(scaled, y)
            loss.backward()
            return loss

        opt.step(_closure)
        self.temperature = float(T.detach().clamp_min(1e-3).item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))


def calibrate(logits: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, TemperatureScaler]:
    scaler = TemperatureScaler().fit(logits, y_true)
    return scaler.transform(logits), scaler
