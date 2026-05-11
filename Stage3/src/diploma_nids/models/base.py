"""Base classes and registry for models.

Two top-level abstractions:
    * BaseDetector  — sklearn-style fit/predict/predict_proba/score.
    * Torch-based DL models inherit from torch.nn.Module and a thin
      DLDetector mixin that adds score/load/save and a uniform interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class ModelOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)


class BaseDetector(ABC):
    """Sklearn-style detector interface for classical baselines."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseDetector: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return shape ``(n, 2)`` with class probabilities."""

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score for the positive class."""
        return self.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.score(X) >= threshold).astype(np.int64)


class DLDetector(nn.Module):
    """Mixin for torch-based detectors. Subclasses implement ``forward``
    returning a ``ModelOutput``. ``score`` and persistence are uniform."""

    name: str = "dl"
    expects_windows: bool = True

    def score(self, X: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            out = self(X)
        return out.probs[..., -1] if out.probs.ndim > 1 else out.probs

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "name": self.name}, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["state_dict"])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, type] = {}


def register(name: str):
    def _decorator(cls):
        if name in _REGISTRY:
            raise KeyError(f"model already registered: {name}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_model(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model: {name}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(name: str, config: dict[str, Any]) -> Any:
    cls = get_model(name)
    return cls(config)
