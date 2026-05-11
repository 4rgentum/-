"""Classical ML baselines: LR, RF, XGBoost, IsolationForest, OneClassSVM."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM

from .base import BaseDetector, register


def _flatten_windows(X: np.ndarray) -> np.ndarray:
    """Window-shaped (B, W, F) -> (B, W*F). Tabular passes through."""
    if X.ndim == 3:
        return X.reshape(X.shape[0], -1)
    return X


@register("logistic_regression")
class LogisticRegressionDetector(BaseDetector):
    def __init__(self, config: dict[str, Any]) -> None:
        params = config.get("logistic_regression", config)
        self.model = LogisticRegression(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionDetector:
        self.model.fit(_flatten_windows(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_flatten_windows(X))


@register("random_forest")
class RandomForestDetector(BaseDetector):
    def __init__(self, config: dict[str, Any]) -> None:
        params = config.get("random_forest", config)
        self.model = RandomForestClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestDetector:
        self.model.fit(_flatten_windows(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_flatten_windows(X))


@register("xgboost")
class XGBoostDetector(BaseDetector):
    def __init__(self, config: dict[str, Any]) -> None:
        from xgboost import XGBClassifier

        params = dict(config.get("xgboost", config))
        if params.get("scale_pos_weight") == "auto":
            params.pop("scale_pos_weight")
            self._auto_scale = True
        else:
            self._auto_scale = False
        params.setdefault("eval_metric", "logloss")
        self.model = XGBClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostDetector:
        Xf = _flatten_windows(X)
        if self._auto_scale:
            n_pos = max(int(np.sum(y == 1)), 1)
            n_neg = max(int(np.sum(y == 0)), 1)
            self.model.set_params(scale_pos_weight=n_neg / n_pos)
        self.model.fit(Xf, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_flatten_windows(X))


@register("isolation_forest")
class IsolationForestDetector(BaseDetector):
    def __init__(self, config: dict[str, Any]) -> None:
        params = config.get("isolation_forest", config)
        self.model = IsolationForest(**params)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> IsolationForestDetector:
        # Fit on normal samples only when labels are provided
        Xf = _flatten_windows(X)
        if y is not None:
            Xf = Xf[y == 0]
        self.model.fit(Xf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # decision_function: high = normal, low = anomaly
        s = -self.model.decision_function(_flatten_windows(X))
        # Min-max normalize to [0, 1] for interpretability as a probability
        s_min, s_max = float(s.min()), float(s.max())
        s_norm = (s - s_min) / (s_max - s_min + 1e-12)
        return np.column_stack([1 - s_norm, s_norm])


@register("ocsvm")
class OneClassSVMDetector(BaseDetector):
    def __init__(self, config: dict[str, Any]) -> None:
        params = config.get("ocsvm", config)
        self.model = OneClassSVM(**params)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> OneClassSVMDetector:
        Xf = _flatten_windows(X)
        if y is not None:
            Xf = Xf[y == 0]
        self.model.fit(Xf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        s = -self.model.decision_function(_flatten_windows(X))
        s_min, s_max = float(s.min()), float(s.max())
        s_norm = (s - s_min) / (s_max - s_min + 1e-12)
        return np.column_stack([1 - s_norm, s_norm])


def save_classical(model: BaseDetector, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_classical(path: str | Path) -> BaseDetector:
    return joblib.load(path)
