"""Sliding-window construction over preprocessed flow records."""
from __future__ import annotations

import numpy as np

from .schema import WindowingConfig


def build_windows(
    features: np.ndarray,
    labels: np.ndarray,
    cfg: WindowingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(windows, window_labels)``.

    Args:
        features: ``(n, d)`` float array of preprocessed flow records.
        labels:   ``(n,)``  binary array {0, 1}.
        cfg:      windowing configuration.

    Returns:
        windows:        ``(m, W, d)`` float32 array.
        window_labels:  ``(m,)``     int64 array.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D (n, d); got shape {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D; got shape {labels.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must align in length")

    n, d = features.shape
    W, S = cfg.window_size, cfg.stride
    if n < W:
        return np.empty((0, W, d), dtype=np.float32), np.empty((0,), dtype=np.int64)

    starts = np.arange(0, n - W + 1, S)
    m = len(starts)

    windows = np.empty((m, W, d), dtype=np.float32)
    win_labels = np.empty((m,), dtype=np.int64)

    for i, s in enumerate(starts):
        e = s + W
        windows[i] = features[s:e]
        seg = labels[s:e]
        if cfg.label_aggregation == "any":
            win_labels[i] = int(seg.any())
        elif cfg.label_aggregation == "majority":
            win_labels[i] = int(seg.mean() >= cfg.majority_threshold)
        elif cfg.label_aggregation == "last":
            win_labels[i] = int(seg[-1])
        else:
            raise ValueError(f"unknown label_aggregation: {cfg.label_aggregation}")
    return windows, win_labels


def n_windows(n_records: int, cfg: WindowingConfig) -> int:
    if n_records < cfg.window_size:
        return 0
    return (n_records - cfg.window_size) // cfg.stride + 1
