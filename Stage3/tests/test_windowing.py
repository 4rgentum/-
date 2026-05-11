"""Unit tests for sliding-window construction and label aggregation."""
from __future__ import annotations

import numpy as np
import pytest

from diploma_nids.data import WindowingConfig, build_windows, n_windows


def test_window_shapes():
    n, d, W, S = 100, 8, 32, 8
    X = np.random.default_rng(0).normal(size=(n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    cfg = WindowingConfig(window_size=W, stride=S, label_aggregation="any")
    Xw, yw = build_windows(X, y, cfg)
    assert Xw.shape == (n_windows(n, cfg), W, d)
    assert yw.shape == (n_windows(n, cfg),)
    assert Xw.dtype == np.float32
    assert yw.dtype == np.int64


def test_label_aggregation_any():
    cfg = WindowingConfig(window_size=4, stride=1, label_aggregation="any")
    X = np.zeros((6, 1), dtype=np.float32)
    y = np.array([0, 0, 1, 0, 0, 0], dtype=np.int64)
    _, yw = build_windows(X, y, cfg)
    # Windows: [0..3]=any(0,0,1,0)=1; [1..4]=1; [2..5]=1; [3..6 not exist]
    # Actually: starts = 0,1,2 -> 3 windows
    assert yw.tolist() == [1, 1, 1]


def test_label_aggregation_majority():
    cfg = WindowingConfig(window_size=4, stride=1, label_aggregation="majority", majority_threshold=0.5)
    X = np.zeros((5, 1), dtype=np.float32)
    y = np.array([1, 1, 0, 0, 1], dtype=np.int64)
    _, yw = build_windows(X, y, cfg)
    # Windows: [1,1,0,0]->mean=0.5->1; [1,0,0,1]->mean=0.5->1
    assert yw.tolist() == [1, 1]


def test_label_aggregation_last():
    cfg = WindowingConfig(window_size=3, stride=1, label_aggregation="last")
    X = np.zeros((5, 1), dtype=np.float32)
    y = np.array([0, 0, 0, 1, 0], dtype=np.int64)
    _, yw = build_windows(X, y, cfg)
    assert yw.tolist() == [0, 1, 0]


def test_too_short_returns_empty():
    cfg = WindowingConfig(window_size=32, stride=8)
    X = np.zeros((10, 4), dtype=np.float32)
    y = np.zeros(10, dtype=np.int64)
    Xw, yw = build_windows(X, y, cfg)
    assert Xw.shape == (0, 32, 4)
    assert yw.shape == (0,)


def test_input_validation():
    cfg = WindowingConfig(window_size=4, stride=2)
    with pytest.raises(ValueError):
        build_windows(np.zeros((5,), dtype=np.float32), np.zeros(5, dtype=np.int64), cfg)
    with pytest.raises(ValueError):
        build_windows(np.zeros((5, 2), dtype=np.float32), np.zeros((5, 1), dtype=np.int64), cfg)
    with pytest.raises(ValueError):
        build_windows(np.zeros((5, 2), dtype=np.float32), np.zeros(4, dtype=np.int64), cfg)


def test_n_windows_formula():
    cfg = WindowingConfig(window_size=32, stride=8)
    assert n_windows(100, cfg) == (100 - 32) // 8 + 1
    assert n_windows(31, cfg) == 0
    assert n_windows(32, cfg) == 1
