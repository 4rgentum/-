"""Unit tests for split utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from diploma_nids.data import SplitsConfig, split_official, split_random, split_temporal


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(1000), "label": np.arange(1000) % 2})


def test_temporal_preserves_order(df):
    cfg = SplitsConfig(strategy="temporal", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train, val, test = split_temporal(df, cfg)
    assert len(train) + len(val) + len(test) == len(df)
    assert train["x"].max() < val["x"].min()
    assert val["x"].max() < test["x"].min()


def test_random_seed_reproducible(df):
    cfg = SplitsConfig(strategy="random", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    a1, a2, a3 = split_random(df, cfg, seed=42)
    b1, b2, b3 = split_random(df, cfg, seed=42)
    pd.testing.assert_frame_equal(a1, b1)
    pd.testing.assert_frame_equal(a2, b2)
    pd.testing.assert_frame_equal(a3, b3)


def test_random_disjoint(df):
    cfg = SplitsConfig(strategy="random", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train, val, test = split_random(df, cfg, seed=7)
    full = set(df["x"])
    parts = set(train["x"]) | set(val["x"]) | set(test["x"])
    assert parts == full
    assert len(set(train["x"]) & set(val["x"])) == 0
    assert len(set(val["x"]) & set(test["x"])) == 0


def test_official_split_uses_full_test():
    train_full = pd.DataFrame({"x": np.arange(800), "label": np.arange(800) % 2})
    test_full = pd.DataFrame({"x": np.arange(200), "label": np.arange(200) % 2})
    train, val, test = split_official(train_full, test_full, val_ratio=0.2)
    assert len(train) + len(val) == len(train_full)
    assert len(test) == len(test_full)
    assert train["x"].max() < val["x"].min()


def test_split_ratios_validation():
    with pytest.raises(Exception):
        SplitsConfig(strategy="temporal", train_ratio=0.6, val_ratio=0.3, test_ratio=0.3)
