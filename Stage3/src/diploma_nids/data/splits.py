"""Train/val/test splits for UNSW-NB15.

Three strategies:
    - ``official``: respect the partition shipped with UNSW-NB15;
                    val is carved from the tail of the official train split.
    - ``temporal``: preserve order; first train_ratio rows -> train, etc.
    - ``random``  : uniform random shuffle with fixed seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .schema import SplitsConfig

logger = get_logger(__name__)


def split_official(
    train_full: pd.DataFrame,
    test_full: pd.DataFrame,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """UNSW-NB15 official partition with a carved validation set.

    The official training CSV is grouped by ``attack_cat`` (attacks first,
    Normal at the tail), so a naive tail-cut produces a single-class val.
    The training set is shuffled with ``seed`` before the val-cut to obtain
    a class-balanced validation split. The official test partition is
    returned unchanged.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_full))
    shuffled = train_full.iloc[perm].reset_index(drop=True)
    n = len(shuffled)
    n_val = int(round(n * val_ratio))
    train_df = shuffled.iloc[: n - n_val].reset_index(drop=True)
    val_df = shuffled.iloc[n - n_val :].reset_index(drop=True)
    test_df = test_full.reset_index(drop=True)
    logger.info(
        "official split (with shuffled train): train=%d val=%d test=%d (val pos_ratio=%.3f)",
        len(train_df), len(val_df), len(test_df),
        float(val_df["label"].mean()) if "label" in val_df.columns else float("nan"),
    )
    return train_df, val_df, test_df


def split_temporal(
    df: pd.DataFrame,
    cfg: SplitsConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Order-preserving split. The dataframe is assumed to already be in
    insertion order; UNSW-NB15 cleaned partition is.
    """
    n = len(df)
    n_train = int(round(n * cfg.train_ratio))
    n_val = int(round(n * cfg.val_ratio))
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)
    logger.info("temporal split: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def split_random(
    df: pd.DataFrame,
    cfg: SplitsConfig,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_train = int(round(len(df) * cfg.train_ratio))
    n_val = int(round(len(df) * cfg.val_ratio))
    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]
    train_df = df.iloc[idx_train].reset_index(drop=True)
    val_df = df.iloc[idx_val].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)
    logger.info("random split: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df
