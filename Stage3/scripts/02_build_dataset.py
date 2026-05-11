"""Build processed dataset and windows for training.

Steps:
    1. Load raw UNSW-NB15 train/test CSVs.
    2. Concatenate them and re-split (or use official partition with carved val).
    3. Fit Preprocessor on train; transform val and test.
    4. Build sliding windows from each split.
    5. Persist preprocessor and ``.npz`` files for downstream training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from diploma_nids.data import (
    PreprocessConfig,
    Preprocessor,
    UNSWConfig,
    build_windows,
    load_unsw_nb15,
    split_official,
    split_random,
    split_temporal,
)
from diploma_nids.utils import get_logger, load_yaml, set_seed, setup_logging

logger = get_logger(__name__)


def _save_split(out_dir: Path, name: str, X: np.ndarray, y: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{name}.npz", X=X, y=y)
    logger.info("%s: X=%s y=%s -> %s", name, X.shape, y.shape, out_dir / f"{name}.npz")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="data config YAML, e.g. configs/data/unsw_nb15.yaml")
    parser.add_argument("--preprocess", required=True, help="preprocess config YAML")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default=None, help="override splits.strategy from preprocess YAML")
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)

    data_cfg = UNSWConfig(**load_yaml(args.config))
    prep_cfg = PreprocessConfig(**load_yaml(args.preprocess))
    if args.strategy:
        prep_cfg.splits.strategy = args.strategy

    train_full, test_full = load_unsw_nb15(data_cfg)

    if prep_cfg.splits.strategy == "official":
        train_df, val_df, test_df = split_official(train_full, test_full, prep_cfg.splits.val_ratio)
    else:
        full = pd.concat([train_full, test_full], axis=0, ignore_index=True)
        if prep_cfg.splits.strategy == "temporal":
            train_df, val_df, test_df = split_temporal(full, prep_cfg.splits)
        else:
            train_df, val_df, test_df = split_random(full, prep_cfg.splits, seed=args.seed)

    prep = Preprocessor(prep_cfg, data_cfg.schema)
    train_pp = prep.fit_transform(train_df)
    val_pp = prep.transform(val_df)
    test_pp = prep.transform(test_df)

    out_dir = Path(data_cfg.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prep.save(out_dir / "preprocessor.json")

    label_col = data_cfg.schema.label_binary
    feat_cols = prep.state.output_columns

    # UNSW-NB15 cleaned partition is grouped by attack_cat rather than ordered
    # by time, so window aggregation degenerates to homogeneous-class blocks.
    # Shuffling within each split (with a fixed seed) before windowing produces
    # mixed-label windows representative of the joint attack/normal distribution.
    rng = np.random.default_rng(args.seed)

    splits = {"train": train_pp, "val": val_pp, "test": test_pp}
    for name, df_pp in splits.items():
        X = df_pp[feat_cols].to_numpy(dtype=np.float32)
        y = df_pp[label_col].to_numpy(dtype=np.int64)
        _save_split(out_dir / "tabular", name, X, y)

        if prep_cfg.windowing.enabled:
            perm = rng.permutation(len(X))
            Xw, yw = build_windows(X[perm], y[perm], prep_cfg.windowing)
            _save_split(out_dir / "windows", name, Xw, yw)
            pos_ratio = float(yw.mean()) if len(yw) else 0.0
            logger.info("%s windows positive_ratio=%.4f", name, pos_ratio)

    logger.info("dataset build complete -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
