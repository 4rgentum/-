"""Loaders for UNSW-NB15 and CICIDS2017 raw CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.logging import get_logger
from .schema import UNSWConfig

logger = get_logger(__name__)


def load_unsw_nb15(cfg: UNSWConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load official UNSW-NB15 train/test partition.

    The partition is the cleaned distribution
    (UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv) released by
    UNSW Canberra Cyber: https://research.unsw.edu.au/projects/unsw-nb15-dataset
    """
    train_path = Path(cfg.paths.train_csv)
    test_path = Path(cfg.paths.test_csv)
    if not train_path.is_file():
        raise FileNotFoundError(f"UNSW-NB15 training CSV not found: {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"UNSW-NB15 testing CSV not found: {test_path}")

    logger.info("loading UNSW-NB15 train: %s", train_path)
    train_df = pd.read_csv(train_path, low_memory=False)
    logger.info("loading UNSW-NB15 test:  %s", test_path)
    test_df = pd.read_csv(test_path, low_memory=False)

    # The cleaned partition adds an 'id' column that is just an index.
    for df in (train_df, test_df):
        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)

    expected = set(cfg.schema.all_features) | {cfg.schema.label_binary, cfg.schema.label_multiclass}
    missing = expected - set(train_df.columns)
    if missing:
        raise ValueError(f"UNSW-NB15 train CSV is missing columns: {sorted(missing)}")

    return train_df, test_df


def load_cicids2017(csv_glob: str | Path, label_column: str = "Label") -> pd.DataFrame:
    """Load CICIDS2017 by concatenating CSVs under ``csv_glob``.

    Used for cross-evaluation only; no schema validation is enforced here.
    """
    paths = sorted(Path().glob(str(csv_glob)) if isinstance(csv_glob, str) else Path(csv_glob).parent.glob(Path(csv_glob).name))
    if not paths:
        raise FileNotFoundError(f"no CICIDS2017 CSVs match: {csv_glob}")
    logger.info("loading CICIDS2017: %d files", len(paths))
    frames = [pd.read_csv(p, low_memory=False, encoding="latin-1") for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    if label_column not in df.columns:
        raise ValueError(f"CICIDS2017: label column '{label_column}' not found")
    return df
