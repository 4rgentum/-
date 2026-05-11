"""Preprocessor: fit on train, apply uniformly to val/test/inference.

All cleaning, encoding and scaling statistics are estimated on the training
split only and frozen at fit time. This rules out test-time leakage and
keeps inference-time transformations identical to those used at training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .schema import PreprocessConfig, UNSWFeatureSchema

logger = get_logger(__name__)


@dataclass
class FittedState:
    """Frozen statistics computed on the training split."""

    numeric_features: list[str] = field(default_factory=list)
    log_features: list[str] = field(default_factory=list)
    onehot_categories: dict[str, list[str]] = field(default_factory=dict)
    freq_encoding: dict[str, dict[str, float]] = field(default_factory=dict)
    target_encoding: dict[str, dict[str, float]] = field(default_factory=dict)
    target_global_mean: float = 0.0
    scaler_center: dict[str, float] = field(default_factory=dict)
    scaler_scale: dict[str, float] = field(default_factory=dict)
    clip_low: dict[str, float] = field(default_factory=dict)
    clip_high: dict[str, float] = field(default_factory=dict)
    binary_flags: list[str] = field(default_factory=list)
    output_columns: list[str] = field(default_factory=list)
    label_binary: str = "label"

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_features": self.numeric_features,
            "log_features": self.log_features,
            "onehot_categories": self.onehot_categories,
            "freq_encoding": self.freq_encoding,
            "target_encoding": self.target_encoding,
            "target_global_mean": self.target_global_mean,
            "scaler_center": self.scaler_center,
            "scaler_scale": self.scaler_scale,
            "clip_low": self.clip_low,
            "clip_high": self.clip_high,
            "binary_flags": self.binary_flags,
            "output_columns": self.output_columns,
            "label_binary": self.label_binary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FittedState:
        return cls(**data)


class Preprocessor:
    """Sklearn-style preprocessor. Use ``fit_transform`` on train,
    then ``transform`` on val/test/inference."""

    def __init__(self, config: PreprocessConfig, schema: UNSWFeatureSchema) -> None:
        self.cfg = config
        self.schema = schema
        self.state: FittedState | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> Preprocessor:
        self._fit_impl(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.state is None:
            raise RuntimeError("Preprocessor.transform called before fit")
        return self._transform_impl(df)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    @property
    def output_dim(self) -> int:
        if self.state is None:
            raise RuntimeError("output_dim queried before fit")
        return len(self.state.output_columns)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self.state is None:
            raise RuntimeError("cannot save unfitted preprocessor")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.cfg.model_dump(),
            "schema": self.schema.model_dump(),
            "state": self.state.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        logger.info("preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> Preprocessor:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        cfg = PreprocessConfig(**payload["config"])
        schema = UNSWFeatureSchema(**payload["schema"])
        prep = cls(cfg, schema)
        prep.state = FittedState.from_dict(payload["state"])
        return prep

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def _fit_impl(self, df: pd.DataFrame) -> None:
        df = self._cleanup(df.copy(), training=True)
        state = FittedState(label_binary=self.schema.label_binary)

        # Numeric: clip, log-transform, robust/standard scale
        state.numeric_features = list(self.schema.numeric)
        state.log_features = [f for f in self.cfg.scaling.log_transform_features if f in self.schema.numeric]

        for feat in state.numeric_features:
            x = df[feat].astype(float)
            state.clip_low[feat] = float(x.quantile(self.cfg.cleanup.outlier_clip_quantile_low))
            state.clip_high[feat] = float(x.quantile(self.cfg.cleanup.outlier_clip_quantile_high))

        # Apply clip + log to fit scaler stats
        df_num = df[state.numeric_features].astype(float).copy()
        df_num = self._apply_clip(df_num, state)
        df_num = self._apply_log(df_num, state.log_features)

        for feat in state.numeric_features:
            x = df_num[feat]
            if self.cfg.scaling.scaler == "robust":
                center = float(x.median())
                iqr = float(x.quantile(0.75) - x.quantile(0.25))
                scale = iqr if iqr > 1e-12 else 1.0
            elif self.cfg.scaling.scaler == "standard":
                center = float(x.mean())
                std = float(x.std(ddof=0))
                scale = std if std > 1e-12 else 1.0
            elif self.cfg.scaling.scaler == "minmax":
                center = float(x.min())
                rng = float(x.max() - x.min())
                scale = rng if rng > 1e-12 else 1.0
            else:
                raise ValueError(f"unknown scaler: {self.cfg.scaling.scaler}")
            state.scaler_center[feat] = center if self.cfg.scaling.with_centering else 0.0
            state.scaler_scale[feat] = scale if self.cfg.scaling.with_scaling else 1.0

        # Categorical
        threshold = self.cfg.encoding.categorical_low_card_threshold
        for cat in self.schema.categorical:
            categories = sorted(df[cat].astype(str).str.lower().str.strip().unique().tolist())
            if len(categories) <= threshold:
                state.onehot_categories[cat] = categories
            else:
                if self.cfg.encoding.high_card_strategy == "frequency":
                    counts = df[cat].astype(str).str.lower().str.strip().value_counts(normalize=True)
                    state.freq_encoding[cat] = counts.to_dict()
                elif self.cfg.encoding.high_card_strategy == "target":
                    y = df[self.schema.label_binary].astype(int)
                    state.target_global_mean = float(y.mean())
                    grp = df.assign(__tgt=y).groupby(df[cat].astype(str).str.lower().str.strip())
                    means = grp["__tgt"].mean()
                    counts = grp["__tgt"].size()
                    alpha = self.cfg.encoding.target_smoothing_alpha
                    smooth = (means * counts + alpha * state.target_global_mean) / (counts + alpha)
                    state.target_encoding[cat] = smooth.to_dict()
                else:
                    # embedding: nothing to fit here; encoder layer is part of the model
                    state.freq_encoding[cat] = {}

        state.binary_flags = list(self.schema.binary_flags)

        # Output column order
        cols: list[str] = []
        cols.extend(state.numeric_features)
        for cat, cats in state.onehot_categories.items():
            cols.extend([f"{cat}__oh__{c}" for c in cats])
        for cat in state.freq_encoding:
            cols.append(f"{cat}__freq")
        for cat in state.target_encoding:
            cols.append(f"{cat}__tgt")
        cols.extend(state.binary_flags)
        cols.extend(self.schema.context_counts)
        state.output_columns = cols

        self.state = state
        logger.info(
            "preprocessor fitted: %d numeric + %d cat (oh=%d, freq=%d, tgt=%d) + %d flags + %d ctx -> %d cols",
            len(state.numeric_features),
            len(self.schema.categorical),
            len(state.onehot_categories),
            len(state.freq_encoding),
            len(state.target_encoding),
            len(state.binary_flags),
            len(self.schema.context_counts),
            len(state.output_columns),
        )

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def _transform_impl(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.state is not None
        df = self._cleanup(df.copy(), training=False)

        # Numeric
        num = df[self.state.numeric_features].astype(float).copy()
        num = self._apply_clip(num, self.state)
        num = self._apply_log(num, self.state.log_features)
        for feat in self.state.numeric_features:
            num[feat] = (num[feat] - self.state.scaler_center[feat]) / self.state.scaler_scale[feat]

        # Categorical
        cat_frames: list[pd.DataFrame] = []
        for cat, cats in self.state.onehot_categories.items():
            col = df[cat].astype(str).str.lower().str.strip()
            for c in cats:
                cat_frames.append(pd.DataFrame({f"{cat}__oh__{c}": (col == c).astype(np.float32)}))
        for cat, mapping in self.state.freq_encoding.items():
            col = df[cat].astype(str).str.lower().str.strip()
            cat_frames.append(pd.DataFrame({f"{cat}__freq": col.map(mapping).fillna(0.0).astype(np.float32)}))
        for cat, mapping in self.state.target_encoding.items():
            col = df[cat].astype(str).str.lower().str.strip()
            cat_frames.append(
                pd.DataFrame(
                    {f"{cat}__tgt": col.map(mapping).fillna(self.state.target_global_mean).astype(np.float32)}
                )
            )

        # Binary flags
        flag_frame = pd.DataFrame()
        for f in self.state.binary_flags:
            flag_frame[f] = df[f].astype(np.float32)

        # Context counts
        ctx_frame = pd.DataFrame()
        for c in self.schema.context_counts:
            ctx_frame[c] = df[c].astype(np.float32)

        out = pd.concat(
            [num.astype(np.float32), *cat_frames, flag_frame, ctx_frame],
            axis=1,
        )
        # Reorder strictly to fitted columns; missing columns become 0.0
        for col in self.state.output_columns:
            if col not in out.columns:
                out[col] = 0.0
        out = out[self.state.output_columns]

        # Carry label through if present
        if self.schema.label_binary in df.columns:
            out[self.schema.label_binary] = df[self.schema.label_binary].astype(np.int64).to_numpy()
        if self.schema.label_multiclass in df.columns:
            out[self.schema.label_multiclass] = df[self.schema.label_multiclass].astype(str).to_numpy()
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _cleanup(self, df: pd.DataFrame, *, training: bool) -> pd.DataFrame:
        # Categorical normalization
        for cat in self.schema.categorical:
            if cat in df.columns:
                df[cat] = (
                    df[cat]
                    .fillna(self.cfg.cleanup.fillna_categorical)
                    .astype(str)
                    .str.lower()
                    .str.strip()
                )
        # Numeric fill
        for feat in [*self.schema.numeric, *self.schema.context_counts]:
            if feat in df.columns:
                df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(self.cfg.cleanup.fillna_numeric)
        # Binary flags fill
        for flag in self.schema.binary_flags:
            if flag in df.columns:
                df[flag] = pd.to_numeric(df[flag], errors="coerce").fillna(0).astype(np.int64)
        # Drop duplicates only at training
        if training and self.cfg.cleanup.drop_duplicates:
            before = len(df)
            df = df.drop_duplicates(subset=self.cfg.cleanup.duplicates_subset).reset_index(drop=True)
            if len(df) != before:
                logger.info("dropped %d duplicate rows during fit", before - len(df))
        return df

    def _apply_clip(self, df_num: pd.DataFrame, state: FittedState) -> pd.DataFrame:
        for feat in df_num.columns:
            df_num[feat] = df_num[feat].clip(state.clip_low[feat], state.clip_high[feat])
        return df_num

    @staticmethod
    def _apply_log(df_num: pd.DataFrame, log_features: list[str]) -> pd.DataFrame:
        for feat in log_features:
            if feat in df_num.columns:
                df_num[feat] = np.log1p(df_num[feat].clip(lower=0.0))
        return df_num
