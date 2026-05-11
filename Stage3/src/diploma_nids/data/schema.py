"""Pydantic models for dataset schema and preprocessor configuration.

These models validate user-provided YAML configs and provide a typed handle
for the rest of the data pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class UNSWPaths(BaseModel):
    raw_dir: Path
    train_csv: Path
    test_csv: Path
    interim_dir: Path
    processed_dir: Path


class UNSWFeatureSchema(BaseModel):
    numeric: list[str]
    categorical: list[str]
    binary_flags: list[str] = Field(default_factory=list)
    context_counts: list[str] = Field(default_factory=list)
    label_binary: str
    label_multiclass: str

    @property
    def all_features(self) -> list[str]:
        return [*self.numeric, *self.categorical, *self.binary_flags, *self.context_counts]


class UNSWConfig(BaseModel):
    name: Literal["unsw_nb15"]
    paths: UNSWPaths
    schema: UNSWFeatureSchema
    attack_categories: list[str]


class CleanupConfig(BaseModel):
    fillna_categorical: str = "-"
    fillna_numeric: float = 0.0
    outlier_clip_quantile_low: float = 0.001
    outlier_clip_quantile_high: float = 0.999
    drop_duplicates: bool = True
    duplicates_subset: list[str] | None = None


class EncodingConfig(BaseModel):
    categorical_low_card_threshold: int = 16
    low_card_strategy: Literal["onehot"] = "onehot"
    high_card_strategy: Literal["frequency", "target", "embedding"] = "frequency"
    target_smoothing_alpha: float = 10.0


class ScalingConfig(BaseModel):
    log_transform_features: list[str] = Field(default_factory=list)
    scaler: Literal["robust", "standard", "minmax"] = "robust"
    with_centering: bool = True
    with_scaling: bool = True


class WindowingConfig(BaseModel):
    enabled: bool = True
    window_size: int = 32
    stride: int = 8
    label_aggregation: Literal["any", "majority", "last"] = "any"
    majority_threshold: float = 0.5

    @field_validator("stride")
    @classmethod
    def stride_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("stride must be > 0")
        return v


class SplitsConfig(BaseModel):
    strategy: Literal["temporal", "random", "official"] = "temporal"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    @field_validator("test_ratio")
    @classmethod
    def ratios_sum_to_one(cls, v: float, info) -> float:
        s = info.data.get("train_ratio", 0) + info.data.get("val_ratio", 0) + v
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"train+val+test must sum to 1.0, got {s}")
        return v


class BalancingConfig(BaseModel):
    enabled: bool = True
    method: Literal["class_weights", "smote", "none"] = "class_weights"
    smote_k_neighbors: int = 5
    smote_sampling_strategy: float = 0.5


class PreprocessConfig(BaseModel):
    cleanup: CleanupConfig = Field(default_factory=CleanupConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    windowing: WindowingConfig = Field(default_factory=WindowingConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    balancing: BalancingConfig = Field(default_factory=BalancingConfig)
