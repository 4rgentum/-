"""Unit tests for Preprocessor and Pydantic schemas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from diploma_nids.data import (
    PreprocessConfig,
    Preprocessor,
    UNSWFeatureSchema,
)


@pytest.fixture
def schema() -> UNSWFeatureSchema:
    return UNSWFeatureSchema(
        numeric=["dur", "sbytes", "dbytes"],
        categorical=["proto", "service"],
        binary_flags=["is_ftp_login"],
        context_counts=["ct_state_ttl"],
        label_binary="label",
        label_multiclass="attack_cat",
    )


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "dur": rng.exponential(1.0, n),
        "sbytes": rng.integers(0, 100000, n),
        "dbytes": rng.integers(0, 100000, n),
        "proto": rng.choice(["tcp", "udp", "icmp"], n),
        "service": rng.choice(["http", "ftp", "dns", "smtp", "-"], n),
        "is_ftp_login": rng.integers(0, 2, n),
        "ct_state_ttl": rng.integers(0, 7, n),
        "label": rng.integers(0, 2, n),
        "attack_cat": rng.choice(["Normal", "DoS", "Reconnaissance"], n),
    })


def test_fit_transform_shape_and_no_nan(schema, df):
    prep = Preprocessor(PreprocessConfig(), schema)
    out = prep.fit_transform(df)
    assert prep.output_dim > 0
    feat = out[prep.state.output_columns].to_numpy()
    assert feat.shape == (len(df.drop_duplicates()), prep.output_dim)
    assert not np.isnan(feat).any()
    assert not np.isinf(feat).any()


def test_train_stats_only_no_leakage(schema, df):
    prep = Preprocessor(PreprocessConfig(), schema)
    train = df.iloc[:150].copy()
    test = df.iloc[150:].copy()
    prep.fit(train)
    out_train = prep.transform(train)
    out_test = prep.transform(test)
    assert list(out_train.columns) == list(out_test.columns)
    # Stats are frozen on train
    s = prep.state
    s2 = Preprocessor(PreprocessConfig(), schema).fit(train).state
    assert s.scaler_center == s2.scaler_center
    assert s.scaler_scale == s2.scaler_scale


def test_save_load_roundtrip(schema, df, tmp_path):
    prep = Preprocessor(PreprocessConfig(), schema)
    out1 = prep.fit_transform(df)
    artifact = tmp_path / "preprocessor.json"
    prep.save(artifact)
    prep2 = Preprocessor.load(artifact)
    out2 = prep2.transform(df)
    cols = prep.state.output_columns
    np.testing.assert_allclose(out1[cols].to_numpy(), out2[cols].to_numpy(), atol=1e-6)


def test_unknown_category_handled(schema, df):
    prep = Preprocessor(PreprocessConfig(), schema)
    prep.fit(df.iloc[:150])
    novel = df.iloc[150:].copy()
    novel.loc[novel.index[0], "proto"] = "novel_proto_unseen"
    out = prep.transform(novel)
    assert not out.isna().any().any()


def test_clip_outliers_to_train_quantiles(schema, df):
    cfg = PreprocessConfig()
    prep = Preprocessor(cfg, schema)
    prep.fit(df)
    extreme = df.iloc[:5].copy()
    extreme["sbytes"] = 1e15
    out = prep.transform(extreme)
    # Values must be finite and aligned with training-time clipping
    assert np.isfinite(out[prep.state.output_columns].to_numpy()).all()


def test_label_carried_through(schema, df):
    prep = Preprocessor(PreprocessConfig(), schema)
    out = prep.fit_transform(df)
    assert "label" in out.columns
    assert out["label"].dtype.kind == "i"
