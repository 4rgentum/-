"""Smoke tests: every registered model loads from YAML and runs forward."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from diploma_nids.models import (
    DLDetector,
    build_model,
    get_model,
    list_models,
)
from diploma_nids.utils import load_yaml, set_seed

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "models"
DL_MODELS = ["mlp", "cnn1d", "lstm", "gru", "bilstm", "cnn_lstm", "tcn", "transformer", "autoencoder", "vae"]
CLASSICAL_MODELS = ["logistic_regression", "random_forest", "xgboost", "isolation_forest", "ocsvm"]


@pytest.fixture(autouse=True)
def _seed():
    set_seed(0)


def _make_window_batch(B: int, W: int, F: int) -> torch.Tensor:
    return torch.from_numpy(np.random.RandomState(0).normal(size=(B, W, F)).astype(np.float32))


@pytest.mark.parametrize("name", DL_MODELS)
def test_dl_model_loads_and_forwards(name):
    cfg_path = CONFIGS_DIR / f"{name}.yaml"
    if not cfg_path.is_file():
        pytest.skip(f"config not present: {cfg_path}")
    cfg = load_yaml(cfg_path)
    model = build_model(name, cfg)
    assert isinstance(model, DLDetector)

    x = _make_window_batch(4, cfg["architecture"].get("window_size", 1), cfg["architecture"]["input_features"])
    out = model(x)
    assert out.logits.shape[0] == 4
    assert out.probs.shape[0] == 4
    if name not in ("autoencoder", "vae"):
        # supervised models produce probs in [0, 1]
        assert torch.all(out.probs >= 0) and torch.all(out.probs <= 1)


def test_registry_contains_all_dl_and_classical():
    registered = set(list_models())
    expected = set(DL_MODELS) | set(CLASSICAL_MODELS)
    assert expected.issubset(registered), f"missing: {expected - registered}"


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "xgboost", "isolation_forest", "ocsvm"])
def test_classical_fit_predict(name):
    cfg = load_yaml(CONFIGS_DIR / "classical.yaml")
    model = build_model(name, cfg)
    rng = np.random.RandomState(0)
    X = rng.normal(size=(64, 10)).astype(np.float32)
    y = rng.randint(0, 2, size=64)
    if name in ("isolation_forest", "ocsvm"):
        model.fit(X, y)
    else:
        model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (64, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1 + 1e-6)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_model("nonexistent_model_xyz")


def test_save_load_dl_model(tmp_path):
    cfg = load_yaml(CONFIGS_DIR / "cnn_lstm.yaml")
    model = build_model("cnn_lstm", cfg)
    p = tmp_path / "ckpt.pt"
    model.save(p)
    model2 = build_model("cnn_lstm", cfg)
    model2.load(p)
    x = _make_window_batch(2, cfg["architecture"]["window_size"], cfg["architecture"]["input_features"])
    o1 = model(x).probs
    o2 = model2(x).probs
    torch.testing.assert_close(o1, o2)


def test_cnn_lstm_marked_proposed():
    cfg = load_yaml(CONFIGS_DIR / "cnn_lstm.yaml")
    assert cfg.get("proposed") is True
