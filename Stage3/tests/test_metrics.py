"""Unit tests for metrics, calibration and drift."""
from __future__ import annotations

import numpy as np
import pytest

from diploma_nids.eval import (
    TemperatureScaler,
    bootstrap_ci,
    classification_metrics,
    expected_calibration_error,
    find_threshold_for_target_fpr,
    find_threshold_max_f1,
    fp_per_time,
    integral_metrics,
    kl_divergence,
    maximum_mean_discrepancy,
    population_stability_index,
    time_to_detect,
)
from diploma_nids.training import FocalLoss


def test_classification_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 0, 1, 1])
    rep = classification_metrics(y, p)
    assert rep.precision == 1.0
    assert rep.recall == 1.0
    assert rep.f1 == 1.0
    assert rep.fpr == 0.0


def test_classification_metrics_inverted():
    y = np.array([0, 0, 1, 1])
    p = np.array([1, 1, 0, 0])
    rep = classification_metrics(y, p)
    assert rep.recall == 0.0
    assert rep.fpr == 1.0


def test_pr_auc_roc_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    integ = integral_metrics(y, s)
    assert 0.0 <= integ["pr_auc"] <= 1.0
    assert 0.0 <= integ["roc_auc"] <= 1.0


def test_threshold_for_target_fpr_respects_constraint():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.1).astype(int)
    s = rng.random(1000) + 0.5 * y
    thr = find_threshold_for_target_fpr(y, s, target_fpr=0.05)
    pred = (s >= thr).astype(int)
    rep = classification_metrics(y, pred)
    assert rep.fpr <= 0.05 + 1e-3


def test_max_f1_threshold_at_least_default():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    s = rng.random(500) + 0.4 * y
    thr_05 = 0.5
    thr_max = find_threshold_max_f1(y, s)
    f1_05 = classification_metrics(y, (s >= thr_05).astype(int)).f1
    f1_max = classification_metrics(y, (s >= thr_max).astype(int)).f1
    assert f1_max + 1e-9 >= f1_05


def test_ece_perfectly_calibrated_low():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=10000)
    y = (rng.random(10000) < p).astype(int)
    ece = expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.05


def test_temperature_scaler_reduces_logits_when_overconfident():
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.random(n) < 0.3).astype(int)
    # Strongly overconfident logits
    logits = (y * 2 - 1) * 5 + rng.normal(0, 0.5, n)
    scaler = TemperatureScaler().fit(logits, y)
    assert scaler.temperature > 1.0


def test_fp_per_time_basic():
    y = np.array([0, 0, 1, 0, 0])
    p = np.array([1, 0, 1, 1, 0])
    assert fp_per_time(y, p, duration_seconds=10) == 0.2


def test_time_to_detect_first_attack():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0, 0, 0, 0, 1])
    assert time_to_detect(y, p) == 2


def test_time_to_detect_no_attack_returns_none():
    y = np.zeros(5, dtype=int)
    p = np.zeros(5, dtype=int)
    assert time_to_detect(y, p) is None


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    psi = population_stability_index(a, b)
    assert abs(psi) < 0.05


def test_psi_grows_for_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(2, 1, 5000)
    psi = population_stability_index(a, b)
    assert psi > 0.25


def test_kl_zero_for_identical():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    assert kl_divergence(a, b) < 0.05


def test_mmd_zero_for_identical():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    b = rng.normal(0, 1, 500)
    mmd = maximum_mean_discrepancy(a, b)
    assert abs(mmd) < 0.05


def test_mmd_positive_for_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    b = rng.normal(2, 1, 500)
    mmd = maximum_mean_discrepancy(a, b)
    assert mmd > 0.05


def test_bootstrap_ci_contains_sample_metric():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    s = rng.random(500) + 0.5 * y
    pr = float(average_precision_score(y, s))
    mean, lo, hi = bootstrap_ci(y, s, average_precision_score, n_iterations=200, seed=0)
    assert lo - 0.05 <= pr <= hi + 0.05


def test_focal_loss_not_nan():
    import torch

    loss = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.tensor([2.0, -2.0, 0.5, -0.1])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    out = loss(logits, targets)
    assert torch.isfinite(out).item()
    assert out.item() > 0
