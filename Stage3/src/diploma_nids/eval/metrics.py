"""Evaluation metrics for NIDS: classification, integral, calibration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass
class ClassificationReport:
    precision: float
    recall: float
    f1: float
    fpr: float
    mcc: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fpr": self.fpr,
            "mcc": self.mcc,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationReport:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        mcc = 0.0
    return ClassificationReport(
        precision=float(p), recall=float(r), f1=float(f1),
        fpr=float(fpr), mcc=float(mcc),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )


def integral_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """ROC-AUC and PR-AUC (also known as average precision)."""
    out: dict[str, float] = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        out["pr_auc"] = float("nan")
    return out


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (Guo et al., 2017)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def fp_per_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    duration_seconds: float,
) -> float:
    """False positives per second of stream."""
    if duration_seconds <= 0:
        return float("nan")
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return fp / duration_seconds


def time_to_detect(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float | None:
    """Number of windows from first attack onset to first positive prediction.

    Returns ``None`` if no attack ground-truth is present, or if the model
    does not detect any of the attack windows.
    """
    attack_idx = np.flatnonzero(y_true == 1)
    if attack_idx.size == 0:
        return None
    first_attack = int(attack_idx[0])
    detect_idx = np.flatnonzero((y_pred == 1) & (np.arange(len(y_pred)) >= first_attack))
    if detect_idx.size == 0:
        return None
    return float(detect_idx[0] - first_attack)


def find_threshold_for_target_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float
) -> float:
    """Smallest threshold attaining FPR <= target_fpr on the validation set."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    s_sorted = y_score[order]
    n_neg = int((y_true == 0).sum())
    if n_neg == 0:
        return 0.5
    fp_count = 0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 0:
            fp_count += 1
        if fp_count / n_neg > target_fpr:
            return float(s_sorted[i])
    return float(s_sorted[-1] - 1e-12)


def find_threshold_max_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    p, r, t = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / (p + r + 1e-12)
    if len(t) == 0:
        return 0.5
    best = int(np.nanargmax(f1[:-1]))
    return float(t[best])


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval (mean, lo, hi) for a given metric."""
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], y_score[idx]))
    vals = np.asarray(vals, dtype=float)
    alpha = (1 - ci) / 2
    return float(vals.mean()), float(np.quantile(vals, alpha)), float(np.quantile(vals, 1 - alpha))
