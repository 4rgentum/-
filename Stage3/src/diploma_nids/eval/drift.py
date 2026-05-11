"""Drift metrics: PSI, KL, MMD."""
from __future__ import annotations

import numpy as np


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    r"""PSI = sum_b (p_b - q_b) * ln(p_b / q_b).

    PSI ranges suggested in industrial practice:
        < 0.1  : no significant change
        0.1 - 0.25 : moderate change
        > 0.25 : significant change
    Bins are computed from quantiles of the reference distribution.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    if ref.size == 0 or cur.size == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return 0.0
    p, _ = np.histogram(ref, bins=edges)
    q, _ = np.histogram(cur, bins=edges)
    p_norm = p.astype(float) / max(p.sum(), 1) + eps
    q_norm = q.astype(float) / max(q.sum(), 1) + eps
    return float(np.sum((p_norm - q_norm) * np.log(p_norm / q_norm)))


def kl_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 50,
    eps: float = 1e-6,
) -> float:
    """KL(P || Q) on histogram-discretized distributions."""
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    lo = float(min(ref.min(), cur.min())) if ref.size and cur.size else 0.0
    hi = float(max(ref.max(), cur.max())) if ref.size and cur.size else 1.0
    if hi - lo < eps:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(ref, bins=edges, density=False)
    q, _ = np.histogram(cur, bins=edges, density=False)
    p = p.astype(float) / max(p.sum(), 1) + eps
    q = q.astype(float) / max(q.sum(), 1) + eps
    return float(np.sum(p * np.log(p / q)))


def maximum_mean_discrepancy(
    reference: np.ndarray,
    current: np.ndarray,
    bandwidth: float | None = None,
    max_samples: int = 1024,
    seed: int = 42,
) -> float:
    """MMD^2 with Gaussian kernel, biased estimator.

    Auto-bandwidth = median pairwise distance over the merged sample
    (median heuristic).
    """
    rng = np.random.default_rng(seed)
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.ndim == 1:
        ref = ref[:, None]
    if cur.ndim == 1:
        cur = cur[:, None]

    if len(ref) > max_samples:
        ref = ref[rng.choice(len(ref), max_samples, replace=False)]
    if len(cur) > max_samples:
        cur = cur[rng.choice(len(cur), max_samples, replace=False)]

    def _sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a2 = (a * a).sum(axis=1)[:, None]
        b2 = (b * b).sum(axis=1)[None, :]
        return np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0)

    if bandwidth is None:
        merged = np.vstack([ref, cur])
        d = _sq_dist(merged, merged)
        flat = d[np.triu_indices_from(d, k=1)]
        bandwidth = float(np.sqrt(np.median(flat) / 2.0)) if flat.size else 1.0
        bandwidth = max(bandwidth, 1e-6)

    s2 = bandwidth * bandwidth

    def _kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-_sq_dist(a, b) / (2.0 * s2))

    Kxx = _kernel(ref, ref)
    Kyy = _kernel(cur, cur)
    Kxy = _kernel(ref, cur)
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def drift_report(
    reference: np.ndarray,
    current: np.ndarray,
    psi_threshold: float = 0.25,
) -> dict[str, float | bool]:
    psi = population_stability_index(reference, current)
    kl = kl_divergence(reference, current)
    mmd = maximum_mean_discrepancy(reference, current)
    return {
        "psi": psi,
        "kl": kl,
        "mmd": mmd,
        "drift_alarm": bool(psi > psi_threshold),
    }
