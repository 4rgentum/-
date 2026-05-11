"""Evaluate a trained model on the test split with calibration and CI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from diploma_nids.eval import (
    bootstrap_ci,
    calibrate,
    classification_metrics,
    expected_calibration_error,
    find_threshold_for_target_fpr,
    find_threshold_max_f1,
    integral_metrics,
)
from diploma_nids.models import build_model, load_classical
from diploma_nids.utils import get_logger, load_yaml, save_json, set_seed, setup_logging

logger = get_logger(__name__)


def _load(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    return data["X"], data["y"]


def _dl_score(model, X: np.ndarray, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    probs, logits = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch].astype(np.float32))
            out = model(xb)
            probs.append(out.probs.cpu().numpy())
            logits.append(out.logits.cpu().numpy())
    return np.concatenate(probs), np.concatenate(logits)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to model YAML")
    parser.add_argument("--eval", required=True, help="path to eval YAML")
    parser.add_argument("--data-dir", default="data/processed/unsw")
    parser.add_argument("--checkpoint", required=True, help=".pt or .joblib")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)

    model_cfg = load_yaml(args.model)
    eval_cfg = load_yaml(args.eval)
    name = model_cfg["name"]
    is_classical = "architecture" not in model_cfg

    data_root = Path(args.data_dir)
    layout = "windows" if not is_classical else "tabular"
    val_path = data_root / layout / "val.npz"
    test_path = data_root / layout / "test.npz"
    if not test_path.is_file() and layout == "windows":
        layout = "tabular"
        val_path = data_root / layout / "val.npz"
        test_path = data_root / layout / "test.npz"
    X_val, y_val = _load(val_path)
    X_test, y_test = _load(test_path)

    if is_classical:
        model = load_classical(args.checkpoint)
        score_val = model.score(X_val)
        score_test = model.score(X_test)
        logits_val, logits_test = None, None
    else:
        model = build_model(name, model_cfg)
        model.load(args.checkpoint)
        score_val, logits_val = _dl_score(model, X_val)
        score_test, logits_test = _dl_score(model, X_test)

    # Calibration on val (DL only; classical models often output non-logit scores)
    calibrated_test = score_test
    temperature = None
    ece_pre = expected_calibration_error(y_test, score_test, n_bins=int(eval_cfg["calibration"]["ece_n_bins"]))
    ece_post = ece_pre
    if eval_cfg["calibration"]["enabled"] and logits_val is not None and logits_test is not None:
        _, scaler = calibrate(logits_val, y_val)
        calibrated_test = scaler.transform(logits_test)
        ece_post = expected_calibration_error(y_test, calibrated_test, n_bins=int(eval_cfg["calibration"]["ece_n_bins"]))
        temperature = scaler.temperature

    # Threshold selection on val
    thr_cfg = eval_cfg["thresholding"]
    if thr_cfg["strategy"] == "target_fpr":
        threshold = find_threshold_for_target_fpr(y_val, score_val, float(thr_cfg["target_fpr"]))
    elif thr_cfg["strategy"] == "f1_max":
        threshold = find_threshold_max_f1(y_val, score_val)
    else:
        threshold = 0.5

    pred_test = (calibrated_test >= threshold).astype(int)
    cls = classification_metrics(y_test, pred_test).to_dict()
    integ = integral_metrics(y_test, calibrated_test)

    # Bootstrap CI for PR-AUC
    boot = eval_cfg["bootstrap"]
    if boot.get("enabled", True):
        from sklearn.metrics import average_precision_score
        mean, lo, hi = bootstrap_ci(
            y_test, calibrated_test, average_precision_score,
            n_iterations=int(boot.get("n_iterations", 1000)),
            ci=float(boot.get("ci", 0.95)),
            seed=args.seed,
        )
        ci = {"pr_auc_mean": mean, "pr_auc_ci_lo": lo, "pr_auc_ci_hi": hi}
    else:
        ci = {}

    out_payload = {
        "model": name,
        "seed": args.seed,
        "threshold": float(threshold),
        "temperature": temperature,
        "ece_pre": ece_pre,
        "ece_post": ece_post,
        "test": {**cls, **integ, **ci},
    }

    out_path = Path(args.out) if args.out else Path("experiments/runs") / f"{name}_seed{args.seed}_eval.json"
    save_json(out_payload, out_path)
    logger.info("eval saved to %s", out_path)
    import json as _j

    print(_j.dumps(out_payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
