"""Experiment E1: head-to-head comparison of all candidate models on UNSW-NB15.

For each (model, seed):
    - load processed data;
    - build model from YAML;
    - train (DL) or fit (classical);
    - evaluate on val (best-threshold) and test (calibrated for DL);
    - record one row in results CSV.

Outputs:
    experiments/runs/E1_results.csv
    experiments/runs/E1_history_<model>_seed<seed>.json (DL only)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from diploma_nids.eval import (
    bootstrap_ci,
    classification_metrics,
    expected_calibration_error,
    find_threshold_for_target_fpr,
    integral_metrics,
)
from diploma_nids.eval.thresholding import calibrate
from diploma_nids.models import build_model, save_classical
from diploma_nids.training import Trainer
from diploma_nids.utils import get_logger, load_yaml, save_json, set_seed, setup_logging

logger = get_logger(__name__)

DL_MODELS = ["mlp", "cnn1d", "lstm", "gru", "bilstm", "cnn_lstm", "tcn", "transformer"]
CLASSICAL_MODELS = ["logistic_regression", "random_forest", "xgboost"]
SEMI_SUPERVISED = ["autoencoder", "vae"]
UNSUPERVISED = ["isolation_forest"]


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


def run_one(
    model_name: str,
    seed: int,
    data_dir: Path,
    train_cfg_path: str,
    models_dir: Path,
    target_fpr: float,
) -> dict:
    set_seed(seed)
    model_cfg_path = Path("configs/models") / f"{model_name}.yaml"
    classical_cfg_path = Path("configs/models/classical.yaml")
    is_classical = model_name in CLASSICAL_MODELS + UNSUPERVISED
    is_semi = model_name in SEMI_SUPERVISED

    cfg = load_yaml(classical_cfg_path) if is_classical else load_yaml(model_cfg_path)

    # Load data
    layout = "windows" if model_name in DL_MODELS + SEMI_SUPERVISED else "tabular"
    X_train, y_train = _load(data_dir / layout / "train.npz")
    X_val, y_val = _load(data_dir / layout / "val.npz")
    X_test, y_test = _load(data_dir / layout / "test.npz")

    t0 = time.perf_counter()

    if is_classical:
        model = build_model(model_name, cfg)
        model.fit(X_train, y_train)
        score_val = model.score(X_val)
        score_test = model.score(X_test)
        logits_val, logits_test = None, None
        save_classical(model, models_dir / f"{model_name}_seed{seed}.joblib")
    else:
        model = build_model(model_name, cfg)
        ckpt = models_dir / f"{model_name}_seed{seed}.pt"
        train_cfg = load_yaml(train_cfg_path)
        train_cfg["reproducibility"]["seed"] = seed
        trainer = Trainer(train_cfg)
        # For AE/VAE: train only on normal windows
        if is_semi:
            mask = y_train == 0
            Xtr, ytr = X_train[mask], y_train[mask]
        else:
            Xtr, ytr = X_train, y_train
        history = trainer.fit(model, Xtr, ytr, X_val, y_val, ckpt_path=ckpt)
        score_val, logits_val = _dl_score(model, X_val)
        score_test, logits_test = _dl_score(model, X_test)

    t_train = time.perf_counter() - t0

    # Calibration (DL with logits)
    ece_pre = expected_calibration_error(y_test, score_test)
    if logits_val is not None and logits_test is not None and not is_semi:
        _, scaler = calibrate(logits_val, y_val)
        calibrated_test = scaler.transform(logits_test)
        ece_post = expected_calibration_error(y_test, calibrated_test)
        temperature = scaler.temperature
    else:
        calibrated_test = score_test
        ece_post = ece_pre
        temperature = None

    # Threshold from val under target FPR
    threshold = find_threshold_for_target_fpr(y_val, score_val, target_fpr)
    pred_test = (calibrated_test >= threshold).astype(int)
    cls = classification_metrics(y_test, pred_test).to_dict()
    integ = integral_metrics(y_test, calibrated_test)

    # PR-AUC CI
    from sklearn.metrics import average_precision_score
    pr_mean, pr_lo, pr_hi = bootstrap_ci(
        y_test, calibrated_test, average_precision_score, n_iterations=300, seed=seed
    )

    # Latency (single-sample inference for DL)
    if not is_classical:
        import torch
        x = torch.from_numpy(X_test[:1].astype(np.float32))
        with torch.no_grad():
            for _ in range(3):
                model(x)  # warmup
            t1 = time.perf_counter()
            for _ in range(50):
                model(x)
            latency_ms = (time.perf_counter() - t1) / 50 * 1000.0
    else:
        t1 = time.perf_counter()
        for _ in range(20):
            model.score(X_test[:1])
        latency_ms = (time.perf_counter() - t1) / 20 * 1000.0

    return {
        "model": model_name,
        "seed": seed,
        "is_classical": is_classical,
        "is_semi_supervised": is_semi,
        "n_train_windows": int(len(X_train)),
        "n_test_windows": int(len(X_test)),
        "train_time_s": round(t_train, 2),
        "threshold": float(threshold),
        "temperature": temperature,
        "ece_pre": float(ece_pre),
        "ece_post": float(ece_post),
        "precision": cls["precision"],
        "recall": cls["recall"],
        "f1": cls["f1"],
        "fpr": cls["fpr"],
        "mcc": cls["mcc"],
        "tp": cls["tp"],
        "fp": cls["fp"],
        "tn": cls["tn"],
        "fn": cls["fn"],
        "roc_auc": integ["roc_auc"],
        "pr_auc": integ["pr_auc"],
        "pr_auc_ci_lo": pr_lo,
        "pr_auc_ci_hi": pr_hi,
        "latency_ms_per_sample": round(latency_ms, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/unsw")
    parser.add_argument("--train", default="configs/train/full.yaml")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="subset to run; defaults to a representative set",
    )
    parser.add_argument("--out", default="experiments/runs/E1_results.csv")
    args = parser.parse_args()

    setup_logging("INFO")
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.models is None:
        targets = [
            "cnn_lstm", "bilstm", "lstm", "gru", "cnn1d", "mlp",
            "logistic_regression", "random_forest", "xgboost",
            "isolation_forest",
        ]
    else:
        targets = args.models

    rows: list[dict] = []
    if out_path.is_file():
        rows = pd.read_csv(out_path).to_dict(orient="records")
    seen = {(r["model"], r["seed"]) for r in rows}

    for model_name in targets:
        for seed in args.seeds:
            key = (model_name, seed)
            if key in seen:
                logger.info("skip already done: %s seed=%d", model_name, seed)
                continue
            logger.info("===== model=%s seed=%d =====", model_name, seed)
            try:
                row = run_one(model_name, seed, data_dir, args.train, models_dir, args.target_fpr)
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_path, index=False)
                logger.info("recorded: %s", json.dumps({k: row[k] for k in ("model","seed","f1","pr_auc","fpr","latency_ms_per_sample")}))
            except Exception as exc:  # noqa: BLE001
                logger.exception("model=%s seed=%d FAILED: %s", model_name, seed, exc)

    logger.info("E1 done. results -> %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
