"""Experiment E2: error analysis on the best CNN-LSTM checkpoint.

Produces:
    results/tables/E2_confusion_matrix.csv
    results/tables/E2_per_attack_recall.csv
    results/figures/E2_confusion_matrix.png
    results/figures/E2_per_attack_recall.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from diploma_nids.data import UNSWConfig
from diploma_nids.eval import classification_metrics, confusion_matrix_dataframe, per_class_breakdown
from diploma_nids.models import build_model
from diploma_nids.utils import load_yaml, set_seed, setup_logging, get_logger

logger = get_logger(__name__)


def _dl_score(model, X: np.ndarray, batch: int = 512) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch].astype(np.float32))
            out.append(model(xb).probs.cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/models/cnn_lstm.yaml")
    parser.add_argument("--checkpoint", default="models/cnn_lstm_seed42.pt")
    parser.add_argument("--data-dir", default="data/processed/unsw")
    parser.add_argument("--data-cfg", default="configs/data/unsw_nb15.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--results", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)

    # Load test windows + raw labels for attack_cat
    data = np.load(Path(args.data_dir) / "windows" / "test.npz")
    X_test, y_test = data["X"], data["y"]

    # Recreate per-window attack_cat by re-running build on raw with the same seed+shuffle
    # The windowing aggregation is 'last', so we need attack_cat of the last record per window.
    # Cheaper path: re-derive via the official test-set in tabular order shuffled by the same seed.
    data_cfg = UNSWConfig(**load_yaml(args.data_cfg))
    test_df = pd.read_csv(data_cfg.paths.test_csv, low_memory=False)
    if "id" in test_df.columns:
        test_df = test_df.drop(columns=["id"])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(test_df))
    test_df = test_df.iloc[perm].reset_index(drop=True)
    # Replicate 'last' aggregation: attack_cat of the last record in each window
    W = 32
    S = 8
    starts = np.arange(0, len(test_df) - W + 1, S)
    attack_cat = test_df["attack_cat"].iloc[starts + W - 1].to_numpy()

    if len(attack_cat) != len(y_test):
        # Trim to common length
        m = min(len(attack_cat), len(y_test))
        attack_cat = attack_cat[:m]
        y_test = y_test[:m]
        X_test = X_test[:m]

    # Load model
    cfg = load_yaml(args.model)
    model = build_model(cfg["name"], cfg)
    model.load(args.checkpoint)

    scores = _dl_score(model, X_test)
    threshold = args.threshold
    if threshold is None:
        # Reuse E1 threshold if available
        e1 = Path("experiments/runs/E1_results.csv")
        if e1.is_file():
            df = pd.read_csv(e1)
            sub = df[(df["model"] == cfg["name"]) & (df["seed"] == args.seed)]
            if len(sub):
                threshold = float(sub["threshold"].iloc[0])
        if threshold is None:
            threshold = 0.5
    logger.info("using threshold=%.4f", threshold)

    pred = (scores >= threshold).astype(int)

    out = Path(args.results)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix_dataframe(y_test, pred)
    cm.to_csv(out / "tables" / "E2_confusion_matrix.csv")
    logger.info("confusion matrix:\n%s", cm.to_string())

    pcb = per_class_breakdown(attack_cat, y_test, pred)
    pcb.to_csv(out / "tables" / "E2_per_attack_recall.csv", index=False)
    logger.info("per-attack recall:\n%s", pcb.to_string(index=False))

    # Save plots
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns  # type: ignore  # optional
        sns.set_theme(style="whitegrid")

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title("CNN-LSTM confusion matrix (test)")
        fig.tight_layout()
        fig.savefig(out / "figures" / "E2_confusion_matrix.pdf")
        fig.savefig(out / "figures" / "E2_confusion_matrix.png", dpi=150)
        plt.close(fig)

        # Per-class recall
        fig, ax = plt.subplots(figsize=(8, 4))
        pcb_sorted = pcb.sort_values("recall")
        ax.barh(pcb_sorted["attack_cat"], pcb_sorted["recall"], color="#1f77b4")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_title("CNN-LSTM recall per attack_cat (test)")
        for i, v in enumerate(pcb_sorted["recall"]):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center")
        fig.tight_layout()
        fig.savefig(out / "figures" / "E2_per_attack_recall.pdf")
        fig.savefig(out / "figures" / "E2_per_attack_recall.png", dpi=150)
        plt.close(fig)
    except ImportError:
        logger.warning("matplotlib/seaborn not available; skipping figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
