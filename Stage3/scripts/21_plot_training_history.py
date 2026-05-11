"""Plot loss / val_pr_auc / val_f1 curves from saved TrainHistory JSONs.

Reads experiments/runs/<model>_seed<seed>_train.json and emits one PDF/PNG
per model under results/figures/training_history_<model>.{pdf,png}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diploma_nids.utils import get_logger, setup_logging

logger = get_logger(__name__)


def _plot_one(history: dict, model: str, out_dir: Path) -> None:
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_pr = history.get("val_pr_auc", [])
    val_f1 = history.get("val_f1", [])
    if not train_loss:
        return
    epochs = list(range(1, len(train_loss) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, label="train_loss", color="#1f77b4")
    axes[0].plot(epochs, val_loss, label="val_loss", color="#d62728")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{model}: focal-loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_pr, label="val_pr_auc", color="#2ca02c")
    axes[1].plot(epochs, val_f1, label="val_f1", color="#9467bd")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title(f"{model}: validation metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"training_history_{model}.pdf")
    fig.savefig(out_dir / f"training_history_{model}.png", dpi=150)
    plt.close(fig)
    logger.info("plotted %s (%d epochs)", model, len(epochs))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="experiments/runs")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()
    setup_logging("INFO")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = sorted(Path(args.runs_dir).glob("*_train.json"))
    if not runs:
        logger.warning("no *_train.json files found in %s", args.runs_dir)
        return 1
    for path in runs:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if "history" not in payload:
            continue
        # filename: <model>_seed<seed>_train.json
        stem = path.stem.replace("_train", "")
        _plot_one(payload["history"], stem, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
