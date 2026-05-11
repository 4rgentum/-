"""Train a single model.

Reads processed windows (or tabular features), builds a model from YAML,
trains via Trainer, persists best checkpoint and a metrics JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from diploma_nids.eval import (
    classification_metrics,
    integral_metrics,
)
from diploma_nids.models import build_model, save_classical
from diploma_nids.training import Trainer
from diploma_nids.utils import get_logger, load_yaml, save_json, set_seed, setup_logging

logger = get_logger(__name__)


def _load_split(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    return data["X"], data["y"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to model YAML")
    parser.add_argument("--train", required=True, help="path to train config YAML")
    parser.add_argument("--data-dir", default="data/processed/unsw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--metrics-out", default="experiments/runs")
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)

    model_cfg = load_yaml(args.model)
    train_cfg = load_yaml(args.train)
    train_cfg.setdefault("reproducibility", {})["seed"] = args.seed

    name = model_cfg["name"]
    is_classical = model_cfg.get("type", "").startswith("dl_") is False and "architecture" not in model_cfg

    data_root = Path(args.data_dir)
    layout = "windows" if not is_classical else "tabular"
    train_path = data_root / layout / "train.npz"
    val_path = data_root / layout / "val.npz"
    if not train_path.is_file() or not val_path.is_file():
        # Fall back to tabular if windows are missing
        layout = "tabular"
        train_path = data_root / layout / "train.npz"
        val_path = data_root / layout / "val.npz"
    X_train, y_train = _load_split(train_path)
    X_val, y_val = _load_split(val_path)
    logger.info("loaded %s split: train=%s val=%s", layout, X_train.shape, X_val.shape)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.metrics_out)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if is_classical:
        # Classical baselines branch
        model = build_model(name, model_cfg)
        model.fit(X_train, y_train)
        score_val = model.score(X_val)
        pred_val = (score_val >= 0.5).astype(int)
        cls = classification_metrics(y_val, pred_val).to_dict()
        integ = integral_metrics(y_val, score_val)
        metrics = {"val": {**cls, **integ}}
        save_classical(model, out_dir / f"{name}_seed{args.seed}.joblib")
    else:
        from diploma_nids.models import DLDetector

        model = build_model(name, model_cfg)
        assert isinstance(model, DLDetector)
        ckpt = out_dir / f"{name}_seed{args.seed}.pt"
        trainer = Trainer(train_cfg)
        history = trainer.fit(model, X_train, y_train, X_val, y_val, ckpt_path=ckpt)
        # Final eval on val with best weights
        import torch as _t

        model.eval()
        with _t.no_grad():
            outs = []
            B = int(train_cfg["trainer"]["batch_size"])
            for i in range(0, len(X_val), B):
                xb = _t.from_numpy(X_val[i : i + B].astype(np.float32))
                outs.append(model(xb).probs.cpu().numpy())
        score_val = np.concatenate(outs)
        pred_val = (score_val >= 0.5).astype(int)
        cls = classification_metrics(y_val, pred_val).to_dict()
        integ = integral_metrics(y_val, score_val)
        metrics = {
            "val": {**cls, **integ},
            "history": {
                "train_loss": history.train_loss,
                "val_loss": history.val_loss,
                "val_pr_auc": history.val_pr_auc,
                "val_f1": history.val_f1,
            },
        }

    metrics_path = metrics_dir / f"{name}_seed{args.seed}_train.json"
    save_json(metrics, metrics_path)
    logger.info("metrics saved to %s", metrics_path)
    print(json.dumps(metrics["val"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
