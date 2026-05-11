"""Parse stdout training logs into TrainHistory JSON files.

The orchestrator writes per-epoch progress to log files. This script extracts
epoch / loss / val_pr_auc / val_f1 rows and emits one JSON per model that
matches the schema expected by ``21_plot_training_history.py``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from diploma_nids.utils import get_logger, setup_logging

logger = get_logger(__name__)

# matches lines like:
# 2026-05-11 04:54:23 | INFO    | diploma_nids.training.trainer | epoch  29 | train_loss=0.0129 val_loss=0.0277 val_pr_auc=0.9836 val_f1=0.9178 best=...
EPOCH_RE = re.compile(
    r"epoch\s+(\d+)\s+\|\s+train_loss=(\d+\.\d+)\s+val_loss=(\d+\.\d+)\s+val_pr_auc=(\d+\.\d+)\s+val_f1=(\d+\.\d+)"
)

# matches start-of-model lines like:
# 2026-05-11 04:51:46 | INFO    | __main__ | ===== model=cnn_lstm seed=42 =====
MODEL_RE = re.compile(r"model=(\S+)\s+seed=(\d+)")


def parse_log(path: Path) -> dict[tuple[str, int], dict]:
    """Return a {(model, seed): history_dict} for each model in the log."""
    histories: dict[tuple[str, int], dict] = {}
    cur_key: tuple[str, int] | None = None
    cur: dict[str, list] | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            mm = MODEL_RE.search(line)
            if mm:
                cur_key = (mm.group(1), int(mm.group(2)))
                cur = {"train_loss": [], "val_loss": [], "val_pr_auc": [], "val_f1": []}
                histories[cur_key] = cur
                continue
            me = EPOCH_RE.search(line)
            if me and cur is not None:
                cur["train_loss"].append(float(me.group(2)))
                cur["val_loss"].append(float(me.group(3)))
                cur["val_pr_auc"].append(float(me.group(4)))
                cur["val_f1"].append(float(me.group(5)))
    return histories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs",
        nargs="+",
        default=["experiments/runs/E1_cnn_lstm_log.txt", "experiments/runs/E1_alt_log.txt"],
    )
    parser.add_argument("--out-dir", default="experiments/runs")
    args = parser.parse_args()
    setup_logging("INFO")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for log in args.logs:
        log_path = Path(log)
        if not log_path.is_file():
            logger.warning("log not found: %s", log_path)
            continue
        for (model, seed), hist in parse_log(log_path).items():
            if not hist["train_loss"]:
                continue
            json_path = out_dir / f"{model}_seed{seed}_train.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"history": hist}, f, ensure_ascii=False, indent=2)
            logger.info("wrote %s (%d epochs)", json_path, len(hist["train_loss"]))
            total += 1
    logger.info("total: %d history files", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
