"""Experiment E4: stress test with the AI-attacker.

Runs the FSM agent against the trained CNN-LSTM and reports:
    - overall classification metrics on the agent stream;
    - per-FSM-state recall (Normal / ATTACK_* / DRIFT_*);
    - per-attack_cat recall;
    - TTD (time-to-detect) in windows for each ATTACK_* run.

Outputs:
    results/tables/E4_metrics.csv
    results/tables/E4_per_state_recall.csv
    results/figures/E4_per_state_recall.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from diploma_nids.attacker import AttackerRuntime, records_to_dataframe
from diploma_nids.data import Preprocessor, WindowingConfig, build_windows
from diploma_nids.eval import classification_metrics, integral_metrics
from diploma_nids.models import build_model
from diploma_nids.utils import get_logger, load_yaml, save_json, set_seed, setup_logging

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
    parser.add_argument("--policy", default="configs/attacker/policy.yaml")
    parser.add_argument("--drift", default="configs/attacker/drift.yaml")
    parser.add_argument("--model", default="configs/models/cnn_lstm.yaml")
    parser.add_argument("--checkpoint", default="models/cnn_lstm_seed42.pt")
    parser.add_argument("--preprocessor", default="data/processed/unsw/preprocessor.json")
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)

    runtime = AttackerRuntime(args.policy, drift_path=args.drift, seed=args.seed)
    samples = runtime.collect(args.ticks)
    df = records_to_dataframe(samples)
    logger.info("attacker emitted %d records / %d ticks", len(df), df["tick_idx"].nunique())

    prep = Preprocessor.load(args.preprocessor)
    df_pp = prep.transform(df)
    feat_cols = prep.state.output_columns
    X = df_pp[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    attack_cat = df["attack_cat"].to_numpy()
    states = df["fsm_state"].to_numpy()

    cfg = load_yaml(args.model)
    model = build_model(cfg["name"], cfg)
    model.load(args.checkpoint)
    expects_windows = bool(getattr(model, "expects_windows", True))

    if expects_windows:
        W = int(cfg["architecture"].get("window_size", 32))
        S = max(1, W // 4)
        wcfg = WindowingConfig(window_size=W, stride=S, label_aggregation="last")
        Xw, yw = build_windows(X, y, wcfg)
        # Align per-record metadata to window granularity: pick the last record in each window.
        idx = np.arange(0, len(X) - W + 1, S) + W - 1
        attack_cat_w = attack_cat[idx][: len(yw)]
        states_w = states[idx][: len(yw)]
        scores = _dl_score(model, Xw)
        y_eval = yw
    else:
        scores = _dl_score(model, X)
        y_eval = y
        attack_cat_w = attack_cat
        states_w = states

    threshold = args.threshold
    if threshold is None:
        e1 = Path("experiments/runs/E1_results.csv")
        if e1.is_file():
            t = pd.read_csv(e1)
            sub = t[(t["model"] == cfg["name"]) & (t["seed"] == args.seed)]
            if len(sub):
                threshold = float(sub["threshold"].iloc[0])
        if threshold is None:
            threshold = 0.5
    logger.info("threshold=%.4f", threshold)
    pred = (scores >= threshold).astype(int)

    cls = classification_metrics(y_eval, pred).to_dict()
    integ = integral_metrics(y_eval, scores)

    # Per-state recall
    rows = []
    for s in np.unique(states_w):
        mask = states_w == s
        sub_y = y_eval[mask]
        sub_p = pred[mask]
        attack_mask = sub_y == 1
        rec = float((sub_p[attack_mask] == 1).mean()) if attack_mask.any() else float("nan")
        fpr = float((sub_p[~attack_mask] == 1).mean()) if (~attack_mask).any() else float("nan")
        rows.append(
            {
                "fsm_state": str(s),
                "n": int(mask.sum()),
                "n_attack": int(attack_mask.sum()),
                "recall": rec,
                "fpr": fpr,
            }
        )
    per_state = pd.DataFrame(rows).sort_values("fsm_state").reset_index(drop=True)

    # Per attack_cat recall
    rows2 = []
    for c in np.unique(attack_cat_w):
        mask = attack_cat_w == c
        sub_y = y_eval[mask]
        sub_p = pred[mask]
        attack_mask = sub_y == 1
        if not attack_mask.any():
            continue
        rec = float((sub_p[attack_mask] == 1).mean())
        rows2.append({"attack_cat": str(c), "n": int(mask.sum()), "recall": rec})
    per_cat = pd.DataFrame(rows2).sort_values("recall").reset_index(drop=True)

    out = Path(args.results)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([{"threshold": threshold, **cls, **integ, "n_windows": int(len(y_eval))}])
    summary.to_csv(out / "tables" / "E4_metrics.csv", index=False)
    per_state.to_csv(out / "tables" / "E4_per_state_recall.csv", index=False)
    per_cat.to_csv(out / "tables" / "E4_per_attack_recall.csv", index=False)

    logger.info("E4 overall: %s", summary.to_dict(orient="records")[0])
    logger.info("per-state recall:\n%s", per_state.to_string(index=False))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(per_state["fsm_state"], per_state["recall"].fillna(0.0), color="#d62728")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_title("CNN-LSTM recall per FSM state under AI-attacker")
        fig.tight_layout()
        fig.savefig(out / "figures" / "E4_per_state_recall.pdf")
        fig.savefig(out / "figures" / "E4_per_state_recall.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
