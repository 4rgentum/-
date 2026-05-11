"""Reproducible demonstration run.

Runs the AI-attacker against the CNN-LSTM detector for a short interval
and emits a fully reproducible artifact set used in the thesis defense:

    * demo_alerts.jsonl     — per-window alerts with severity / score / state
    * demo_drift.jsonl      — drift-monitor readings (PSI, KL, MMD per minute)
    * demo_summary.json     — overall metrics summary
    * demo_timeline.png     — score / threshold / FSM state vs time
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diploma_nids.attacker import AttackerRuntime, records_to_dataframe
from diploma_nids.data import Preprocessor, WindowingConfig, build_windows
from diploma_nids.eval import classification_metrics, drift_report, integral_metrics
from diploma_nids.inference import AlertFormer
from diploma_nids.models import build_model
from diploma_nids.utils import append_jsonl, get_logger, load_yaml, save_json, set_seed, setup_logging

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
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/demo")
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    alerts_path = out / "demo_alerts.jsonl"
    drift_path = out / "demo_drift.jsonl"
    summary_path = out / "demo_summary.json"
    timeline_pdf = out / "demo_timeline.pdf"
    timeline_png = out / "demo_timeline.png"
    if alerts_path.exists():
        alerts_path.unlink()
    if drift_path.exists():
        drift_path.unlink()

    # ---------------- Load model + preprocessor ----------------
    cfg = load_yaml(args.model)
    model = build_model(cfg["name"], cfg)
    model.load(args.checkpoint)
    prep = Preprocessor.load(args.preprocessor)

    threshold = args.threshold
    if threshold is None:
        e1 = Path("experiments/runs/E1_results.csv")
        if e1.is_file():
            df = pd.read_csv(e1)
            sub = df[(df["model"] == cfg["name"]) & (df["seed"] == args.seed)]
            if len(sub):
                threshold = float(sub["threshold"].iloc[0])
        if threshold is None:
            threshold = 0.5
    logger.info("demo threshold=%.4f", threshold)

    W = int(cfg["architecture"].get("window_size", 32))
    S = max(1, W // 4)
    wcfg = WindowingConfig(window_size=W, stride=S, label_aggregation="last")

    former = AlertFormer(threshold=threshold, dedup_window_seconds=2.0, history_size=10_000)

    # ---------------- Run attacker ----------------
    runtime = AttackerRuntime(args.policy, drift_path=args.drift, seed=args.seed)
    logger.info("starting demo: %d ticks", args.ticks)
    samples = runtime.collect(args.ticks)
    df = records_to_dataframe(samples)
    logger.info("emitted %d records / %d ticks", len(df), df["tick_idx"].nunique())

    # ---------------- Preprocess + window + score ----------------
    df_pp = prep.transform(df)
    X = df_pp[prep.state.output_columns].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    attack_cat = df["attack_cat"].to_numpy()
    states = df["fsm_state"].to_numpy()
    ticks = df["tick_idx"].to_numpy()

    Xw, yw = build_windows(X, y, wcfg)
    idx = np.arange(0, len(X) - W + 1, S) + W - 1
    attack_cat_w = attack_cat[idx][: len(yw)]
    states_w = states[idx][: len(yw)]
    ticks_w = ticks[idx][: len(yw)]
    scores = _dl_score(model, Xw)
    pred = (scores >= threshold).astype(int)

    # ---------------- Stream-style alerting ----------------
    base_ts = time.time()
    for i in range(len(yw)):
        ts = base_ts + float(ticks_w[i])
        alert = former.maybe_alert(
            score=float(scores[i]),
            attack_cat_pred=str(attack_cat_w[i]) if pred[i] else None,
            profile={"fsm_state": str(states_w[i]), "tick": int(ticks_w[i])},
            ts=ts,
        )
        if alert is not None:
            append_jsonl(
                {
                    "alert_id": alert.alert_id,
                    "ts": alert.timestamp,
                    "tick": int(ticks_w[i]),
                    "severity": alert.severity,
                    "score": alert.score,
                    "decision": alert.decision,
                    "fsm_state": str(states_w[i]),
                    "attack_cat_pred": alert.attack_cat_pred,
                    "ground_truth_label": int(yw[i]),
                    "ground_truth_cat": str(attack_cat_w[i]),
                },
                alerts_path,
            )

    # ---------------- Drift monitoring per chunk ----------------
    ref = np.load("data/processed/unsw/tabular/train.npz")["X"][:5000]
    chunk_size = max(1, len(X) // 10)
    for c in range(0, len(X), chunk_size):
        cur = X[c : c + chunk_size]
        if len(cur) < 100:
            continue
        rep = drift_report(ref, cur)
        rep["chunk_start"] = int(c)
        rep["chunk_end"] = int(c + len(cur))
        rep["fsm_state_majority"] = str(pd.Series(states[c : c + chunk_size]).mode().iloc[0])
        append_jsonl(rep, drift_path)

    # ---------------- Summary ----------------
    cls = classification_metrics(yw, pred).to_dict()
    integ = integral_metrics(yw, scores)
    summary = {
        "model": cfg["name"],
        "checkpoint": args.checkpoint,
        "threshold": float(threshold),
        "seed": args.seed,
        "n_records": int(len(df)),
        "n_ticks": int(df["tick_idx"].nunique()),
        "n_windows": int(len(yw)),
        "n_alerts": len(list(open(alerts_path, encoding="utf-8"))) if alerts_path.exists() else 0,
        "metrics": {**cls, **integ},
        "fsm_state_counts": df["fsm_state"].value_counts().to_dict(),
    }
    save_json(summary, summary_path)
    logger.info("summary saved -> %s", summary_path)
    logger.info("metrics: F1=%.3f PR-AUC=%.3f FPR=%.3f n_alerts=%d",
                cls["f1"], integ["pr_auc"], cls["fpr"], summary["n_alerts"])

    # ---------------- Timeline plot ----------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(ticks_w, scores, color="#1f77b4", linewidth=1)
    axes[0].axhline(threshold, color="red", linestyle="--", label=f"threshold = {threshold:.3f}")
    axes[0].fill_between(ticks_w, 0, scores, where=(yw == 1), alpha=0.2, color="#d62728",
                         step="mid", label="attack windows (ground truth)")
    axes[0].set_ylabel("anomaly score")
    axes[0].set_title("Demo run: CNN-LSTM scores vs FSM state")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # FSM state as colored strip
    state_codes = pd.factorize(states_w)[0]
    axes[1].scatter(ticks_w, state_codes, c=state_codes, cmap="tab20", s=6)
    state_labels = pd.factorize(states_w)[1]
    axes[1].set_yticks(range(len(state_labels)))
    axes[1].set_yticklabels(state_labels)
    axes[1].set_xlabel("tick")
    axes[1].set_ylabel("FSM state")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(timeline_pdf)
    fig.savefig(timeline_png, dpi=150)
    plt.close(fig)
    logger.info("timeline -> %s", timeline_pdf)

    return 0


if __name__ == "__main__":
    sys.exit(main())
