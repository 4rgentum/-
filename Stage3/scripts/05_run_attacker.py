"""Run the AI-attacker offline against a trained detector.

Pipeline:
    1. AttackerRuntime emits a stream of FlowRecord with ground-truth labels.
    2. Records are passed through the same Preprocessor that trained the model.
    3. Optional windowing is applied if model expects windows.
    4. Detector scores each input; metrics + per-state recall are reported.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from diploma_nids.attacker import AttackerRuntime, records_to_dataframe
from diploma_nids.data import Preprocessor, WindowingConfig, build_windows
from diploma_nids.eval import classification_metrics, integral_metrics, time_to_detect
from diploma_nids.eval.error_analysis import per_class_breakdown
from diploma_nids.models import build_model, load_classical
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
    parser.add_argument("--policy", required=True)
    parser.add_argument("--drift", default=None)
    parser.add_argument("--model", required=True, help="model YAML")
    parser.add_argument("--checkpoint", required=True, help=".pt or .joblib")
    parser.add_argument("--preprocessor", required=True, help="preprocessor JSON from build_dataset")
    parser.add_argument("--ticks", type=int, default=None, help="override n_ticks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default="experiments/runs/attacker_run.json")
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)

    runtime = AttackerRuntime(args.policy, drift_path=args.drift, seed=args.seed)
    samples = runtime.collect(args.ticks)
    df = records_to_dataframe(samples)
    logger.info("attacker emitted %d samples across %d ticks", len(df), df["tick_idx"].nunique())

    prep = Preprocessor.load(args.preprocessor)
    df_pp = prep.transform(df)
    feat_cols = prep.state.output_columns

    X = df_pp[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    attack_cat = df["attack_cat"].to_numpy()
    states = df["fsm_state"].to_numpy()

    model_cfg = load_yaml(args.model)
    name = model_cfg["name"]
    is_classical = "architecture" not in model_cfg

    if is_classical:
        model = load_classical(args.checkpoint)
        score = model.score(X)
    else:
        model = build_model(name, model_cfg)
        model.load(args.checkpoint)
        if model.expects_windows:
            wcfg = WindowingConfig(
                window_size=int(model_cfg["architecture"].get("window_size", 32)),
                stride=int(model_cfg["architecture"].get("window_size", 32)) // 4 or 1,
            )
            Xw, yw = build_windows(X, y, wcfg)
            score = _dl_score(model, Xw)
            y = yw
            # propagate per-state info to window granularity by majority
            ack = []
            for i in range(len(yw)):
                start = i * wcfg.stride
                ack.append(np.bincount(np.array([0]) if start >= len(states) else np.array([0]))[0])
            attack_cat = attack_cat[: len(yw)]
            states = states[: len(yw)]
        else:
            score = _dl_score(model, X)

    pred = (score >= args.threshold).astype(int)
    cls = classification_metrics(y, pred).to_dict()
    integ = integral_metrics(y, score)
    ttd = time_to_detect(y, pred)

    breakdown = per_class_breakdown(attack_cat, y, pred)

    payload = {
        "n_samples": int(len(y)),
        "n_attack": int(y.sum()),
        "threshold": float(args.threshold),
        "metrics": {**cls, **integ},
        "time_to_detect_windows": ttd,
        "per_state_recall": {
            s: float(((pred[states == s] == 1) & (y[states == s] == 1)).sum() / max((y[states == s] == 1).sum(), 1))
            for s in np.unique(states)
        },
        "per_attack_cat": breakdown.to_dict(orient="records"),
    }
    save_json(payload, args.out)
    logger.info("attacker run saved to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
