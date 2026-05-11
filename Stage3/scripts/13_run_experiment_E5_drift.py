"""Experiment E5: drift sensitivity.

Sweeps drift intensity from 0.0 to 1.0 (covariate / concept) and reports
F1 / PR-AUC / PSI / MMD for each level. Produces a CSV and a line plot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from diploma_nids.attacker import AttackerRuntime, DriftInjector, records_to_dataframe
from diploma_nids.attacker.templates import build_template
from diploma_nids.data import Preprocessor, WindowingConfig, build_windows
from diploma_nids.eval import classification_metrics, drift_report, integral_metrics
from diploma_nids.models import build_model
from diploma_nids.utils import get_logger, load_yaml, set_seed, setup_logging

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


def _make_records(template_name: str, n: int, rng: np.random.Generator):
    cfg_path = Path("configs/attacker") / f"{template_name}.yaml"
    tmpl_cfg = load_yaml(cfg_path) if cfg_path.is_file() else {"params": {}}
    return build_template(template_name, tmpl_cfg).generate(n, rng)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/models/cnn_lstm.yaml")
    parser.add_argument("--checkpoint", default="models/cnn_lstm_seed42.pt")
    parser.add_argument("--preprocessor", default="data/processed/unsw/preprocessor.json")
    parser.add_argument("--drift", default="configs/attacker/drift.yaml")
    parser.add_argument("--intensities", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--kinds", nargs="+", default=["covariate", "concept"])
    parser.add_argument("--n-attack", type=int, default=3000)
    parser.add_argument("--n-normal", type=int, default=3000)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)

    cfg = load_yaml(args.model)
    model = build_model(cfg["name"], cfg)
    model.load(args.checkpoint)
    prep = Preprocessor.load(args.preprocessor)
    drift_cfg = load_yaml(args.drift)
    injector = DriftInjector(drift_cfg)

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

    W = int(cfg["architecture"].get("window_size", 32))
    S = max(1, W // 4)
    wcfg = WindowingConfig(window_size=W, stride=S, label_aggregation="last")

    # Reference normal window for drift metrics: from processed data
    ref_X = np.load("data/processed/unsw/tabular/train.npz")["X"][:5000]

    rows = []
    rng = np.random.default_rng(args.seed)
    for kind in args.kinds:
        for inten in args.intensities:
            # Build a stream: half normal + half DDoS, with drift applied
            attack_rec = _make_records("ddos", args.n_attack, rng)
            normal_rec = _make_records("normal", args.n_normal, rng)
            attack_rec = injector.apply(attack_rec, kind, inten, rng)

            df_rows = []
            for r in normal_rec:
                d = r.to_dict()
                d["label"] = 0
                d["attack_cat"] = "Normal"
                df_rows.append(d)
            for r in attack_rec:
                d = r.to_dict()
                d["label"] = 1
                d["attack_cat"] = "DoS"
                df_rows.append(d)
            df = pd.DataFrame(df_rows)
            df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

            df_pp = prep.transform(df)
            X = df_pp[prep.state.output_columns].to_numpy(dtype=np.float32)
            y = df["label"].to_numpy(dtype=np.int64)
            Xw, yw = build_windows(X, y, wcfg)
            scores = _dl_score(model, Xw)
            pred = (scores >= threshold).astype(int)
            cls = classification_metrics(yw, pred).to_dict()
            integ = integral_metrics(yw, scores)
            dr = drift_report(ref_X, X)
            rows.append(
                {
                    "kind": kind,
                    "intensity": inten,
                    "precision": cls["precision"],
                    "recall": cls["recall"],
                    "f1": cls["f1"],
                    "fpr": cls["fpr"],
                    "pr_auc": integ["pr_auc"],
                    "roc_auc": integ["roc_auc"],
                    "psi": float(dr["psi"]),
                    "mmd": float(dr["mmd"]),
                    "drift_alarm": bool(dr["drift_alarm"]),
                }
            )
            logger.info("kind=%s int=%.2f -> F1=%.3f PR-AUC=%.3f PSI=%.3f MMD=%.4f",
                        kind, inten, cls["f1"], integ["pr_auc"], dr["psi"], dr["mmd"])

    out = Path(args.results)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "tables" / "E5_drift_sweep.csv", index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for kind in df["kind"].unique():
            sub = df[df["kind"] == kind]
            axes[0].plot(sub["intensity"], sub["f1"], marker="o", label=kind)
            axes[1].plot(sub["intensity"], sub["psi"], marker="o", label=kind)
        axes[0].set_xlabel("drift intensity")
        axes[0].set_ylabel("F1")
        axes[0].set_title("F1 vs drift intensity")
        axes[0].legend()
        axes[1].set_xlabel("drift intensity")
        axes[1].set_ylabel("PSI")
        axes[1].axhline(0.25, color="red", linestyle="--", label="alarm threshold 0.25")
        axes[1].set_title("PSI vs drift intensity")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(out / "figures" / "E5_drift_sweep.pdf")
        fig.savefig(out / "figures" / "E5_drift_sweep.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
