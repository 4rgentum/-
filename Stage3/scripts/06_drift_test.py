"""Drift-injection experiment (E5).

Compares feature distributions of agent-generated stream against a reference
(clean) window in the training set, using PSI / KL / MMD. Emits a JSON report
with metric values per ramp-up step.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from diploma_nids.attacker import AttackerRuntime, records_to_dataframe
from diploma_nids.data import Preprocessor
from diploma_nids.eval import drift_report
from diploma_nids.utils import get_logger, save_json, set_seed, setup_logging

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--drift", required=True)
    parser.add_argument("--preprocessor", required=True)
    parser.add_argument("--reference", required=True, help="path to .npz with reference normal data")
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", nargs="+", default=["dur", "sbytes", "dbytes", "sinpkt", "dinpkt"])
    parser.add_argument("--out", default="experiments/runs/drift_report.json")
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)

    runtime = AttackerRuntime(args.policy, drift_path=args.drift, seed=args.seed)
    df = records_to_dataframe(runtime.collect(args.ticks))

    prep = Preprocessor.load(args.preprocessor)
    df_pp = prep.transform(df)

    ref = np.load(args.reference)
    ref_X = ref["X"].astype(np.float32)
    if ref_X.ndim == 3:
        ref_X = ref_X.reshape(-1, ref_X.shape[-1])

    cur_X = df_pp[prep.state.output_columns].to_numpy(dtype=np.float32)

    # Per-feature drift metrics (only for numeric features in the schema)
    feat_index = {col: i for i, col in enumerate(prep.state.output_columns)}
    per_feature: dict[str, dict[str, float | bool]] = {}
    for name in args.features:
        if name not in feat_index:
            logger.warning("feature %s not in preprocessor output; skipping", name)
            continue
        idx = feat_index[name]
        report = drift_report(ref_X[:, idx], cur_X[:, idx])
        per_feature[name] = report

    # Aggregate (joint) MMD on selected features
    sel = [feat_index[n] for n in args.features if n in feat_index]
    joint_psi_mean = float(np.mean([per_feature[n]["psi"] for n in per_feature]))
    joint_mmd = drift_report(ref_X[:, sel], cur_X[:, sel])["mmd"] if sel else 0.0

    payload = {
        "n_ticks": int(args.ticks),
        "n_records_current": int(len(df)),
        "fsm_state_distribution": df["fsm_state"].value_counts().to_dict(),
        "drift_kind_distribution": df["drift_kind"].value_counts().to_dict(),
        "per_feature": per_feature,
        "joint_psi_mean": joint_psi_mean,
        "joint_mmd": float(joint_mmd),
    }
    save_json(payload, args.out)
    logger.info("drift report saved to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
