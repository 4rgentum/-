"""Error analysis: confusion matrix and per-class breakdowns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_class_breakdown(
    attack_cat: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """For each unique value of ``attack_cat``, report support, recall and FN.

    ``attack_cat`` is the multiclass label (e.g. 'DoS', 'Reconnaissance', ...).
    The breakdown is computed on rows where the binary ground-truth equals 1
    (i.e. attack samples), so each row is a per-attack-class recall.
    """
    df = pd.DataFrame({"cat": attack_cat, "y": y_true.astype(int), "p": y_pred.astype(int)})
    attack = df[df["y"] == 1]
    rows = []
    for cat, sub in attack.groupby("cat"):
        n = len(sub)
        tp = int((sub["p"] == 1).sum())
        fn = int((sub["p"] == 0).sum())
        rows.append({"attack_cat": str(cat), "support": n, "recall": tp / n if n else 0.0, "fn": fn})
    return pd.DataFrame(rows).sort_values("recall", ascending=True).reset_index(drop=True)


def confusion_matrix_dataframe(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix as _cm

    cm = _cm(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=pd.Index(["true_0", "true_1"], name="actual"),
        columns=pd.Index(["pred_0", "pred_1"], name="predicted"),
    )
