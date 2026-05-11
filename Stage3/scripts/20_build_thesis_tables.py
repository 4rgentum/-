"""Aggregate experimental CSVs into LaTeX tables for the final thesis."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from diploma_nids.utils import get_logger, setup_logging

logger = get_logger(__name__)


def _fmt(x, fmt="{:.4f}"):
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "---"
    return fmt.format(x)


def _e1_table(e1_csv: Path, out_dir: Path) -> None:
    if not e1_csv.is_file():
        logger.warning("E1 results not found: %s", e1_csv)
        return
    df = pd.read_csv(e1_csv)
    # Aggregate by model: mean +- std across seeds
    agg = df.groupby("model").agg(
        n_seeds=("seed", "count"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        fpr_mean=("fpr", "mean"),
        mcc_mean=("mcc", "mean"),
        roc_auc_mean=("roc_auc", "mean"),
        latency_ms_mean=("latency_ms_per_sample", "mean"),
    ).reset_index().sort_values("f1_mean", ascending=False)

    csv_path = out_dir / "tables" / "E1_summary.csv"
    agg.to_csv(csv_path, index=False)
    logger.info("E1 summary CSV -> %s", csv_path)

    # LaTeX table
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Эксперимент E1: сравнение моделей-кандидатов на UNSW-NB15. "
        r"Все модели обучаются и оцениваются в идентичных условиях; порог калиброван "
        r"под целевой FPR\,$=$\,0{,}01 на валидационной выборке. "
        r"Значения --- среднее по числу seeds.}",
        r"\label{tab:E1-summary}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\small",
        r"\begin{tabularx}{\linewidth}{@{}lcccccccc@{}}",
        r"\toprule",
        r"\textbf{Модель} & \textbf{seeds} & \textbf{F1} & \textbf{PR-AUC} & "
        r"\textbf{ROC-AUC} & \textbf{Precision} & \textbf{Recall} & \textbf{FPR} & "
        r"\textbf{Latency, мс} \\",
        r"\midrule",
    ]
    for _, r in agg.iterrows():
        bold = r["model"] == "cnn_lstm"
        name = r"\textbf{%s}" % r["model"].replace("_", r"\_") if bold else r["model"].replace("_", r"\_")
        f1 = _fmt(r["f1_mean"])
        pr = _fmt(r["pr_auc_mean"])
        roc = _fmt(r["roc_auc_mean"])
        prec = _fmt(r["precision_mean"])
        rec = _fmt(r["recall_mean"])
        fpr = _fmt(r["fpr_mean"])
        lat = _fmt(r["latency_ms_mean"], "{:.2f}")
        line = f"{name} & {int(r['n_seeds'])} & {f1} & {pr} & {roc} & {prec} & {rec} & {fpr} & {lat} \\\\"
        lines.append(line)
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ]
    tex_path = out_dir / "tables" / "E1_summary.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("E1 LaTeX  -> %s", tex_path)


def _e2_table(out_dir: Path) -> None:
    cm_path = out_dir / "tables" / "E2_confusion_matrix.csv"
    pa_path = out_dir / "tables" / "E2_per_attack_recall.csv"
    if not cm_path.is_file() or not pa_path.is_file():
        logger.warning("E2 CSVs missing")
        return
    cm = pd.read_csv(cm_path, index_col=0)
    pa = pd.read_csv(pa_path)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Эксперимент E2: матрица ошибок CNN-LSTM на тестовой выборке UNSW-NB15.}",
        r"\label{tab:E2-confusion}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Истинное / Предсказанное} & \textbf{normal (0)} & \textbf{attack (1)} \\",
        r"\midrule",
        f"normal (0) & {int(cm.iloc[0, 0])} & {int(cm.iloc[0, 1])} \\\\",
        f"attack (1) & {int(cm.iloc[1, 0])} & {int(cm.iloc[1, 1])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Эксперимент E2: полнота CNN-LSTM по классам атак UNSW-NB15 "
        r"(отсортировано по возрастанию recall).}",
        r"\label{tab:E2-per-attack}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{attack\_cat} & \textbf{support} & \textbf{recall} & \textbf{FN} \\",
        r"\midrule",
    ]
    for _, r in pa.iterrows():
        cat = str(r["attack_cat"]).replace("_", r"\_")
        lines.append(f"{cat} & {int(r['support'])} & {r['recall']:.3f} & {int(r['fn'])} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (out_dir / "tables" / "E2_summary.tex").write_text("\n".join(lines), encoding="utf-8")


def _e4_table(out_dir: Path) -> None:
    metrics_path = out_dir / "tables" / "E4_metrics.csv"
    state_path = out_dir / "tables" / "E4_per_state_recall.csv"
    if not metrics_path.is_file() or not state_path.is_file():
        logger.warning("E4 CSVs missing")
        return
    m = pd.read_csv(metrics_path).iloc[0].to_dict()
    s = pd.read_csv(state_path)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Эксперимент E4: качество CNN-LSTM против ИИ-злоумышленника "
        r"(шаблоны + FSM + drift, 600 тиков).}",
        r"\label{tab:E4-overall}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"\textbf{Метрика} & \textbf{Значение} \\",
        r"\midrule",
        f"Precision & {m['precision']:.3f} \\\\",
        f"Recall    & {m['recall']:.3f} \\\\",
        f"F1        & {m['f1']:.3f} \\\\",
        f"FPR       & {m['fpr']:.3f} \\\\",
        f"PR-AUC    & {m['pr_auc']:.3f} \\\\",
        f"Окон, всего & {int(m['n_windows'])} \\\\",
        f"Порог     & {m['threshold']:.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Эксперимент E4: полнота по состояниям FSM-агента.}",
        r"\label{tab:E4-per-state}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Состояние FSM} & \textbf{n} & \textbf{Recall} & \textbf{FPR} \\",
        r"\midrule",
    ]
    for _, r in s.iterrows():
        name = str(r["fsm_state"]).replace("_", r"\_")
        rec = "---" if pd.isna(r["recall"]) else f"{r['recall']:.3f}"
        fpr = "---" if pd.isna(r["fpr"]) else f"{r['fpr']:.3f}"
        lines.append(f"{name} & {int(r['n'])} & {rec} & {fpr} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (out_dir / "tables" / "E4_summary.tex").write_text("\n".join(lines), encoding="utf-8")


def _e5_table(out_dir: Path) -> None:
    path = out_dir / "tables" / "E5_drift_sweep.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Эксперимент E5: устойчивость CNN-LSTM к дрейфу распределений.}",
        r"\label{tab:E5-drift}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"\textbf{Тип} & \textbf{Интенсивность} & \textbf{F1} & \textbf{PR-AUC} & "
        r"\textbf{PSI} & \textbf{MMD} & \textbf{drift\_alarm} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['kind']} & {r['intensity']:.2f} & {r['f1']:.3f} & {r['pr_auc']:.3f} & "
            f"{r['psi']:.3f} & {r['mmd']:.4f} & {'yes' if r['drift_alarm'] else 'no'} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (out_dir / "tables" / "E5_summary.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", default="experiments/runs/E1_results.csv")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    setup_logging("INFO")
    out_dir = Path(args.results)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    _e1_table(Path(args.e1), out_dir)
    _e2_table(out_dir)
    _e4_table(out_dir)
    _e5_table(out_dir)
    logger.info("LaTeX tables built in %s/tables/", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
