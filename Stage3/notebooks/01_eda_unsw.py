# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [markdown]
# # 01 — EDA UNSW-NB15
#
# Цель: первичный анализ датасета UNSW-NB15 на предмет
# распределений, дисбаланса классов, корреляций и идентификации
# проблемных признаков.
#
# Запуск: jupytext --to ipynb 01_eda_unsw.py && jupyter lab 01_eda_unsw.ipynb
# или открыть .py напрямую как notebook в JupyterLab/VSCode.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from diploma_nids.data import UNSWConfig, load_unsw_nb15
from diploma_nids.utils import load_yaml, set_seed

set_seed(42)
sns.set_theme(style="whitegrid")

# %%
cfg = UNSWConfig(**load_yaml("../configs/data/unsw_nb15.yaml"))
train_df, test_df = load_unsw_nb15(cfg)

print(f"train: {train_df.shape}  test: {test_df.shape}")
train_df.head()

# %% [markdown]
# ## Дисбаланс классов

# %%
binary_counts = train_df[cfg.schema.label_binary].value_counts()
multi_counts = train_df[cfg.schema.label_multiclass].value_counts()
print("\nbinary distribution (train):")
print(binary_counts)
print(f"attack ratio: {binary_counts.get(1, 0) / len(train_df):.4f}")

print("\nmulticlass distribution (train):")
print(multi_counts)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
binary_counts.plot.bar(ax=axes[0], color=["#2ca02c", "#d62728"])
axes[0].set_title("Binary label distribution")
axes[0].set_xticklabels(["normal (0)", "attack (1)"], rotation=0)

multi_counts.plot.bar(ax=axes[1])
axes[1].set_title("attack_cat distribution")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()

# %% [markdown]
# ## Распределения ключевых признаков (log-scale)

# %%
heavy_tail = ["dur", "sbytes", "dbytes", "sload", "dload", "sinpkt", "dinpkt"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, feat in zip(axes.flat, heavy_tail):
    x = pd.to_numeric(train_df[feat], errors="coerce").fillna(0)
    ax.hist(np.log1p(x.clip(lower=0)), bins=50, color="#1f77b4", alpha=0.8)
    ax.set_title(f"log1p({feat})")
plt.tight_layout()

# %% [markdown]
# ## Кардинальность категориальных признаков

# %%
for cat in cfg.schema.categorical:
    n_unique = train_df[cat].astype(str).str.lower().str.strip().nunique()
    print(f"{cat:10s} unique levels = {n_unique}")

# %% [markdown]
# ## Пропуски и нулевые доли

# %%
nan_pct = (train_df.isna().mean() * 100).sort_values(ascending=False)
print("Top NaN columns:")
print(nan_pct.head(10))

zero_pct = ((train_df.select_dtypes("number") == 0).mean() * 100).sort_values(ascending=False)
print("\nTop zero-value columns:")
print(zero_pct.head(10))

# %% [markdown]
# ## Корреляции числовых признаков с меткой

# %%
num = train_df[cfg.schema.numeric].select_dtypes(include=[np.number]).copy()
num["label"] = train_df[cfg.schema.label_binary]
corr = num.corr(numeric_only=True)["label"].drop("label").sort_values(key=abs, ascending=False)
print("\nTop |corr| with label:")
print(corr.head(15))

fig, ax = plt.subplots(figsize=(10, 5))
corr.head(20).plot.barh(ax=ax)
ax.invert_yaxis()
ax.set_title("Top |corr(numeric, label)|")
plt.tight_layout()

# %% [markdown]
# ## Сводный отчёт EDA
#
# Дисбаланс классов и тяжелохвостые распределения подтверждают необходимость
# log-преобразования и focal-loss; высокая кардинальность `service` подтверждает
# выбор frequency/target encoding для этого признака. Эти выводы фиксируются
# в Stage2 § 2.6, реализуются в `Preprocessor`.
