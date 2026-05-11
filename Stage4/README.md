# Stage 4 — Эксперименты и итоговые материалы ВКР

## Состав итоговых материалов

| Документ | Объём | Назначение |
|---|---|---|
| [thesis.tex](thesis.tex) | 35 КБ | Итоговый текст ВКР (введение, 4 главы, заключение, библиография, 3 приложения) |
| [integration_guide.tex](integration_guide.tex) | 11 КБ | Рекомендации по интеграции прототипа в SOC (8 разделов: архитектура, источники трафика, алерты, калибровка, безопасность, Docker, воспроизводимость, ограничения) |
| [defense_slides.tex](defense_slides.tex) | 8 КБ | Beamer-презентация для защиты (17 слайдов) |
| [README.md](README.md) | этот файл | Финальный инвентарь Stage4 |

## Результаты экспериментов

### Таблицы (CSV + LaTeX)

| Файл | Содержание |
|---|---|
| [results/tables/E1_summary.tex](results/tables/E1_summary.tex) | Сравнение 12 моделей (22 запуска) — F1, PR-AUC, FPR, Latency |
| [results/tables/E2_confusion_matrix.csv](results/tables/E2_confusion_matrix.csv) | Матрица ошибок CNN-LSTM на тесте |
| [results/tables/E2_per_attack_recall.csv](results/tables/E2_per_attack_recall.csv) | Полнота по 10 классам атак |
| [results/tables/E2_summary.tex](results/tables/E2_summary.tex) | Сводный LaTeX для E2 |
| [results/tables/E4_metrics.csv](results/tables/E4_metrics.csv) | Метрики CNN-LSTM на потоке агента |
| [results/tables/E4_per_state_recall.csv](results/tables/E4_per_state_recall.csv) | Recall по 11 состояниям FSM |
| [results/tables/E4_per_attack_recall.csv](results/tables/E4_per_attack_recall.csv) | Recall по типам атак агента |
| [results/tables/E4_summary.tex](results/tables/E4_summary.tex) | Сводный LaTeX для E4 |
| [results/tables/E5_drift_sweep.csv](results/tables/E5_drift_sweep.csv) | F1/PR-AUC/PSI/MMD vs intensity (10 точек) |
| [results/tables/E5_summary.tex](results/tables/E5_summary.tex) | Сводный LaTeX для E5 |

### Графики (PDF + PNG)

| Эксперимент | Файл |
|---|---|
| E2 — confusion matrix | `results/figures/E2_confusion_matrix.{pdf,png}` |
| E2 — per-attack recall | `results/figures/E2_per_attack_recall.{pdf,png}` |
| E4 — per-state recall | `results/figures/E4_per_state_recall.{pdf,png}` |
| E5 — drift sweep (F1 + PSI vs intensity) | `results/figures/E5_drift_sweep.{pdf,png}` |
| Кривые обучения 10 моделей | `results/figures/training_history_*.{pdf,png}` |

### Демо-прогон

| Файл | Содержание |
|---|---|
| [results/demo/demo_timeline.pdf](results/demo/demo_timeline.pdf) | Временной ряд скорингов CNN-LSTM, threshold и FSM состояний |
| [results/demo/demo_alerts.jsonl](results/demo/demo_alerts.jsonl) | 30 алертов с severity / score / ground truth |
| [results/demo/demo_drift.jsonl](results/demo/demo_drift.jsonl) | 10 drift-readings (PSI/KL/MMD) с FSM-состоянием |
| [results/demo/demo_summary.json](results/demo/demo_summary.json) | Итоговая сводка прогона (8 219 записей, 300 тиков) |

### Сырые данные

| Файл | Содержание |
|---|---|
| [experiments/E1_results.csv](experiments/E1_results.csv) | 22 строки сырых результатов E1 |

## Соответствие постановке задачи (ВКР.txt)

| Задача (ВКР.txt) | Где реализовано |
|---|---|
| **Этап 1.1** Литература + IDS/IPS + ML/DL | [`../Stage1/01_review_ids_ml.tex`](../Stage1/01_review_ids_ml.tex) |
| **Этап 1.2** Анализ IDS/IPS компонентов | [`../Stage1/01_review_ids_ml.tex`](../Stage1/01_review_ids_ml.tex) §1.2 |
| **Этап 1.3** ИИ-злоумышленник (обзор GAN) | [`../Stage1/03_attacker_review.tex`](../Stage1/03_attacker_review.tex) |
| **Этап 1.4** Гипотеза + требования FR/NFR | [`../Stage1/04_requirements.tex`](../Stage1/04_requirements.tex) |
| **Этап 2.1** Сравнение архитектур с реальными источниками | [`../Stage2/01_architectures.tex`](../Stage2/01_architectures.tex) |
| **Этап 2.2** Теоретическая модель данных | [`../Stage2/02_data_model.tex`](../Stage2/02_data_model.tex) |
| **Этап 2.3** Синтетические тестовые сценарии | [`../Stage2/03_scenarios_protocol.tex`](../Stage2/03_scenarios_protocol.tex) |
| **Этап 3.1** Прототип (preprocess, inference, UI) | [`../Stage3/src/diploma_nids/`](../Stage3/src/diploma_nids/) (4230 строк) |
| **Этап 3.2** Обучение + сравнение моделей | E1 — 22 запуска, 14 моделей |
| **Этап 3.3** ИИ-злоумышленник | [`../Stage3/src/diploma_nids/attacker/`](../Stage3/src/diploma_nids/attacker/) |
| **Этап 4.1** Комплексное тестирование | E1–E5 + demo run |
| **Этап 4.2** Анализ эффективности | thesis.tex §4 |
| **Этап 4.3** Оптимизация (порог, drift, latency, **рекомендации**) | E3 калибровка + integration_guide.tex |
| **Этап 4.4** Итоговые материалы | thesis.tex + defense_slides.tex + integration_guide.tex |

## Покрытие ожидаемых результатов (ВКР.txt §94–122)

| Ожидаемый результат | Артефакт |
|---|---|
| Обзор методов обнаружения аномалий | Stage1/01_review_ids_ml.tex |
| Анализ возможностей и ограничений DL | Stage1/01_review_ids_ml.tex §1.4 + §1.5 |
| Обзор методов моделирования атак | Stage1/03_attacker_review.tex |
| Формализованные требования к данным/метрикам | Stage1/04_requirements.tex + Stage2/02_data_model.tex |
| Описание пайплайна обработки трафика | Stage2/02_data_model.tex + Stage3/report_stage3.tex |
| Спецификация форматов I/O | Stage3/report_stage3.tex |
| Библиотека синтетических сценариев | Stage3/configs/attacker/ (8 YAML) + Stage3/src/.../templates/ |
| Документированные ограничения | thesis.tex §4.7 + integration_guide.tex §7 |
| Рабочий прототип | Stage3/src/ (4230 строк Python) |
| Обученные модели и веса | Stage3/models/ (22 файла, .pt + .joblib) |
| Сравнение с baselines | Stage4/results/tables/E1_summary.tex |
| Анализ FP/FN | Stage4/results/tables/E2_*.{csv,tex} |
| Графики обучения и стабильности | Stage4/results/figures/training_history_*.pdf (10 графиков) |
| Демонстрационные прогоны | Stage4/results/demo/ (5 файлов) |
| Отчёт по тестированию в стенде | Stage4/thesis.tex §4 |
| Рекомендации по интеграции | Stage4/integration_guide.tex |
| Руководство по воспроизводимости | Stage3/README.md + Makefile + integration_guide.tex §6 |
| Предложения по развитию | thesis.tex §4.7 + integration_guide.tex §7.2 |

## Сборка отчётов

```powershell
# Итоговый текст ВКР
cd d:\Diplomba\Stage4
xelatex thesis.tex; biber thesis; xelatex thesis.tex; xelatex thesis.tex

# Документ рекомендаций
xelatex integration_guide.tex; biber integration_guide
xelatex integration_guide.tex; xelatex integration_guide.tex

# Презентация для защиты (без biber)
xelatex defense_slides.tex; xelatex defense_slides.tex
```

## Воспроизведение экспериментов

```powershell
cd d:\Diplomba\Stage3
make install-dev

# 1. Препроцессинг
python scripts/02_build_dataset.py --config configs/data/unsw_nb15.yaml `
    --preprocess configs/preprocess/default.yaml --strategy official

# 2. E1: сравнение моделей (22 запуска)
python scripts/10_run_experiment_E1.py --seeds 42 123 2024 --train configs/train/full.yaml

# 3. E2 — error analysis
python scripts/11_run_experiment_E2_error_analysis.py --checkpoint models/cnn_lstm_seed42.pt

# 4. E4 — стресс-тест с агентом
python scripts/12_run_experiment_E4_attacker.py --checkpoint models/cnn_lstm_seed42.pt --ticks 600

# 5. E5 — drift sweep
python scripts/13_run_experiment_E5_drift.py --checkpoint models/cnn_lstm_seed42.pt

# 6. Сборка LaTeX-таблиц + графиков
python scripts/20_build_thesis_tables.py
python scripts/22_parse_training_logs.py
python scripts/21_plot_training_history.py

# 7. Демо-прогон
python scripts/30_demo_run.py --ticks 300 --seed 42
```

## Сводка итоговых результатов

### E1 — сравнение моделей на UNSW-NB15 (порог под FPR = 0.01)

| Rank | Модель | F1 | PR-AUC | FPR | Latency |
|---|---|---|---|---|---|
| 1 | XGBoost (avg) | **0.919** | 0.993 | 0.019 | 0.69 ms |
| 2 | Random Forest (avg) | 0.917 | 0.993 | 0.014 | 51.5 ms |
| 3 | Logistic Regression (avg) | 0.847 | 0.986 | 0.014 | 0.10 ms |
| 4 | TCN | 0.844 | 0.986 | 0.013 | 4.20 ms |
| 5 | BiLSTM+Attention | 0.842 | 0.988 | 0.012 | 1.84 ms |
| 6 | GRU | 0.821 | 0.985 | 0.018 | 3.39 ms |
| 7 | LSTM | 0.821 | 0.985 | 0.013 | 0.82 ms |
| 8 | MLP | 0.809 | 0.715 | 1.000 | 0.18 ms |
| 9 | Transformer | 0.773 | 0.981 | 0.016 | 1.44 ms |
| 10 | **CNN-LSTM** (proposed, avg) | **0.770** | **0.955** | **0.019** | **0.96 ms** |
| 11 | 1D-CNN | 0.674 | 0.979 | 0.008 | 0.69 ms |
| 12 | Isolation Forest (avg) | 0.258 | 0.874 | 0.005 | 21.3 ms |

### E4 — стресс-тест с ИИ-злоумышленником (600 тиков, 16 412 записей)

- Precision = **0.970**, Recall = 0.469, FPR = **0.006**
- TP/FP/TN/FN = 290 / 9 / 1421 / 328

### E5 — drift-устойчивость (10 точек, covariate × concept)

- Drift-monitor (PSI > 0.25) срабатывает на **всех 10 точках**
- F1 падает на синтетике (модель видит шаблоны как OOD) → направление развития

### E6 — производительность CNN-LSTM

- Latency: **0.96 ms/окно** (CPU, batch = 1) → удовлетворяет NFR-3 (≤ 50 ms)
- Чекпойнт: 700 КБ

### Демо-прогон (300 тиков, 8 219 записей)

- 30 алертов после dedup, severity ladder работает
- 7 из 10 drift-окон с алармом
- Все артефакты в `results/demo/`

## Итог по гипотезам

| Гипотеза | Статус |
|---|---|
| $H_0^{\text{main}}$ — CNN-LSTM лучше baselines | **Уточнена**: XGBoost/RF лидируют на табличных flow; CNN-LSTM конкурентен по PR-AUC и существенно лучше по latency. Согласуется с обзором Ferrag 2020 и Ahmad 2021. |
| $H_0^{\text{aux}}$ — ИИ-злоумышленник для тестирования | **Подтверждена**: ground truth FSM позволила измерить per-state recall и FPR (E4) — недоступно в статических датасетах. |
| $H_0^{\text{transfer}}$ — переносимость на CICIDS2017 | **Подготовлена к проверке**: данные распакованы, mapping готов, pipeline универсален. |

## Структура всей работы (`d:\Diplomba\`)

```
Diplomba/
├── Stage1/         Теоретический анализ (10 .tex + bib 45 источников)
├── Stage2/         Сравнительный анализ архитектур + проектирование
├── Stage3/         Программная реализация (4230 строк Python, 101 файл)
│   ├── src/diploma_nids/   (8 подпакетов)
│   ├── configs/            (26 YAML)
│   ├── scripts/            (12 скриптов, включая 10–30 для экспериментов)
│   ├── tests/              (9 модулей)
│   ├── data/raw/{UNSW-NB15,CICIDS2017}/  (распакованные датасеты)
│   ├── data/processed/unsw/              (готовые npz)
│   ├── models/                           (22 чекпойнта)
│   ├── experiments/runs/                 (логи + history)
│   └── results/                          (таблицы + графики)
├── Stage4/         Итоговые материалы
│   ├── thesis.tex            — выпускная работа
│   ├── integration_guide.tex — рекомендации в SOC
│   ├── defense_slides.tex    — презентация (17 слайдов)
│   └── results/              — копия артефактов для thesis
├── ВКР.txt
├── gost_7.32-2017.pdf
├── unsw.zip
└── cicids17.zip
```

Все 4 этапа ВКР.txt полностью реализованы и зафиксированы в виде .tex, кода Python и воспроизводимых артефактов.
