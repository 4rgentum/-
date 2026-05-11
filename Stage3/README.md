# diploma-nids

Прототип системы обнаружения несанкционированных действий и аномальной активности
в корпоративной сети на основе нейронных сетей. Программная реализация
выпускной квалификационной работы.

> Тема: «Методика обнаружения несанкционированных действий и аномальной
> активности в корпоративной сети с использованием нейронных сетей».

## Архитектурные решения

| Параметр | Решение |
|---|---|
| Основной датасет | UNSW-NB15 (UNSW Canberra Cyber) |
| Cross-evaluation | CICIDS2017 (CIC) |
| Целевая модель | **CNN-LSTM** |
| Альтернативы | BiLSTM (+attention), Transformer-encoder, TCN |
| Baselines DL | MLP, 1D-CNN, LSTM, GRU, Autoencoder, VAE |
| Baselines ML | Logistic Regression, Random Forest, XGBoost, Isolation Forest, One-Class SVM |
| Окно | W = 32, шаг S = 8 (75% перекрытие) |
| ИИ-злоумышленник | 7 параметризованных шаблонов + FSM-планировщик + drift-инжектор |
| Препроцессинг | clean → log-transform → robust-scale → encode → window |
| Балансировка | focal loss (γ=2.0) + class weights |
| Drift-monitoring | PSI (порог 0.25), KL, MMD |
| Frontend | FastAPI (inference) + Streamlit (UI) |
| Стек | Python 3.10+, PyTorch 2.x, scikit-learn, XGBoost, FastAPI, Streamlit |

Подробное обоснование решений — в [`../Stage1/`](../Stage1) и [`../Stage2/`](../Stage2).

## Структура проекта

```
Stage3/
├── pyproject.toml           # пакет diploma-nids, опц. зависимости [dev]
├── requirements.txt         # runtime-зависимости
├── Makefile                 # data | preprocess | train | evaluate | attack | drift-test | demo
├── configs/
│   ├── data/                # схемы UNSW-NB15, CICIDS2017
│   ├── preprocess/          # параметры pipeline предобработки
│   ├── models/              # архитектуры (cnn_lstm.yaml — proposed)
│   ├── train/               # epochs, optimizer, scheduler, focal loss
│   ├── eval/                # метрики, калибровка, bootstrap CI, drift
│   ├── attacker/            # 7 шаблонов + policy.yaml + drift.yaml
│   └── pipeline/realtime.yaml
├── src/diploma_nids/
│   ├── data/                # схема, загрузка, препроцессинг, окна  (Stage 3.2)
│   ├── models/              # все модели-кандидаты                (Stage 3.3)
│   ├── training/            # trainer, losses, schedulers          (Stage 3.4)
│   ├── eval/                # метрики, threshold, drift, error analysis (Stage 3.4)
│   ├── attacker/            # шаблоны, FSM, drift-инжектор, runtime (Stage 3.5)
│   ├── inference/           # online-сервис, alerting              (Stage 3.6)
│   ├── ui/                  # Streamlit-дэшборд                    (Stage 3.6)
│   └── utils/               # seed, IO, logging                    (Stage 3.1) ✓
├── scripts/                 # 01_download → 07_demo_realtime
├── tests/                   # pytest: utils, configs (+ models, attacker позже)
├── notebooks/               # EDA, error analysis (Stage 3.2 / 4)
├── data/{raw,interim,processed}/       # gitignored
├── models/                  # обученные веса (gitignored)
├── experiments/             # журналы запусков
└── results/{figures,tables}/  # для LaTeX-отчётов
```

## Установка

```powershell
# Среда (рекомендуется отдельный venv)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Зависимости
make install        # или: pip install -r requirements.txt
make install-dev    # с dev/test/lint
```

## Быстрый старт (после Stage 3.2+)

```powershell
make data           # инструкции по получению UNSW-NB15
make preprocess     # обработка raw-CSV → processed
make train          # обучение CNN-LSTM (по умолчанию)
make evaluate       # метрики и таблицы
make attack         # ИИ-злоумышленник против детектора
make drift-test     # стресс drift-сценариев
make demo           # FastAPI + Streamlit
make reproduce      # полный цикл одной командой
```

## Тестирование

```powershell
make test           # полный набор pytest
make test-fast      # без slow
make lint           # ruff
make type           # mypy
```

## Текущий статус (Stage 3 завершён)

- [x] **3.1 Скаффолдинг** — пакет `diploma-nids`, 15 YAML-конфигов, Makefile, utils
- [x] **3.2 Препроцессинг** — Preprocessor (fit/transform), windowing, splits, EDA-нотбук
- [x] **3.3 Модели** — CNN-LSTM (proposed) + 9 альтернатив + 5 classical baselines в едином registry
- [x] **3.4 Training + evaluation** — Trainer (AdamW, cosine, focal, early stop), метрики, калибровка, drift
- [x] **3.5 ИИ-злоумышленник** — 7 templates + FSM (11 состояний) + drift-injector + runtime (offline/online)
- [x] **3.6 FastAPI + Streamlit** — REST inference, WindowScorer, AlertFormer, dashboard, demo-orchestrator
- [x] **3.7 Отчёт** — [`report_stage3.tex`](report_stage3.tex) с traceability проектных решений

## Ссылки на отчёты

- [Stage 1 — Теоретический анализ](../Stage1/)
- [Stage 2 — Сравнительный анализ архитектур и проектирование](../Stage2/)
- Stage 3 — Программная реализация (текущая)
- Stage 4 — Эксперименты и итоговый отчёт (предстоит)

## Лицензия

MIT (учебно-исследовательский характер; использование UNSW-NB15 и CICIDS2017
регулируется лицензиями оригинальных правообладателей).
