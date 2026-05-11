# Stage 1 — Теоретический анализ (промежуточный отчёт)

## Состав

| Файл | Назначение |
|---|---|
| [report_stage1.tex](report_stage1.tex) | Мастер-документ отчёта Этапа 1 |
| [01_review_ids_ml.tex](01_review_ids_ml.tex) | Раздел 1. IDS/NIDS, методы обнаружения аномалий, DL, метрики |
| [02_datasets_comparison.tex](02_datasets_comparison.tex) | Раздел 2. Сравнение 10 датасетов NIDS, обоснование UNSW-NB15 |
| [03_attacker_review.tex](03_attacker_review.tex) | Раздел 3. Моделирование атак, ИИ-злоумышленник |
| [04_requirements.tex](04_requirements.tex) | Раздел 4. Требования к системе и гипотезы |
| [bib/refs.bib](bib/refs.bib) | Библиография (стиль ГОСТ Р 7.0.100-2018) |
| [preamble/](preamble/) | LaTeX-преамбула: пакеты, стиль ГОСТ, команды |

## Сборка

Требуется TeX-дистрибутив с XeLaTeX и Biber (рекомендуется TeX Live 2022+ или MiKTeX).
Стиль `gost-numeric` пакета `biblatex-gost` входит в современные дистрибутивы.

```powershell
# из директории Stage1
xelatex report_stage1.tex
biber   report_stage1
xelatex report_stage1.tex
xelatex report_stage1.tex
```

Результат: `report_stage1.pdf`.

Альтернатива через `latexmk`:

```powershell
latexmk -xelatex -interaction=nonstopmode report_stage1.tex
```

## Соответствие постановке задачи

Отчёт покрывает все четыре задачи Этапа 1, заявленные в `ВКР.txt`:

1. Изучение литературы по теме исследования — раздел 1.
2. Анализ систем IDS/IPS и их компонентов — раздел 1.2, 1.7.
3. Обзор методов создания ИИ-злоумышленника — раздел 3.
4. Формулировка гипотезы и требований к системе — раздел 4.

## Источники

Используются только верифицируемые публикации (IEEE, Springer, ACM, NDSS,
ICML, NeurIPS, IETF RFC) и государственные стандарты. Полный список —
в [bib/refs.bib](bib/refs.bib).
