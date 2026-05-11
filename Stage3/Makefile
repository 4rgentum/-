# Makefile для проекта diploma-nids
# Совместимость: GNU Make (PowerShell -> Make 4+ или WSL)

PY        := python
PIP       := $(PY) -m pip
PYTEST    := $(PY) -m pytest
RUFF      := $(PY) -m ruff
MYPY      := $(PY) -m mypy

CONFIG_DATA  ?= configs/data/unsw_nb15.yaml
CONFIG_PREP  ?= configs/preprocess/default.yaml
CONFIG_MODEL ?= configs/models/cnn_lstm.yaml
CONFIG_TRAIN ?= configs/train/default.yaml
CONFIG_EVAL  ?= configs/eval/default.yaml
SEED         ?= 42

.PHONY: help install install-dev clean lint type test test-fast \
        data preprocess train evaluate attack drift-test demo \
        all reproduce

help:
	@echo "Targets:"
	@echo "  install       - install runtime dependencies"
	@echo "  install-dev   - install dev/test/lint dependencies"
	@echo "  lint          - run ruff"
	@echo "  type          - run mypy"
	@echo "  test          - run full pytest suite"
	@echo "  test-fast     - run only smoke tests"
	@echo "  data          - download / prepare raw data"
	@echo "  preprocess    - run preprocessing pipeline"
	@echo "  train         - train model (CONFIG_MODEL=...)"
	@echo "  evaluate      - evaluate model"
	@echo "  attack        - run AI attacker against detector"
	@echo "  drift-test    - run drift-injection experiment"
	@echo "  demo          - launch FastAPI + Streamlit demo"
	@echo "  reproduce     - full pipeline: data -> preprocess -> train -> eval"

install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -e ".[dev]"

clean:
	@$(PY) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	@$(PY) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"

lint:
	$(RUFF) check src tests scripts

type:
	$(MYPY) src

test:
	$(PYTEST) tests

test-fast:
	$(PYTEST) tests -m "not slow"

data:
	$(PY) scripts/01_download_unsw.py

preprocess:
	$(PY) scripts/02_build_dataset.py --config $(CONFIG_DATA) --preprocess $(CONFIG_PREP)

train:
	$(PY) scripts/03_train.py --model $(CONFIG_MODEL) --train $(CONFIG_TRAIN) --seed $(SEED)

evaluate:
	$(PY) scripts/04_evaluate.py --model $(CONFIG_MODEL) --eval $(CONFIG_EVAL)

attack:
	$(PY) scripts/05_run_attacker.py --policy configs/attacker/policy.yaml

drift-test:
	$(PY) scripts/06_drift_test.py --policy configs/attacker/drift.yaml

demo:
	$(PY) scripts/07_demo_realtime.py

reproduce: data preprocess train evaluate
