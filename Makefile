.PHONY: install install-notorch install-withtorch lint format test refactor all run-api run-cli

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

# -------------------------
# Installation targets
# -------------------------

install-notorch:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .[notorch]

install-withtorch:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -e .[withtorch]

# Default install
install: install-withtorch

# -------------------------
# Quality
# -------------------------

lint:
	$(PY) -m pylint src || true

format:
	$(PY) -m black src tests || true

test:
	$(PY) -m pytest -v --cov=src

refactor: format lint

# -------------------------
# Run
# -------------------------

run-api:
	$(UVICORN) src.api.api:app --reload

run-cli:
	$(PY) -m src.cli.cli

all: install-withtorch format lint test

