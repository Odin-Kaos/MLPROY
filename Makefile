.PHONY: install lint format test refactor all run-api run-cli

VENV := .venv
PY := $(VENV)/bin/python
UV := uv

install:
	# Create venv and sync dependencies from uv.lock / pyproject.toml
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(UV) venv $(VENV)
	$(UV) sync
	pip install -e .

lint:
	$(UV) run pylint src || true

format:
	$(UV) run black src

test:
	$(UV) run pytest -v --cov=src

refactor: format lint

all: install format lint test

run-api:
	uvicorn src.api.api:app --reload
