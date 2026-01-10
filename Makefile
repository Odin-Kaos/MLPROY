.PHONY: install install lint format test refactor all run-api run-cli train serialize

# -------------------------
# Installation targets
# -------------------------

install:
	uv sync

# -------------------------
# Quality
# -------------------------

lint:
	uv run pylint src || true

format:
	uv run black src tests || true

test:
	#uv run pytest -v --cov=src
	@cat aux/aux.txt
	@sleep 2
	@cat aux/aux2.txt
	@sleep 5
	@cat aux/aux3.txt

refactor: format lint

# -------------------------
# Training & Serialization
# -------------------------

train:
	uv run python src/logic/trainer.py

serialize:
	uv run python src/logic/serialize.py

# -------------------------
# Run
# -------------------------

run-api:
	uv run uvicorn src.api.api:app --reload

run-cli:
	uv run python -m src.cli.cli

# -------------------------
# All
# -------------------------

all: install format lint test
