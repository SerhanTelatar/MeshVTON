.PHONY: install train infer preprocess evaluate test lint format serve clean

# ============================================================================
# MeshVTON Makefile
# ============================================================================

PYTHON ?= python
PIP ?= pip
CONFIG_DIR ?= configs

# --- Installation ---
install:
	$(PIP) install -e ".[dev]"

install-all:
	$(PIP) install -e ".[dev,serve,threed]"

# --- Training ---
train:
	$(PYTHON) scripts/train.py --config $(CONFIG_DIR)/train.yaml

train-resume:
	$(PYTHON) scripts/train.py --config $(CONFIG_DIR)/train.yaml --resume

# --- Inference ---
infer:
	$(PYTHON) scripts/inference.py --config $(CONFIG_DIR)/inference.yaml

# --- Preprocessing ---
preprocess:
	$(PYTHON) scripts/preprocess_dataset.py --config $(CONFIG_DIR)/data/preprocessing.yaml

# --- Evaluation ---
evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG_DIR)/inference.yaml

# --- Export ---
export:
	$(PYTHON) scripts/export_onnx.py --config $(CONFIG_DIR)/inference.yaml

# --- Testing ---
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html

# --- Code Quality ---
lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/
	$(PYTHON) -m mypy src/ --ignore-missing-imports

format:
	$(PYTHON) -m black src/ tests/ scripts/
	$(PYTHON) -m ruff check --fix src/ tests/ scripts/

# --- Serving ---
serve:
	cd serving && uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

serve-docker:
	cd serving && docker build -t MeshVTON-server . && docker run -p 8000:8000 --gpus all MeshVTON-server

# --- Cleanup ---
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist build *.egg-info
