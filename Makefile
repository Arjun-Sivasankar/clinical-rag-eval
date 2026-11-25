# One-command targets. On Windows run them from Git Bash, or call the commands directly.
PY ?= .venv/Scripts/python.exe
ifeq ($(OS),)
PY = .venv/bin/python
endif

.PHONY: setup test lint

setup:            ## venv with CUDA 12.8 PyTorch (RTX 50xx) + project
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) torch --index-url https://download.pytorch.org/whl/cu128
	uv pip install --python $(PY) -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests
