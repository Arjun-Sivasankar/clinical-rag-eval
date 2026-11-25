# One-command targets. On Windows run them from Git Bash, or call the `crag` commands directly.
PY ?= .venv/Scripts/python.exe
ifeq ($(OS),)
PY = .venv/bin/python
endif

.PHONY: setup qdrant prepare index eval-retrieval eval-rerank report eval test lint

setup:            ## venv with CUDA 12.8 PyTorch (RTX 50xx) + project
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) torch --index-url https://download.pytorch.org/whl/cu128
	uv pip install --python $(PY) -e ".[dev]"

qdrant:           ## start Qdrant in Docker
	docker compose up -d qdrant

prepare:          ## download PubMedQA and chunk the corpus into data/
	$(PY) -m clinical_rag.cli prepare

index: qdrant     ## embed chunks (dense + BM25) and load Qdrant
	$(PY) -m clinical_rag.cli index

eval-retrieval:   ## M1 ablation -> results/m1_retrieval.{json,md}
	$(PY) -m clinical_rag.cli eval-retrieval

RERANK_SYSTEMS = --systems hybrid_dbsf,hybrid_dbsf+rerank,hybrid_dbsf+rerank_parent
eval-rerank:      ## M2: + cross-encoder rerank of chunks / parent abstracts -> results/m2_rerank_*
	$(PY) -m clinical_rag.cli eval-retrieval $(RERANK_SYSTEMS) \
		--reranker ncbi/MedCPT-Cross-Encoder --name m2_rerank_MedCPT-Cross-Encoder
	$(PY) -m clinical_rag.cli eval-retrieval $(RERANK_SYSTEMS) \
		--reranker cross-encoder/ms-marco-MiniLM-L6-v2 --name m2_rerank_ms-marco-MiniLM-L6-v2
	$(PY) -m clinical_rag.cli eval-retrieval $(RERANK_SYSTEMS) \
		--reranker BAAI/bge-reranker-base --name m2_rerank_bge-reranker-base

report:           ## results table -> results/summary.md and the README
	$(PY) -m clinical_rag.report

eval: prepare index eval-retrieval eval-rerank report

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests
