# clinical-rag-eval

A biomedical question-answering RAG pipeline on PubMedQA, built to measure what each retrieval
stage adds: BM25 vs dense vs Qdrant hybrid vs hybrid + cross-encoder reranking, with retrieval
metrics, RAGAS and LLM-judge scores, and per-stage latency on one laptop GPU.

Work in progress: see [docs/PLAN.md](docs/PLAN.md) for milestones and the issue tracker for the
current work. Results appear here only once a run in this repo has produced them.

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
pre-commit install
python -m pytest
```

Work runs through issues and pull requests: one branch per issue (`feat/<issue#>-slug`),
conventional commits, CI (ruff + pytest on CPU) must pass before a squash-merge.

## Licence

MIT, see [LICENSE](LICENSE). PubMedQA is MIT-licensed
([github.com/pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa)).
