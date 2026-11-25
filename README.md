# clinical-rag-eval

A biomedical question-answering RAG pipeline on PubMedQA, built to measure what each retrieval
stage adds: BM25 vs dense vs Qdrant hybrid vs hybrid + cross-encoder reranking. It reports
retrieval metrics and per-stage latency, all on one laptop GPU.

**Status:** M1 (retrieval ablation) is done. Reranking, generation with vLLM, RAGAS, the
validated LLM judge, Langfuse tracing and the API come next; see the
[milestones](https://github.com/Arjun-Sivasankar/clinical-rag-eval/milestones).

## Results (retrieval, 1,000 PubMedQA questions, 62,249-abstract corpus)

| Pipeline | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Search p50 / p95 (ms) | End-to-end p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| BM25 (Qdrant sparse, IDF) | 0.793 | 0.908 | 0.934 | 0.956 | 0.843 | 0.865 | 17.2 / 29.5 | 17.3 / 29.6 |
| Dense (bge-small-en-v1.5) | 0.822 | 0.952 | 0.968 | 0.975 | 0.878 | 0.900 | 17.9 / 30.0 | 30.0 / 50.8 |
| Hybrid, RRF fusion | 0.829 | 0.959 | 0.977 | 0.980 | 0.885 | 0.908 | 19.3 / 33.0 | 31.5 / 55.2 |
| Hybrid, DBSF fusion | 0.843 | 0.958 | 0.972 | 0.981 | 0.893 | 0.913 | 18.5 / 31.4 | 30.6 / 54.2 |

Source: `results/m1_retrieval.md` / `.json` (per-query rankings in `results/runs/`).
Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB), Windows 11, Qdrant 1.20.0 in Docker on the
same laptop. Latency is one query at a time (batch size 1), systems interleaved per query;
*search* is the Qdrant call, end-to-end adds query encoding (fp16 on the GPU).

Hybrid search with DBSF fusion is the best of the four on Recall@1, MRR and nDCG; dense alone
beats BM25 by 2.9 points of Recall@1. BM25 normalises document length by the corpus's own mean
(42.4 BM25 tokens per chunk); with fastembed's default of 256 it scored 0.782 Recall@1 (#17).
Significance tests and reranking come in M2.

### Caveats

- PubMedQA questions are derived from article titles, and the contexts are the same articles'
  abstracts without their conclusions, so lexical overlap is high and BM25 is a strong baseline.
- There is no separate dev set: all variants were run on the same 1,000 questions and all are
  reported.
- Re-running the same system changes 1 of 1,000 rankings (approximate HNSW search, fp16), and
  rebuilding the index can move dense results by a few queries.
- Latency is measured on a shared laptop. Systems in one results file run interleaved query by
  query (#19), so they see the same machine state; latencies from different files or runs can
  differ by 10-20 ms and should not be compared.

## How it works

- **Data:** the 1,000 expert-labelled PubMedQA questions (`pqa_labeled`) are the queries, each
  with one gold abstract. The corpus pools the abstracts of `pqa_labeled` + `pqa_unlabeled`
  (62,249 abstracts), so every question has ~62k distractors.
- **Hierarchical chunking:** one chunk per labelled abstract section (BACKGROUND, METHODS,
  RESULTS, ...), long sections split into windows of whole sentences (max 200 words):
  207,664 chunks, 60 words on average. Scoring is per abstract (best chunk per PMID).
- **Qdrant:** one collection, a named dense vector (`BAAI/bge-small-en-v1.5`, 384-d, cosine) and a
  named sparse vector (fastembed `Qdrant/bm25` with Qdrant's IDF modifier) per chunk. Hybrid search
  is one `query_points` call with two prefetches (top 100 each) fused by RRF or DBSF.
- **Indexing cost** (`results/m1_corpus_and_indexing.json`): dense embedding of 207,664 chunks
  137 s on the RTX 5060, BM25 encoding 22 s on CPU, upload to Qdrant 93 s.

## Reproduce

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, an NVIDIA GPU with CUDA 12.8+
drivers (CPU works, slower). From the repo root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
docker compose up -d qdrant

python -m clinical_rag.cli prepare          # download PubMedQA, chunk -> data/
python -m clinical_rag.cli index            # embed + load Qdrant
python -m clinical_rag.cli eval-retrieval   # M1 -> results/m1_retrieval.{json,md}
```

(Use `.venv/bin/python` on Linux/macOS.)
Settings come from environment variables (`QDRANT_URL`, `CRAG_DENSE_MODEL`, `CRAG_DEVICE`, ...;
see `src/clinical_rag/config.py`). Tests run without downloads or a GPU (in-memory Qdrant, fake
encoders): `python -m pytest`.

## Layout

```
src/clinical_rag/   data.py chunking.py encoders.py index.py retrieve.py metrics.py evaluate.py cli.py
results/            metrics JSON + Markdown per run, runs/*.jsonl (gold rank and top 10 per query)
tests/              pytest suite (data loading, chunking, metrics, retrieval on in-memory Qdrant)
```

## Data and model licences

- PubMedQA: MIT licence ([github.com/pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa);
  Jin et al., EMNLP 2019), loaded from the authors' Hugging Face copy `qiaojin/PubMedQA`.
- Models (downloaded at run time, not redistributed): `BAAI/bge-small-en-v1.5` (MIT),
  fastembed `Qdrant/bm25`.

## Licence

MIT, see [LICENSE](LICENSE). Author: Arjun Sivasankar ([github.com/Arjun-Sivasankar](https://github.com/Arjun-Sivasankar)).
