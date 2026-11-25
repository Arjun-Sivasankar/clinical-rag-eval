# Plan: clinical-rag-eval

Goal: a biomedical QA RAG pipeline on PubMedQA with an ablation table that shows what each
stage buys, measured end to end on one laptop GPU. Every number in the README comes from a run
in this repo, and every table states the hardware it ran on.

## Status (25 Nov 2026)

- M1 done: full corpus (62,249 abstracts, 207,664 chunks), BM25 / dense / hybrid RRF / hybrid
  DBSF over all 1,000 questions, results in `results/m1_*`.
- Next: M2 (cross-encoder reranking and per-stage latency).

## Data

PubMedQA (MIT licence, github.com/pubmedqa/pubmedqa; Hugging Face mirror `qiaojin/PubMedQA`).

- Queries and gold labels: the 1,000 expert-labelled questions (`pqa_labeled`). Each has one gold
  abstract (its PMID), a yes/no/maybe label and a long answer (the abstract's conclusion).
- Corpus: the contexts of `pqa_labeled` + `pqa_unlabeled` (~62k abstracts) pooled into one Qdrant
  collection, so each query has ~62k distractor abstracts. `pqa_artificial` (211k) is a later
  option for a harder corpus.
- Hierarchical chunking: one child chunk per labelled abstract section (BACKGROUND, METHODS,
  RESULTS, ...), long sections split by sentence windows; the parent is the whole abstract.
  Retrieval runs over chunks and is scored at abstract level (max chunk score per PMID);
  generation gets the parent abstract.
- Caveat to state in the README: PubMedQA questions are derived from article titles and the
  context excludes the conclusion, so lexical overlap is high and BM25 is a strong baseline.

## Milestones

| # | Deliverable | Runs on |
|---|---|---|
| M1 | Data loader, chunking, Qdrant collection with dense + sparse (BM25) vectors; BM25 vs dense vs hybrid (RRF and DBSF fusion): Recall@1/5/10, MRR@10, nDCG@10, p50/p95 latency; results in `results/`; pytest + CI | Laptop (GPU embedder), Qdrant in Docker |
| M2 | Batched cross-encoder reranking over the hybrid top-50; per-stage latency (embed, search, rerank); reranker comparison | Laptop GPU |
| M3 | vLLM generation (OpenAI-compatible server in WSL2/Docker); query rewriting row; RAGAS faithfulness, context recall, answer correctness; LLM judge for yes/no/maybe with agreement vs gold labels (accuracy, Cohen's kappa, confusion matrix) | Laptop (generator); Kaggle for the 8B judge over all 1,000 |
| M4 | Langfuse self-hosted tracing (Docker), prompts versioned there; FastAPI `/search` and `/answer` endpoints; Docker Compose for the full stack | Mac mini or laptop (services) |
| M5 | README results table (5 rows: BM25, dense, hybrid, hybrid+rerank, hybrid+rerank+rewrite), GitHub Pages results page, CPU Hugging Face Space (live retrieval + rerank, cached generations), demo GIF of a Langfuse trace | Laptop; publishing only with Arjun's OK |

## Models that fit 8 GB (RTX 5060 Laptop; ~1.7 GB already taken by the desktop)

VRAM figures are estimates (weights + working memory), to be replaced by measured peaks.

| Role | First choice | Alternatives | Est. VRAM |
|---|---|---|---|
| Dense embedder | `BAAI/bge-small-en-v1.5` (33M, 384-d) | `BAAI/bge-base-en-v1.5` (110M), `ncbi/MedCPT` query/article encoders | 0.3-1 GB fp16 |
| Sparse | `Qdrant/bm25` via fastembed (CPU), Qdrant IDF modifier | SPLADE (`prithivida/Splade_PP_en_v1`) | 0 (CPU) |
| Reranker | `BAAI/bge-reranker-base` (278M) | `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M), `ncbi/MedCPT-Cross-Encoder` | ~1 GB fp16, batch 32 |
| Generator + rewriter | `Qwen/Qwen3-1.7B` bf16 in vLLM, max_model_len 4096 | Qwen3-4B AWQ 4-bit | ~3.5 GB weights + ~1.5 GB KV cache |
| Judge / RAGAS evaluator | Qwen3-4B AWQ on the laptop (subset) | Qwen3-8B AWQ on Kaggle T4 for all 1,000 | ~3 GB local; ~6 GB on Kaggle |

The final model list is fixed at the start of M3 after checking what vLLM supports on sm_120.

## Where things run

- Laptop (Windows, RTX 5060 8 GB): embedding, indexing, retrieval and rerank evaluation (native
  Windows venv with CUDA 12.8 PyTorch); vLLM only inside WSL2 or Docker.
- Docker (laptop now, Mac mini later): Qdrant; Langfuse v3 needs Postgres, ClickHouse, Redis and
  MinIO (~4 GB RAM), better on the Mac mini so the laptop keeps its RAM for vLLM.
- Kaggle (Arjun runs a prepared notebook): the 8B judge and full-set RAGAS if the laptop is too
  slow. Results come back as JSON committed to `results/`.

## Risks

- vLLM on Blackwell (sm_120) in WSL2 needs a CUDA 12.8+ build; fallback is the official
  `vllm/vllm-openai` Docker image, then llama.cpp server (same OpenAI API) with the change stated.
- 8 GB VRAM is shared with the desktop and other sessions: check `nvidia-smi` first, stop servers
  when done, keep max_model_len short.
- Small evaluator models make RAGAS scores noisy: the judge is validated against gold labels
  before its scores are reported, and RAGAS runs are pinned to one version.
- High lexical overlap in PubMedQA may make BM25 hard to beat; report it as found, not tuned away.
- Docker volumes live on C: (Qdrant index ~0.5 GB, rebuildable); the D: drive is exFAT, so no
  Qdrant storage bind mount there.
