# m2_rerank_bge-reranker-base

Corpus: 62249 abstracts, 207664 chunks. Queries: 1000 PubMedQA expert-labelled questions. Dense model: `BAAI/bge-small-en-v1.5`; sparse: `Qdrant/bm25` (BM25 avg_len 42.4 = corpus mean); reranker: `BAAI/bge-reranker-base` (`+rerank`: top 50 chunks; `+rerank_parent`: top 20 parent abstracts). Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB), Windows-11-10.0.26200-SP0; Qdrant 1.20.0 at `http://localhost:6333`.

Scored at abstract level (best chunk per PMID) over the top 100 retrieved chunks; one gold abstract per question, so Recall@k = Hit@k. Latency: one query at a time; *search* is the Qdrant call; *rerank* is the batched cross-encoder pass; *end-to-end* adds query encoding.

| System | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Search p50 / p95 (ms) | Rerank p50 / p95 (ms) | End-to-end p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| hybrid_dbsf | 0.843 | 0.958 | 0.972 | 0.981 | 0.893 | 0.913 | 16.4 / 30.9 | - | 37.9 / 59.4 |
| hybrid_dbsf+rerank | 0.660 | 0.878 | 0.939 | 0.974 | 0.754 | 0.798 | 18.7 / 31.6 | 152.8 / 220.2 | 194.9 / 264.2 |
| hybrid_dbsf+rerank_parent | 0.831 | 0.949 | 0.971 | 0.981 | 0.883 | 0.905 | 16.9 / 30.3 | 109.4 / 149.6 | 149.1 / 198.7 |

Generated 2026-11-25T18:44:35+00:00 by `crag eval-retrieval`.
