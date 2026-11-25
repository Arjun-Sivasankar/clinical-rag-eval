# m2_rerank_ms-marco-MiniLM-L6-v2

Corpus: 62249 abstracts, 207664 chunks. Queries: 1000 PubMedQA expert-labelled questions. Dense model: `BAAI/bge-small-en-v1.5`; sparse: `Qdrant/bm25` (BM25 avg_len 42.4 = corpus mean); reranker: `cross-encoder/ms-marco-MiniLM-L6-v2` (`+rerank`: top 50 chunks; `+rerank_parent`: top 20 parent abstracts). Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB), Windows-11-10.0.26200-SP0; Qdrant 1.20.0 at `http://localhost:6333`.

Scored at abstract level (best chunk per PMID) over the top 100 retrieved chunks; one gold abstract per question, so Recall@k = Hit@k. Latency: one query at a time; *search* is the Qdrant call; *rerank* is the batched cross-encoder pass; *end-to-end* adds query encoding.

| System | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Search p50 / p95 (ms) | Rerank p50 / p95 (ms) | End-to-end p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| hybrid_dbsf | 0.843 | 0.958 | 0.972 | 0.981 | 0.893 | 0.913 | 19.1 / 31.3 | - | 38.6 / 55.9 |
| hybrid_dbsf+rerank | 0.832 | 0.939 | 0.963 | 0.977 | 0.880 | 0.901 | 19.1 / 32.6 | 40.7 / 63.6 | 79.9 / 116.9 |
| hybrid_dbsf+rerank_parent | 0.864 | 0.959 | 0.974 | 0.981 | 0.905 | 0.922 | 19.2 / 31.6 | 33.5 / 52.6 | 72.9 / 106.6 |

Generated 2026-11-25T18:36:02+00:00 by `crag eval-retrieval`.
