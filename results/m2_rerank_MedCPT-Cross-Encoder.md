# m2_rerank_MedCPT-Cross-Encoder

Corpus: 62249 abstracts, 207664 chunks. Queries: 1000 PubMedQA expert-labelled questions. Dense model: `BAAI/bge-small-en-v1.5`; sparse: `Qdrant/bm25` (BM25 avg_len 42.4 = corpus mean); reranker: `ncbi/MedCPT-Cross-Encoder` (`+rerank`: top 50 chunks; `+rerank_parent`: top 20 parent abstracts). Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB), Windows-11-10.0.26200-SP0; Qdrant 1.20.0 at `http://localhost:6333`.

Scored at abstract level (best chunk per PMID) over the top 100 retrieved chunks; one gold abstract per question, so Recall@k = Hit@k. Latency: one query at a time; *search* is the Qdrant call; *rerank* is the batched cross-encoder pass; *end-to-end* adds query encoding.

| System | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Search p50 / p95 (ms) | Rerank p50 / p95 (ms) | End-to-end p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| hybrid_dbsf | 0.843 | 0.958 | 0.972 | 0.981 | 0.893 | 0.913 | 16.5 / 30.6 | - | 38.6 / 60.0 |
| hybrid_dbsf+rerank | 0.872 | 0.967 | 0.981 | 0.986 | 0.915 | 0.931 | 18.1 / 30.9 | 127.2 / 187.2 | 167.6 / 234.3 |
| hybrid_dbsf+rerank_parent | 0.892 | 0.975 | 0.979 | 0.981 | 0.930 | 0.943 | 16.4 / 31.2 | 98.7 / 133.6 | 136.3 / 182.2 |

Generated 2026-11-25T18:30:44+00:00 by `crag eval-retrieval`.
