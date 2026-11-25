# m1_retrieval

Corpus: 62249 abstracts, 207664 chunks. Queries: 1000 PubMedQA expert-labelled questions. Dense model: `BAAI/bge-small-en-v1.5`; sparse: `Qdrant/bm25`. Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB), Windows-11-10.0.26200-SP0; Qdrant 1.20.0 at `http://localhost:6333`.

Scored at abstract level (best chunk per PMID) over the top 100 retrieved chunks; one gold abstract per question, so Recall@k = Hit@k. Latency: one query at a time; *search* is the Qdrant call; *end-to-end* adds query encoding.

| System | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Search p50 / p95 (ms) | End-to-end p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.782 | 0.899 | 0.931 | 0.959 | 0.835 | 0.859 | 20.8 / 34.5 | 20.9 / 34.6 |
| dense | 0.822 | 0.952 | 0.968 | 0.976 | 0.878 | 0.901 | 12.4 / 33.3 | 39.2 / 63.4 |
| hybrid_rrf | 0.834 | 0.954 | 0.973 | 0.981 | 0.887 | 0.908 | 16.0 / 32.7 | 42.6 / 71.1 |
| hybrid_dbsf | 0.847 | 0.957 | 0.970 | 0.981 | 0.896 | 0.915 | 18.7 / 33.0 | 44.0 / 69.8 |

Generated 2026-11-25T13:07:03+00:00 by `crag eval-retrieval`.
