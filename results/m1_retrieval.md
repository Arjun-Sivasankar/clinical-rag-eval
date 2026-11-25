# m1_retrieval

Corpus: 62249 abstracts, 207664 chunks. Queries: 1000 PubMedQA expert-labelled questions. Dense model: `BAAI/bge-small-en-v1.5`; sparse: `Qdrant/bm25` (BM25 avg_len 42.4 = corpus mean). Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB), Windows-11-10.0.26200-SP0; Qdrant 1.20.0 at `http://localhost:6333`.

Scored at abstract level (best chunk per PMID) over the top 100 retrieved chunks; one gold abstract per question, so Recall@k = Hit@k. Latency: one query at a time; *search* is the Qdrant call; *end-to-end* adds query encoding.

| System | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Search p50 / p95 (ms) | End-to-end p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.793 | 0.908 | 0.934 | 0.956 | 0.843 | 0.865 | 20.3 / 35.4 | 20.4 / 35.6 |
| dense | 0.822 | 0.950 | 0.966 | 0.973 | 0.877 | 0.899 | 12.9 / 32.7 | 38.1 / 60.2 |
| hybrid_rrf | 0.827 | 0.958 | 0.977 | 0.979 | 0.884 | 0.907 | 17.8 / 31.7 | 31.3 / 61.6 |
| hybrid_dbsf | 0.843 | 0.956 | 0.971 | 0.981 | 0.893 | 0.912 | 18.2 / 27.7 | 26.8 / 37.3 |

Generated 2026-11-25T14:14:26+00:00 by `crag eval-retrieval`.
