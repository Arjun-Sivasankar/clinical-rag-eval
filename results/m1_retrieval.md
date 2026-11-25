# m1_retrieval

Corpus: 62249 abstracts, 207664 chunks. Queries: 1000 PubMedQA expert-labelled questions. Dense model: `BAAI/bge-small-en-v1.5`; sparse: `Qdrant/bm25` (BM25 avg_len 42.4 = corpus mean). Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB), Windows-11-10.0.26200-SP0; Qdrant 1.20.0 at `http://localhost:6333`.

Scored at abstract level (best chunk per PMID) over the top 100 retrieved chunks; one gold abstract per question, so Recall@k = Hit@k. Latency: one query at a time; *search* is the Qdrant call; *end-to-end* adds query encoding.

| System | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Search p50 / p95 (ms) | End-to-end p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.793 | 0.908 | 0.934 | 0.956 | 0.843 | 0.865 | 17.2 / 29.5 | 17.3 / 29.6 |
| dense | 0.822 | 0.952 | 0.968 | 0.975 | 0.878 | 0.900 | 17.9 / 30.0 | 30.0 / 50.8 |
| hybrid_rrf | 0.829 | 0.959 | 0.977 | 0.980 | 0.885 | 0.908 | 19.3 / 33.0 | 31.5 / 55.2 |
| hybrid_dbsf | 0.843 | 0.958 | 0.972 | 0.981 | 0.893 | 0.913 | 18.5 / 31.4 | 30.6 / 54.2 |

Generated 2026-11-25T18:17:57+00:00 by `crag eval-retrieval`.
