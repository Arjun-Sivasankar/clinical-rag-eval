| Pipeline | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Rerank p50 / p95 (ms) | End-to-end p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| BM25 (Qdrant sparse, IDF) | 0.793 | 0.908 | 0.934 | 0.843 | 0.865 | - | 17 / 30 |
| Dense (bge-small-en-v1.5) | 0.822 | 0.952 | 0.968 | 0.878 | 0.900 | - | 30 / 51 |
| Hybrid, RRF fusion | 0.829 | 0.959 | 0.977 | 0.885 | 0.908 | - | 32 / 55 |
| Hybrid, DBSF fusion | 0.843 | 0.958 | 0.972 | 0.893 | 0.913 | - | 31 / 54 |
| Hybrid DBSF + bge-reranker-base, chunks | 0.660 | 0.878 | 0.939 | 0.754 | 0.798 | 153 / 220 | 195 / 264 |
| Hybrid DBSF + bge-reranker-base, abstracts | 0.831 | 0.949 | 0.971 | 0.883 | 0.905 | 109 / 150 | 149 / 199 |
| Hybrid DBSF + MiniLM-L6 (ms-marco), chunks | 0.832 | 0.939 | 0.963 | 0.880 | 0.901 | 41 / 64 | 80 / 117 |
| Hybrid DBSF + MiniLM-L6 (ms-marco), abstracts | 0.864 | 0.959 | 0.974 | 0.905 | 0.922 | 33 / 53 | 73 / 107 |
| Hybrid DBSF + MedCPT cross-encoder, chunks | 0.872 | 0.967 | 0.981 | 0.915 | 0.931 | 127 / 187 | 168 / 234 |
| Hybrid DBSF + MedCPT cross-encoder, abstracts | 0.892 | 0.975 | 0.979 | 0.930 | 0.943 | 99 / 134 | 136 / 182 |

Hardware: NVIDIA GeForce RTX 5060 Laptop GPU, Windows-11-10.0.26200-SP0.
