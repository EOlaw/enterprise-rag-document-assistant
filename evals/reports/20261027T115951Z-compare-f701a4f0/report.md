# Retrieval comparison `20261027T115951Z-compare-f701a4f0`

- Dataset: `nist-security` v1, 38 questions (questions sha256 `db52c2cf75f9`, corpus manifest `61074e3ae0b9`)
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4`; index fingerprint `f701a4f08fcd5caa`; BM25 {"b": 0.75, "k1": 1.2, "kind": "bm25", "avgdl": 74.0, "tokenizer": "bm25-v1"}
- Reranker: {"model": "cross-encoder/ms-marco-MiniLM-L6-v2", "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a", "max_length": 512}
- Search settings: {"overfetch": 4, "max_candidates": 200, "hybrid_candidates": 50, "rrf_k": 60, "rerank_candidates": 30} (none tuned on this dataset)
- Environment: Python 3.13.3, Windows-11-10.0.26300-SP0, CPU only: True, Qdrant 1.18.3

| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Hit@5 |
|---|---|---|---|---|---|
| dense | 0.553 [0.39, 0.70] | 0.711 [0.57, 0.86] | 0.392 [0.27, 0.51] | 0.441 [0.32, 0.56] | 0.579 [0.42, 0.74] |
| bm25 | 0.776 [0.64, 0.89] | 0.829 [0.71, 0.93] | 0.611 [0.48, 0.73] | 0.632 [0.51, 0.75] | 0.816 [0.68, 0.92] |
| hybrid | 0.789 [0.66, 0.89] | 0.868 [0.76, 0.96] | 0.554 [0.42, 0.68] | 0.603 [0.49, 0.71] | 0.816 [0.68, 0.92] |
| hybrid+rerank | 0.829 [0.71, 0.93] | 0.882 [0.78, 0.97] | 0.724 [0.60, 0.84] | 0.727 [0.61, 0.84] | 0.842 [0.71, 0.95] |

95% percentile bootstrap intervals over questions (1,000 resamples).

Paired per-question differences (bootstrap CI over questions):

| Comparison | ΔRecall@5 | ΔRecall@10 | ΔMRR@10 | better / worse / same (R@5) |
|---|---|---|---|---|
| bm25 - dense | +0.224 [+0.05, +0.38] | +0.118 [+0.00, +0.25] | +0.220 [+0.07, +0.38] | 11 / 2 / 25 |
| hybrid - dense | +0.237 [+0.11, +0.38] | +0.158 [+0.05, +0.28] | +0.162 [+0.06, +0.26] | 10 / 1 / 27 |
| hybrid - bm25 | +0.013 [-0.11, +0.13] | +0.039 [+0.00, +0.11] | -0.058 [-0.16, +0.03] | 3 / 3 / 32 |
| hybrid+rerank - dense | +0.276 [+0.08, +0.46] | +0.171 [+0.05, +0.30] | +0.332 [+0.20, +0.46] | 13 / 3 / 22 |
| hybrid+rerank - hybrid | +0.039 [-0.08, +0.16] | +0.013 [-0.05, +0.09] | +0.170 [+0.05, +0.29] | 4 / 2 / 32 |

Recall@5 by category:

| Category | n | dense | bm25 | hybrid | hybrid+rerank |
|---|---|---|---|---|---|
| factual | 22 | 0.682 | 0.932 | 0.977 | 0.955 |
| identifier | 3 | 0.333 | 1.000 | 0.667 | 1.000 |
| multi_document | 2 | 0.500 | 0.500 | 0.250 | 0.250 |
| paraphrase | 8 | 0.250 | 0.250 | 0.500 | 0.625 |
| version_sensitive | 3 | 0.667 | 1.000 | 0.667 | 0.667 |

Older publications and table-of-contents results:

| Configuration | older pub. in top 5 | older outranks first relevant | questions with TOC in top 5 | TOC share of top-5 results |
|---|---|---|---|---|
| dense | 5 / 11 | 4 / 11 | 9 / 38 | 0.079 |
| bm25 | 3 / 11 | 2 / 11 | 5 / 38 | 0.026 |
| hybrid | 4 / 11 | 2 / 11 | 6 / 38 | 0.047 |
| hybrid+rerank | 2 / 11 | 2 / 11 | 5 / 38 | 0.026 |

| Latency p50 / p95 (ms) | total | embed | dense | sparse | rerank |
|---|---|---|---|---|---|
| dense | 43.8 / 59.16 | 9.11 / 11.14 | 29.06 / 46.67 | - | - |
| bm25 | 30.19 / 46.7 | - | - | 24.75 / 42.24 | - |
| hybrid | 90.77 / 131.61 | 9.43 / 11.66 | 42.67 / 63.3 | 31.52 / 55.27 | - |
| hybrid+rerank | 729.72 / 787.81 | 14.66 / 19.28 | 52.89 / 65.5 | 36.79 / 59.56 | 611.21 / 654.58 |

Cross-tenant check: 0 results from 2 foreign-tenant documents (checked: True). Model load seconds (excluded): {"embedder": 8.04, "reranker": 0.21}.
