# Dense retrieval evaluation `20261027T100544Z-dense-120db434`

- Retriever: dense (cosine, Qdrant) + PostgreSQL re-check; no reranking
- Dataset: `nist-security` v1, 38 questions (questions sha256 `db52c2cf75f9`, corpus manifest `61074e3ae0b9`)
- Model: `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4`, 384-d, max 256 tokens
- Index fingerprint `120db4341bb8548c`, parser v1, chunking {"strategy": "structure_aware", "max_tokens": 200, "heading_aware": true, "overlap_tokens": 32, "page_boundaries": false, "include_heading_prefix": true, "max_heading_prefix_tokens": 32}
- Environment: Python 3.13.3, Windows-11-10.0.26300-SP0, CPU only: True, Qdrant 1.18.3

| Metric | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| recall | 0.250 [0.12, 0.39] | 0.421 [0.26, 0.58] | 0.553 [0.39, 0.70] | 0.711 [0.57, 0.86] |
| precision | 0.263 [0.13, 0.42] | 0.149 [0.10, 0.21] | 0.126 [0.09, 0.16] | 0.084 [0.07, 0.11] |
| hit | 0.263 [0.13, 0.42] | 0.421 [0.26, 0.58] | 0.579 [0.42, 0.74] | 0.737 [0.58, 0.87] |
| reciprocal_rank | 0.263 [0.13, 0.42] | 0.333 [0.20, 0.47] | 0.368 [0.23, 0.49] | 0.392 [0.27, 0.51] |
| ndcg | 0.263 [0.13, 0.42] | 0.336 [0.21, 0.47] | 0.386 [0.26, 0.50] | 0.441 [0.32, 0.56] |

95% percentile bootstrap intervals over questions (1,000 resamples).

| Category | n | Recall@5 | MRR@10 |
|---|---|---|---|
| factual | 22 | 0.682 | 0.472 |
| identifier | 3 | 0.333 | 0.083 |
| multi_document | 2 | 0.500 | 0.225 |
| paraphrase | 8 | 0.250 | 0.223 |
| version_sensitive | 3 | 0.667 | 0.667 |

- Version-sensitive questions with a superseded document in the top 5: 2 / 3
- Cross-tenant check: 0 results from 2 foreign-tenant documents (checked: True)

| Latency (ms) | p50 | p95 |
|---|---|---|
| setup | 5.94 | 7.34 |
| embed | 8.37 | 9.96 |
| vector_search | 23.87 | 39.62 |
| recheck | 7.64 | 9.68 |
| total | 45.95 | 61.81 |

Model load (excluded from latency): 6.38 s.
