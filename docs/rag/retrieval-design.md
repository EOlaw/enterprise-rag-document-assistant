# Retrieval Design

Status legend: ✅ implemented and tested · 🧪 designed, to be implemented and benchmarked.

## 1. Parsing and normalization ✅

* One **canonical text** per document version; every block, chunk and citation is an
  offset into it. Stored in object storage next to the original file so offsets remain
  verifiable after parser upgrades (`PARSER_VERSION` is part of the index fingerprint).
* NFKC normalization, invisible/control-character removal, Private-Use-Area bullet glyphs
  mapped to `•` (observed as U+F03C in NIST PDFs).
* Line-end hyphens are **kept** (`AC-\n2` → `AC-2`, `self-\nservice` → `self-service`).
  Cost: true hyphenation (`manage-\nment`) stays split. Chosen because corrupting
  identifiers is worse than a small lexical-recall loss.
* PDF running headers/footers removed by digit-masked repetition in page margins.
  Known false positive: body lines on very short pages.
* Markdown: a small line-based parser (headings, lists, tables, fenced code, front-matter
  title). `__x__` emphasis is *not* stripped because it collides with identifiers.

## 2. Chunking ✅ (parameters 🧪 to be tuned)

| Strategy | Description | Purpose |
|---|---|---|
| `structure_aware` (default) | Sections from heading hierarchy; pack whole blocks, fall back to sentences, then token windows; never crosses a section (optionally page) boundary; sentence-granular overlap | Candidate production strategy |
| `fixed_tokens` | Sliding token windows over the canonical text | Baseline for experiments |

* **Budget unit = embedding-model word pieces.** `all-MiniLM-L6-v2` has
  `max_seq_length=256` (254 content tokens). Its shipped `tokenizer.json` pads/truncates
  to 128, so `HuggingFaceTokenCounter` disables both. A test asserts this, because
  otherwise every text counts as 128 tokens.
* Defaults: `max_tokens=200`, `overlap_tokens=32`, heading prefix ≤ 32 tokens. Rationale:
  the model's hard limit minus headroom for the heading prefix. **Not yet tuned**: the
  chunk-size sweep (128/200/254 × overlap 0/32/64 × page boundaries on/off) selects
  final values.
* `embedding_text = "H1 > H2\n\n" + chunk_text`. Contextual heading prefix for both dense
  and BM25; the citation span covers `chunk_text` only.
* Observed on NIST SP 800-61r2 (80 pages): 295 chunks, mean 174 / max 200 tokens, ~2.6 s
  parse+chunk on a laptop CPU (single run; not a benchmark).

## 3. Embeddings ✅ (Phase 3; alternatives not yet evaluated)

> Implemented as designed except for embedding reuse: unchanged chunks are currently
> re-embedded. Model properties are verified at load; see
> [vector indexing](../architecture/vector-indexing.md).

* Baseline: `sentence-transformers/all-MiniLM-L6-v2` (384-d, cosine, normalized), pinned
  by **Hugging Face revision hash**, recorded in `embedding_models`.
* Batch size tuned for CPU; embeddings cached by `(content_hash, model@revision)` so
  unchanged chunks across versions are not re-embedded.
* Candidates if evaluation justifies: `BAAI/bge-small-en-v1.5` (384-d, 512 tokens),
  `intfloat/e5-small-v2` (requires `query:`/`passage:` prefixes). Changing the model
  creates a new index version (new collection), backfilled, evaluated, then switched;
  never mixed.

## 4. Qdrant collection ✅ dense part (Phase 3)

> Implemented: one collection per index version named `kip_<fingerprint[:24]>`, the
> `dense` vector, Qdrant's default HNSW settings, and payload indexes on `org_id`
> (`is_tenant`), `kb_id`, `document_id`, `document_version_id`, `document_index_id`,
> `chunk_index` and `published`. Not yet implemented: the `bm25` sparse vector
> (Phase 4) and the `restricted`, `file_type`, `tags` and `uploaded_at` fields (they
> arrive with document permissions and filters). The target configuration is below.

One collection **per index version** (`kip_chunks__<index_fingerprint>`), multi-tenant via payload:

```jsonc
{
  "vectors": { "dense": { "size": 384, "distance": "Cosine" } },
  "sparse_vectors": { "bm25": { "modifier": "idf" } },
  "hnsw_config": { "m": 16, "ef_construct": 128 },
  "payload_indexes": {
    "org_id": { "type": "keyword", "is_tenant": true },
    "kb_id": "keyword", "document_id": "keyword", "document_version_id": "keyword",
    "document_index_id": "keyword", "published": "bool", "restricted": "bool",
    "file_type": "keyword", "tags": "keyword", "uploaded_at": "datetime"
  }
}
```

* Payload holds **filter fields and IDs only**. Chunk text lives in Postgres and is
  loaded after the authorization re-check, so a stale Qdrant payload can never leak text.
* One point = one chunk with **both** named vectors, so dense and sparse share one
  upsert and one filter definition. Same access rules for both retrievers by
  construction.
* Per-tenant HNSW (`m=0, payload_m=16`) is an option once tenants are large; not
  adopted until measured.

## 5. Sparse retrieval: BM25 inside Qdrant 🧪

No separate search engine (avoids duplicate infrastructure and duplicate ACL logic).

* Tokenizer: lowercase; emits **identifier tokens intact** (`ac-2(1)`, `800-53`,
  `err_conn_reset`, `v1.2.3`, `0x80070005`) **and** their sub-tokens (`ac`, `2`), so both
  exact and partial matches score; light stop-word list; no stemming initially (A/B later).
* Document side: BM25 term-frequency saturation `tf·(k1+1)/(tf+k1·(1−b+b·dl/avgdl))`
  with `k1=1.2, b=0.75`, `avgdl` estimated per index version at build time.
  Qdrant's `idf` modifier applies corpus IDF at query time, so IDF stays correct as
  documents are added.
* Term → index: first 4 bytes of BLAKE2b (stable across processes; Python's `hash()` is
  salted). Collision rate is reported in index stats.

## 6. Hybrid fusion and reranking 🧪

* Dense top-50 and sparse top-50 retrieved with identical filters, fused **client-side**
  with Reciprocal Rank Fusion, `score = Σ 1/(k + rank)`, `k=60` (sweep 10–100). Client-side
  so the Retrieval Explorer can show each list; raw scores of different retrievers are
  never compared directly.
* Postgres re-check of fused candidates (authorization, current version, ready index).
* Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2` (max 512 positions) over the top
  **N=30** fused candidates (N swept: 10/20/30/50 against latency). Input is
  `(query, embedding_text)`.
* Query routing: if identifiers are detected, sparse is mandatory and the identifier must
  appear verbatim in the rewritten query (a guard rejects rewrites that drop or alter
  identifiers).

## 7. Evidence sufficiency and context selection 🧪

* Reranker logits are **not probabilities**. Abstention threshold `τ` is chosen on the dev
  split to hit a target precision for answerable/unanswerable classification, and
  reported with that precision/recall.
* Greedy selection by rerank score into a token budget (default 2,000 context tokens,
  swept against the model window) with: near-duplicate removal (overlapping offsets or
  Jaccard ≥ 0.8), per-document cap, merging of adjacent chunks from the same document,
  and preservation of conflicting versions (both cited, with dates).

## 8. Citation metadata strategy ✅ (data) / 🧪 (verification)

Each chunk carries `document_id, document_version_id, chunk_index, char_start, char_end,
page_start, page_end, heading_path, content_hash` (char offsets into canonical text are
tested to slice exactly to the chunk text).

Generation sees evidence as labelled blocks `[S1]…[Sn]`; the model cites labels, never
IDs. The server maps labels → chunks, so fabricated IDs are impossible and unknown
labels are flagged. A separate **support check** (NLI/LLM judge validated against human
labels) estimates whether a cited passage actually supports the sentence citing it; a
valid reference is not treated as proof of support.
