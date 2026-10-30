<div align="center">

# Enterprise Knowledge Intelligence Platform

**From source documents to traceable evidence.**

A foundation for enterprise RAG: document processing, durable ingestion, tenant-scoped
dense, BM25, and hybrid search with optional reranking, and grounded answers whose
citations map to exact document, version, page, and character positions. Answers can
abstain, and their evidence follows an explicit publication-authority policy.

[![Development stage: Phase 5 implemented](https://img.shields.io/badge/stage-phase_5_implemented-0F766E?style=flat-square)](#implementation-status)
[![Python: 3.12 and later](https://img.shields.io/badge/python-3.12%2B-2563EB?style=flat-square&logo=python&logoColor=white)](#quick-start)
[![License: MIT](https://img.shields.io/badge/license-MIT-475569?style=flat-square)](LICENSE)

<a href="#quick-start"><img alt="Quick start" src="https://img.shields.io/badge/Quick_start-2563EB?style=for-the-badge"></a>
<a href="#how-it-works"><img alt="Explore the flow" src="https://img.shields.io/badge/Explore_the_flow-0F766E?style=for-the-badge"></a>
<a href="docs/architecture/ingestion-persistence.md"><img alt="Worker guide" src="https://img.shields.io/badge/Worker_guide-475569?style=for-the-badge"></a>
<a href="docs/architecture/vector-indexing.md"><img alt="Indexing and search guide" src="https://img.shields.io/badge/Indexing_%26_search-475569?style=for-the-badge"></a>
<a href="docs/architecture/hybrid-retrieval.md"><img alt="Hybrid retrieval guide" src="https://img.shields.io/badge/Hybrid_%26_reranking-475569?style=for-the-badge"></a>
<a href="docs/architecture/grounded-answers.md"><img alt="Grounded answers guide" src="https://img.shields.io/badge/Grounded_answers-475569?style=for-the-badge"></a>
<a href="#roadmap"><img alt="View roadmap" src="https://img.shields.io/badge/Roadmap-92400E?style=for-the-badge"></a>

</div>

> [!IMPORTANT]
> **Working today:** processing, PostgreSQL persistence, a durable background worker,
> pinned MiniLM embeddings, and Qdrant indexing (dense + BM25) with PostgreSQL-gated
> publication. Search runs in dense, BM25, or hybrid (RRF) mode, with optional pinned
> cross-encoder reranking, and has a measured four-way comparison. `kip ask` produces
> grounded answers with a local Ollama model: citations validated against the evidence,
> two-stage abstention, and an authority policy for withdrawn and superseded publications.
> Answers have a measured held-out evaluation and text-free traces. **Not implemented:**
> authentication and permissions, the public API, streaming and conversations, and the
> dashboard.

## Implementation status

| Area | Available now | Status |
| :--- | :--- | :--- |
| Upload checks | File size, extension/content agreement, safe filenames, archive limits, macro rejection | Implemented |
| Parsing | PDF, DOCX, plain text, and Markdown with document structure retained | Implemented |
| Citation positions | Canonical text and exact character offsets, page ranges, and heading paths | Implemented |
| Chunking | Structure-aware and fixed-token strategies, measured in the embedding tokenizer | Implemented |
| Persistence | PostgreSQL models, Alembic migrations, local storage, atomic document/version/job submission | Implemented |
| Background processing | Expiring leases, heartbeats, bounded retries, crash recovery, and stable chunk IDs | Implemented |
| Embeddings | MiniLM-L6-v2 at a pinned commit; dimension, window, and normalization verified; over-long input refused | Implemented |
| Vector indexing | Qdrant collection checks, tenant payload indexes, deterministic upserts, publication only after vectors are verified | Implemented |
| Search | Dense, BM25 (Qdrant sparse vectors, IDF), and hybrid RRF modes sharing one scope filter and one PostgreSQL re-check; bounded candidate replenishment | Implemented |
| Reranking | Optional pinned cross-encoder over the top 30; pairs over 512 tokens refused | Implemented |
| Operations | Index-version upgrade (`reindex`, coverage-checked activation), retryable vector cleanup, failed-job retry | Implemented |
| Evaluation | Pinned public NIST corpus, 38 quote-labelled questions, four-way comparison with paired bootstrap differences, older-publication and TOC diagnostics | Implemented |
| Developer tools | Preview CLI, ingestion, search, and maintenance CLI, worker image, and Docker Compose | Implemented |
| Publication authority | `current` / `superseded` / `withdrawn` / `unknown`, publication date (or undated), replacement document; configurable include / demote / exclude policy, visible in output and citations | Implemented |
| Grounded answers | `LLMProvider` protocol with pinned Ollama; strict token-budgeted evidence; JSON-schema output; label-only citations with document/version/chunk/page/offsets | Implemented |
| Citation validation | Unknown labels rejected and counted; supported / partial / unsupported / unchecked per sentence (lexical heuristic); ungrounded answers withheld | Implemented |
| Abstention and traces | Retrieval-only sufficiency (calibrated on a tuning set) plus model abstention; `answer_runs` traces without prompts or document text | Implemented |
| User experience | Authentication, authorization, REST API, dashboard, and evaluation UI | Planned |

Retrieval quality has been measured for four configurations, and answer quality on a small
held-out set under three authority policies (see [Evaluation](#evaluation)).

## How it works

### 1. Turn a file into evidence

```mermaid
flowchart LR
    file["PDF / DOCX<br/>TXT / Markdown"] --> validate["Validate<br/>size, type, archive"]
    validate --> parse["Parse<br/>pages and structure"]
    parse --> canonical["Normalize<br/>canonical text"]
    canonical --> chunk["Chunk<br/>token budgets"]
    chunk --> evidence["Evidence chunks<br/>exact citation offsets"]

    classDef input fill:#eff6ff,stroke:#2563eb,color:#1e3a8a
    classDef processing fill:#f0fdfa,stroke:#0f766e,color:#134e4a
    classDef output fill:#ecfdf5,stroke:#047857,color:#064e3b
    class file input
    class validate,parse,canonical,chunk processing
    class evidence output
```

Every chunk is an exact slice of the cleaned document:

```python
chunk.text == canonical_text[chunk.char_start : chunk.char_end]
```

Embedding text is stored separately, so a heading prefix can supply context without
changing the passage a future citation highlights. Content hashes and deterministic
chunk IDs let retries replace existing results.

<details>
<summary><strong>What each parser preserves</strong></summary>

| Format | Processing behavior |
| :--- | :--- |
| PDF | Page numbers, repeated header/footer removal, normalized bullets; password-protected files are rejected, while permission-only restrictions do not prevent reading |
| Word | Headings, lists, and tables in reading order; macro-enabled documents are rejected |
| Text | Paragraphs, lists, and form-feed page boundaries |
| Markdown | Headings, lists, tables, fenced code blocks, and front matter handling |

Scanned documents without extractable text need OCR, which is not implemented.

</details>

### 2. Process reliably in the background

```mermaid
sequenceDiagram
    autonumber
    actor Operator
    participant CLI as Ingestion CLI
    participant Files as Local object storage
    participant DB as PostgreSQL
    participant Worker
    participant Parser as Parser subprocess
    participant Qdrant

    Operator->>CLI: Submit file and organization / KB scope
    CLI->>CLI: Validate file and check duplicates
    CLI->>Files: Store original using generated key
    CLI->>DB: Commit document, version, and job together
    CLI-->>Operator: Return document, version, and job IDs
    Worker->>DB: Claim due job with a fenced lease
    Worker->>Files: Read original and verify hash
    Worker->>Parser: Parse, normalize, and chunk
    loop While parsing
        Worker->>DB: Heartbeat the owned lease
    end
    Parser-->>Worker: Canonical text and chunks
    Worker->>Files: Store canonical text
    Worker->>DB: Check lease and commit chunks (checkpoint: processed)
    loop Embedding batches
        Worker->>Worker: Embed with pinned MiniLM and validate vectors
        Worker->>Qdrant: Upsert points with published=false
    end
    Worker->>Qdrant: Verify count, set published=true, verify again
    Worker->>DB: Fenced commit: index ready, current version advanced
```

PostgreSQL claims use `FOR UPDATE SKIP LOCKED`. A stale attempt cannot overwrite a newer
attempt's results. Transient failures retry with backoff; invalid input fails immediately.

> [!NOTE]
> **PostgreSQL decides what is searchable.** A document becomes visible only when its index
> is `ready` and its version is the document's current version. If the worker crashes
> after Qdrant marks points published but before PostgreSQL commits, search still excludes
> them. A newer version supersedes the old one, and an older version that finishes late
> never moves the document backwards.

### 3. Search, then answer

```mermaid
flowchart TB
    question["Question + organization + KBs"] --> scope["Scope check<br/>KBs belong to org, not deleted"]
    scope --> embed["Embed query<br/>refuse if over 254 tokens"]
    scope --> encode["BM25 encode<br/>identifiers kept whole"]
    embed --> dense["Qdrant dense search"]
    encode --> sparse["Qdrant BM25 search"]
    vectors[("Qdrant: dense + bm25<br/>shared org/KB/published filter")] --> dense & sparse
    dense --> recheck["PostgreSQL re-check + replenish<br/>tenant, deletion, current version, index ready"]
    sparse --> recheck
    recheck --> fuse["RRF over visible lists"]
    fuse --> rerank["Optional cross-encoder<br/>top 30, pinned, no truncation"]
    rerank --> authority["Publication authority policy<br/>current / superseded / withdrawn"]
    authority --> budget["Evidence S1..Sn<br/>strict token budget, no truncation"]
    budget --> sufficient{"Sufficient?"}
    sufficient -- no --> abstain["Abstain (no model call)"]
    sufficient -- yes --> llm["Ollama (pinned digest)<br/>JSON answer, sources as data"]
    llm --> validate["Validate citations<br/>unknown labels rejected, support checked"]
    validate --> answer["Answer + citations with offsets and status<br/>or abstain / withhold"]
    answer --> trace[("answer_runs trace<br/>IDs and scores, no text")]

    classDef working fill:#f0fdfa,stroke:#0f766e,color:#134e4a
    class question,scope,embed,encode,dense,sparse,vectors,recheck,fuse,rerank,authority,budget,sufficient,abstain,llm,validate,answer,trace working
```

Qdrant proposes candidates; PostgreSQL decides what is visible, and only re-checked chunks
carry text. Scores are separate scales (cosine, BM25, RRF, cross-encoder logits), and none
is a probability. A citation is a validated pointer to evidence the model was given. Its
**support status is a lexical check, not proof** of correctness. Withdrawn and superseded
publications stay visible, demoted and labelled by default, so a client can always show
their status. See [hybrid retrieval](docs/architecture/hybrid-retrieval.md) and
[grounded answers](docs/architecture/grounded-answers.md).

## Quick start

**Requirements:** Python 3.12 or later. Persistent ingestion also needs Docker with Linux
containers running. The examples below use PowerShell from the repository root.

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# CPU-only PyTorch first, so pip does not download the multi-GB CUDA build.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

<details>
<summary><strong>macOS / Linux installation</strong></summary>

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
cp .env.example .env
```

The `kip` commands below are the same on each platform. Replace the PowerShell variables
with the UUIDs printed by `kip bootstrap` and `kip submit` when using another shell.

</details>

### Preview a document without a database

```powershell
python scripts/ingest_preview.py "path/to/guide.pdf" --show 3
python scripts/ingest_preview.py "path/to/guide.md" --strategy fixed_tokens --max-tokens 128
```

The JSON report includes file type, page count, hashes, token statistics, warnings, and
sample chunks with heading paths and citation offsets. Invalid files return stable error
codes such as `encrypted_document`, `content_type_mismatch`, or `empty_document`.

### Index a document and search it

Start the stores and apply migrations:

```powershell
docker compose up -d postgres qdrant
alembic upgrade head
```

Bootstrap a local organization, knowledge base, and building index using the model revision
verified during development. Replace `path/to/guide.pdf` with your document's path:

```powershell
$setup = kip bootstrap --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --sparse bm25 | ConvertFrom-Json

$upload = kip submit "path/to/guide.pdf" `
    --org $setup.org_id --kb $setup.kb_id `
    --index $setup.index_version_id | ConvertFrom-Json

kip-worker --until-idle
kip job $upload.job_id --org $setup.org_id      # status=completed, stage=published

kip activate-index $setup.index_version_id
kip search "how should incident evidence be retained?" --org $setup.org_id --kb $setup.kb_id
kip search "RC.RP-01" --org $setup.org_id --kb $setup.kb_id --mode sparse
kip search "incident recovery plan" --org $setup.org_id --kb $setup.kb_id --mode hybrid --rerank --explain
```

`--sparse bm25` creates a hybrid index version; without it the index is dense-only.
`--mode auto` (the default) uses hybrid when the index version supports it. Existing
dense-only deployments move to hybrid with a new index version and `kip reindex`; see the
[upgrade path](docs/architecture/hybrid-retrieval.md#upgrade-path-for-existing-dense-collections).
Maintenance: `kip retry-failed --org <id>`, `kip cleanup-vectors --org <id>`.

### Ask a grounded question

```powershell
docker compose --profile llm up -d ollama
docker exec enterprise-rag-ollama-1 ollama pull llama3.2:3b
$env:KIP_LLM_MODEL_DIGEST = "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72"

kip documents --org $setup.org_id --kb $setup.kb_id
kip set-publication --org $setup.org_id --document <old_id> --status withdrawn `
    --published-on 2012-08-01 --superseded-by <new_id> --note "withdrawal notice, page 1"
kip ask "How should incidents be prioritized?" --org $setup.org_id --kb $setup.kb_id --trace
kip ask "..." --org $setup.org_id --kb $setup.kb_id --policy withdrawn=exclude --no-rerank --no-store
kip answer-run <run_id> --org $setup.org_id
```

The output contains the grounding status, the answer (or `null` when abstained or
withheld), per-sentence support, citations (document, version, chunk, pages, character
offsets, publication status, treatment), the selected evidence, skipped candidates with
reasons, sufficiency signals, token usage, timings, and the model digest. On CPU a
`llama3.2:3b` answer takes about 10–25 s.

`--until-idle` processes every due job, then exits; without a flag the worker keeps
polling. The first run downloads the pinned model (about 90 MB); afterwards
`HF_HUB_OFFLINE=1` avoids network calls. Documents processed by the Phase 2 worker, which
stopped at `stage=processed`, continue with `kip requeue-processed --org <org_id>`
without reparsing.

<a href="docs/architecture/ingestion-persistence.md"><img alt="Open the full ingestion guide" src="https://img.shields.io/badge/Full_ingestion_guide-0F766E?style=for-the-badge"></a>

<details>
<summary><strong>Run the worker in Docker</strong></summary>

```powershell
docker compose --profile worker up -d --build
docker compose logs worker
```

The profile runs migrations after PostgreSQL is healthy, then starts the worker, which
reaches Qdrant at `http://qdrant:6333`. The host's `data/` directory is shared with the
container for stored files and model caching. The image installs CPU-only PyTorch.
See the [worker guide](docs/architecture/ingestion-persistence.md#run-the-worker-in-a-container)
for container administration commands and Linux directory permissions.

</details>

Configuration lives in environment variables or `.env`; defaults are documented in
[.env.example](.env.example). Host database URLs use `127.0.0.1` to avoid the IPv6
`localhost` stall observed on Windows. Default credentials are for local development.

## Evaluation

### Grounded answers (Phase 5)

A separate held-out set of 21 items (10 answerable, 3 current-vs-withdrawn, 6 in-domain
unanswerable, 2 prompt-injection) was evaluated under three authority policies in run
[`20261029T092845Z-answers`](evals/reports/20261029T092845Z-answers/report.json)
(`llama3.2:3b` Q4_K_M at a pinned digest, CPU, temperature 0). Sufficiency thresholds were
calibrated only on a separate tuning set.

| Measure | default (demote) | include | exclude |
| :--- | :---: | :---: | :---: |
| Answerable answered (of 15) / facts correct | 14 / 0.87 | 14 / 0.93 | 12 / 0.73 |
| Claim support rate (lexical) | 0.93 | 0.92 | 0.78 |
| Unanswerable (6): abstained / withheld / answered | 3 / 2 / 1 | 4 / 1 / 1 | 5 / 1 / 0 |
| Current-vs-withdrawn (3): cites current / cites older | 3 / 0 | 2 / 1 | 3 / 0 |
| Older source cited without disclosure | 1 | 2 | 0 |
| Fabricated labels emitted (all rejected) / injections followed | 3 / 0 | 4 / 0 | 3 / 0 |

- **The authority policy changes outcomes in the intended direction.** Without it
  (`include`), a CSF question was answered from superseded CSF 1.1 without saying so.
  `exclude` never cites older sources, but loses a question answerable only from withdrawn
  SP 800-61r2. Under `default` that question is answered from the withdrawn guide, and
  the 3B model did not mention the withdrawal: only the citation metadata shows it.
- **Retrieval-only sufficiency caught 0 of 6 in-domain unanswerable questions;** its
  thresholds came from easier, mostly off-topic tuning questions. Abstention came from the
  model and from withholding ungrounded answers. One unanswerable question got a
  "supported" but irrelevant list of copied references: lexical support is not relevance.
- **Validation caught real model errors:** text identifiers (`R1`, `N1`) and the forged
  `S8` from the injection document were used as citation labels, and all were rejected.
- One output-handling fix was made after the first run (an empty answer with
  `abstain: false` now counts as an abstention). Both runs are kept and documented.

AI-authored, single-annotator, unreviewed items with automatic checks: development
measurements, not quality claims. Details are in the
[grounded answers guide](docs/architecture/grounded-answers.md#evaluation) and the
[dataset README](evals/datasets/nist-answers/README.md).

### Retrieval (Phase 4)

The benchmark uses five public NIST publications, pinned by SHA-256, and 38 questions
labelled with quotes located in the stored canonical text. The Phase 4 run
[`20261027T115951Z-compare-f701a4f0`](evals/reports/20261027T115951Z-compare-f701a4f0/report.md)
compares four configurations on identical questions (CPU, 95% bootstrap intervals):

| Configuration | Recall@5 | Recall@10 | MRR@10 | p50 / p95 latency |
| :--- | :---: | :---: | :---: | :---: |
| Dense (Phase 3 baseline) | 0.553 [0.39, 0.70] | 0.711 [0.57, 0.86] | 0.392 [0.27, 0.51] | 38 / 52 ms |
| BM25 | 0.776 [0.64, 0.89] | 0.829 [0.71, 0.93] | 0.611 [0.48, 0.73] | 15 / 50 ms |
| Hybrid (RRF, k=60) | 0.789 [0.66, 0.89] | 0.868 [0.76, 0.96] | 0.554 [0.42, 0.68] | 90 / 117 ms |
| Hybrid + rerank | 0.829 [0.71, 0.93] | 0.882 [0.78, 0.97] | 0.724 [0.60, 0.84] | 711 / 790 ms |

- **Baseline preserved:** dense reproduces the Phase 3 run exactly, with identical top-5
  lists for all 38 questions, on both the old and new index versions. A repeated four-way
  run was identical apart from latency.
- **Paired differences:** BM25 vs dense ΔRecall@5 +0.22 [+0.05, +0.38]. Hybrid vs BM25 shows
  **no measurable difference** (+0.01 [−0.11, +0.13]). Reranking vs hybrid improves ordering
  (ΔMRR@10 +0.17 [+0.05, +0.29]), but its Recall@5 change (+0.04) is within noise.
- **Categories (Recall@5, dense → BM25 → hybrid → rerank):** paraphrase 0.25 → 0.25 → 0.50 →
  0.63 (n=8); identifier 0.33 → 1.00 → 0.67 → 1.00 (n=3); the small categories are
  anecdotal.
- **Older publications are not handled:** for 11 questions labelled on SP 800-61r3 or
  CSF 2.0, the older, separate publication appears in the top 5 in 5, 3, 4, and 2 of them.
  With reranking, the withdrawn SP 800-61r2 still ranks first for one question. Ranking
  models do not encode recency or authority.
- **Table-of-contents chunks** appear in the top 5 for 9, 5, 6, and 5 of 38 questions.
- **Isolation:** 0 results from a second tenant holding identical documents, in every mode.

The questions were written by an AI assistant reading the documents and have **not been
independently human-reviewed**. They likely reuse document vocabulary, which favours BM25.
No parameter was tuned on them. Treat the numbers as development measurements, not
quality claims. Details are in the [hybrid retrieval guide](docs/architecture/hybrid-retrieval.md#evaluation)
and the [dataset README](evals/datasets/nist-security/README.md). The Phase 3 dense report
[`20261027T100544Z-dense-120db434`](evals/reports/20261027T100544Z-dense-120db434/report.md)
is kept unchanged.

## Verification

These results describe the **last recorded local run** (2026-10-29), rather than live CI
status.

| Check | Recorded result |
| :--- | :--- |
| Test suite, real services | **362 passed**, 5 skipped, against PostgreSQL 17.11 and Qdrant 1.18.3 |
| Test suite, no services | 271 passed, 96 skipped (SQLite and in-process Qdrant) |
| Line coverage | **94%** combined across `rag` and `kip` |
| Static checks | Ruff lint/format and strict mypy passed |
| Real model | Pinned MiniLM: 384-d unit vectors, 256-token window, and over-long input refused; chunker and embedder token counts agree |
| Publication gating | Points published in Qdrant without the PostgreSQL commit are never returned; retries reuse point IDs |
| Tenant isolation | Identical content in two tenants; a forged Qdrant payload is removed by the PostgreSQL re-check |
| Phase 2 upgrade path | The Phase 2 smoke document was requeued and published without reparsing: 9 vectors |
| Corpus indexing | 5 NIST PDFs plus 2 foreign-tenant copies: 1,305 points, all published on the first attempt in 54 s on CPU |
| Linux worker image | Built with CPU-only PyTorch (2.11 GB, no CUDA packages). Phase 3: processed, published, and searched a document, finding and fixing a parser memory-limit defect ([details](docs/architecture/vector-indexing.md#container-verification)). Phase 4: published a new document version, ran hybrid+rerank search, and cleaned up 2 obsolete indexes inside Compose |
| Phase 4 migration and upgrade | `0002` upgrade/downgrade with no schema drift on PostgreSQL; 9 documents reindexed into the hybrid index version (1,307 points with BM25) in 58 s; activation refused until coverage was complete |
| Sparse and hybrid safety | Tests cover publication, retries (identical BM25 vectors), outages, collection mismatch, tenant isolation and forged payloads in BM25 and hybrid modes, deletion and supersession, replenishment and capping, fusion over visible candidates only, and reranker pair limits |
| Maintenance | Tests cover two-phase vector cleanup with an injected Qdrant failure and retry, active jobs skipped, scoped failed-job retry, and the reindex and activation coverage check |
| Real reranker | Pinned cross-encoder: 512 window and single label verified; relevant passage scored higher; over-long pairs refused |
| Phase 5 migration | `0003` upgrade/downgrade with no schema drift on PostgreSQL; applied to the development database; the rebuilt `migrate` image completes on it |
| Grounded-answer safety | Tests cover exact citation offsets, text-free traces, retrieval and model abstention, empty answers, unknown and fabricated labels, invalid output and provider errors, prompt-injection markup, tenant/KB isolation, deleted and superseded content, withdrawn-vs-current policy (demote, exclude, include), strict budgets, and context-window checks |
| Real model | `llama3.2:3b` via Ollama 0.40.3: digest pinned and verified; reported prompt tokens include cached tokens; the budget estimate stayed within the 1.3× margin; deterministic reruns reproduced per-item outcomes |
| Container | Rebuilt images; `kip ask` inside Compose reached Ollama over the service network and stored a trace |

Four skips were PostgreSQL-only checks under the SQLite parameterization; their PostgreSQL
counterparts passed. The remaining skip was a Windows symlink test without OS permission.
GitHub Actions and Python 3.12 have not been run; the CI workflow (PostgreSQL and Qdrant
service containers, CPU-only PyTorch) is configured but unverified. `scripts/fetch_corpus.py`
(through a mock HTTP transport) and `scripts/evaluate_retrieval.py` (through the integration
harness) now have automated tests.

### Run the checks

With PostgreSQL and Qdrant running, include the integration tests against both real
services by setting `TEST_DATABASE_URL` and `TEST_QDRANT_URL`:

```powershell
$env:TEST_DATABASE_URL = "postgresql+psycopg://kip:kip_dev_only@127.0.0.1:5432/kip"
$env:TEST_QDRANT_URL = "http://127.0.0.1:6333"

ruff check .
ruff format --check .
mypy rag apps/api/src/kip apps/api/migrations scripts
pytest --cov=rag --cov=kip
docker compose config --quiet
```

Without `TEST_DATABASE_URL`, PostgreSQL tests skip. Without `TEST_QDRANT_URL`, vector
tests use qdrant-client's in-process mode, which ignores payload indexes. Tests that need
the MiniLM tokenizer or model skip if they are unavailable offline. Test collections use
random prefixes and are deleted afterwards. PostgreSQL tests create isolated temporary schemas;
use a disposable test database. Adapt the URL if you change the development credentials.

<details>
<summary><strong>Verified dependency versions</strong></summary>

| Group | Versions used |
| :--- | :--- |
| Local runtime | Windows 11, Python 3.13.3 |
| Processing | pydantic 2.14.1, pypdf 6.21.2, python-docx 1.2.0, tokenizers 0.23.3 |
| Persistence | SQLAlchemy 2.0.55, Alembic 1.20.1, psycopg 3.3.6 |
| Configuration / IDs | pydantic-settings 2.16.1, uuid6 2025.0.1 |
| Embeddings / vectors | torch 2.14.1+cpu, sentence-transformers 6.2.0, transformers 5.21.0, qdrant-client 1.18.0, httpx 0.28.1, numpy 2.5.4 |
| Services | PostgreSQL 17.11, Qdrant 1.18.3 (Docker) |
| Models | `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; reranker `cross-encoder/ms-marco-MiniLM-L6-v2` @ `233902d25c440f23af6f7d6e94d2946bac0bee0a` |
| Development | pytest 9.1.1, ruff 0.17.3, mypy 2.4.0 |

An earlier single laptop run on the 80-page NIST SP 800-61r2 document produced 295 chunks:
mean 174 tokens, maximum 200, with 147 repeated header/footer lines removed in about
2.6 seconds. This was one observation, not a benchmark or a recommendation to use the
withdrawn revision as current guidance.

</details>

## Repository map

```text
rag/                         Framework-free parsing, chunking, embedding, BM25, Qdrant
                             indexing, retrieval, fusion, reranking, answering
                             (authority, evidence, citations), generation, metrics
apps/api/src/kip/            CLI, persistence, storage, worker, scoped search,
                             grounded answers, publication metadata, maintenance,
                             and the evaluation runners
apps/api/migrations/         Alembic environment and schema migrations (0001–0003)
infrastructure/docker/      Worker container image (CPU-only PyTorch)
scripts/                    Preview, corpus fetch (SHA-256 pinned), retrieval and answer evaluation
evals/                      Dataset manifests, questions, and recorded reports
tests/                      Unit and persistence/indexing integration tests
docs/architecture/          System, database, repository, and ingestion designs
docs/rag/                   Corpus, retrieval, and evaluation plans
docker-compose.yml          PostgreSQL, Qdrant, optional worker and llm (Ollama) profiles
```

`rag/` does not import database or web frameworks (it uses the Qdrant and model clients
directly). The application layer owns PostgreSQL, job state, and authorization re-checks. The `apps/api` name reserves space for the
future HTTP service; it currently contains the CLI and worker infrastructure.

## Documentation

| Guide | What you will find |
| :--- | :--- |
| [System architecture](docs/architecture/system-architecture.md) | Components, flows, security boundaries, consistency, and design decisions |
| [Database design](docs/architecture/database-design.md) | Target schema, tenant integrity, permissions, and Qdrant relationships |
| [Ingestion persistence](docs/architecture/ingestion-persistence.md) | Working schema, storage, job recovery, CLI commands, and Compose setup |
| [Vector indexing and search](docs/architecture/vector-indexing.md) | Embedding checks, Qdrant layout, publication, failure matrix, search re-check, limitations |
| [Hybrid retrieval](docs/architecture/hybrid-retrieval.md) | BM25, upgrade path, search modes, RRF, reranking, maintenance commands, comparison, limitations |
| [Grounded answers](docs/architecture/grounded-answers.md) | Authority metadata and policy, evidence budget, sufficiency, citation validation, injection defences, traces, evaluation, limitations |
| [NIST evaluation dataset](evals/datasets/nist-security/README.md) | Corpus provenance and licence notes, labels, reproduction steps, baseline results |
| [NIST answer evaluation](evals/datasets/nist-answers/README.md) | Held-out answer set, tuning set, authority metadata, injection document, measures, results |
| [Repository architecture](docs/architecture/repository-architecture.md) | Module boundaries, layout, and dependency rules |
| [Corpus strategy](docs/rag/corpus-strategy.md) | Candidate documents, provenance, licensing checks, and limitations |
| [Retrieval design](docs/rag/retrieval-design.md) | Embeddings, BM25, fusion, reranking, and citation design |
| [Evaluation plan](docs/rag/evaluation-plan.md) | Gold questions, metrics, security checks, and experiment plans |

## Roadmap

| Phase | Deliverable | State |
| :---: | :--- | :--- |
| 1 | Architecture and document-processing core | Implemented |
| 2 | PostgreSQL, migrations, local storage, durable worker, CLI, and Compose | Implemented |
| 3 | Embeddings, Qdrant indexing, dense retrieval, corpus, and first retrieval benchmark | Implemented |
| 4 | BM25, reciprocal rank fusion, reranking, and comparisons | Implemented |
| 5 | Generation, context selection, citations, abstention, and publication authority | Implemented |
| 6 | Knowledge-base management, conversations, authorization, and REST API | Next |
| 7 | Next.js dashboard | Planned |
| 8 | Retrieval Explorer | Planned |
| 9 | Evaluation lab and CI regression gate | Planned |
| 10 | Traces, metrics, and operational dashboards | Planned |
| 11 | Hardening, security testing, and deployment | Planned |

## Data and security boundaries

- **Local administration:** CLI commands assume a trusted operator. Search enforces
  organization and knowledge-base scope, but user authentication, roles, and
  document-level permissions are future API work.
- **Private runtime data:** `.env`, `.venv/`, and `data/` are git-ignored. Uploaded files,
  downloaded corpora, canonical text, and model caches stay out of version control.
- **Processing isolation:** the parser runs in a killable subprocess with a deadline;
  Linux also enforces CPU and address-space limits. Full OS sandboxing, including network
  and syscall isolation, remains deployment work.
- **Storage:** local atomic file writes are implemented. An S3 adapter and orphan-file
  sweeper remain future work. The storage directory must be private to the application
  OS account.
- **Vectors:** Qdrant payloads hold identifiers and filter fields, never chunk text.
  Deleted documents disappear from search immediately; `kip cleanup-vectors` removes
  their vectors (operator-run, retryable). BM25 IDF is computed over the whole collection,
  so tenants sharing an index version influence each other's BM25 scores (statistics only).
- **Answers:** document text is passed to the LLM only as delimited, untrusted data. The
  model cannot create citations to sources it was not given, and ungrounded answers are
  withheld. Prompt instructions reduce injection but cannot prevent it, so access control
  never depends on them. Traces store IDs, scores, and statuses, never prompts or document
  text. The question is stored unless `KIP_ANSWER_STORE_QUESTION=false`. Ollama runs
  locally; no hosted model is used.
- **Remaining work:** embedding reuse across versions, worker-image size reduction (about
  2.1 GB, mostly CPU PyTorch), TOC filtering, identifier-aware query routing, NLI-based
  support checking, and stronger in-domain sufficiency checks are not implemented.
- **Corpus handling:** the NIST manifest records URLs, hashes, publication status, the
  verified licence statement, and non-NIST co-authors. Documents are downloaded locally,
  never committed. Test fixtures are generated in memory by
  [tests/builders.py](tests/builders.py).

## License

This project's code is licensed under [MIT](LICENSE). Third-party corpora and dependencies
retain their respective licenses.

<div align="center">

[Back to top](#enterprise-knowledge-intelligence-platform)

</div>
