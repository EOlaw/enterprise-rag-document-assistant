<div align="center">

# Enterprise Knowledge Intelligence Platform

**From source documents to traceable evidence.**

A foundation for enterprise RAG: document processing, durable ingestion, vector
indexing, and tenant-scoped dense search that returns exact citation positions.
Hybrid retrieval and grounded answers are on the roadmap.

[![Development stage: Phase 3 implemented](https://img.shields.io/badge/stage-phase_3_implemented-0F766E?style=flat-square)](#implementation-status)
[![Python: 3.12 and later](https://img.shields.io/badge/python-3.12%2B-2563EB?style=flat-square&logo=python&logoColor=white)](#quick-start)
[![License: MIT](https://img.shields.io/badge/license-MIT-475569?style=flat-square)](LICENSE)

<a href="#quick-start"><img alt="Quick start" src="https://img.shields.io/badge/Quick_start-2563EB?style=for-the-badge"></a>
<a href="#how-it-works"><img alt="Explore the flow" src="https://img.shields.io/badge/Explore_the_flow-0F766E?style=for-the-badge"></a>
<a href="docs/architecture/ingestion-persistence.md"><img alt="Worker guide" src="https://img.shields.io/badge/Worker_guide-475569?style=for-the-badge"></a>
<a href="docs/architecture/vector-indexing.md"><img alt="Indexing and search guide" src="https://img.shields.io/badge/Indexing_%26_search-475569?style=for-the-badge"></a>
<a href="#roadmap"><img alt="View roadmap" src="https://img.shields.io/badge/Roadmap-92400E?style=for-the-badge"></a>

</div>

> [!IMPORTANT]
> **Working today:** processing, PostgreSQL persistence, a durable background worker,
> pinned MiniLM embeddings, Qdrant indexing with PostgreSQL-gated publication, and a
> dense-search CLI with a reproducible retrieval benchmark. **Coming next:** BM25, hybrid
> fusion, and reranking. Answer generation, authentication, the public API, and the
> dashboard are designed but not implemented.

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
| Dense search | `kip search` with mandatory organization/KB scope and a PostgreSQL re-check of every candidate | Implemented |
| Evaluation | Pinned public NIST corpus, 38 quote-labelled questions, metrics with bootstrap intervals | Implemented (baseline) |
| Developer tools | Preview CLI, ingestion and search CLI, worker image, and Docker Compose | Implemented |
| Retrieval and answers | BM25, fusion, reranking, and grounded generation | Planned |
| User experience | Authentication, authorization, REST API, dashboard, and evaluation UI | Planned |

Retrieval quality has been measured for dense search only (see [Evaluation](#evaluation)).
No answer-quality metrics exist, because nothing generates answers yet.

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

Solid boxes work today through the CLI. Dashed boxes are planned.

```mermaid
flowchart TB
    question["Query + organization + KBs"] --> scope["Scope check<br/>KBs belong to org, not deleted"]
    scope --> embed["Embed query<br/>refuse if over 254 tokens"]
    embed --> dense["Qdrant dense search<br/>org, KB, published filters"]
    vectors[("Qdrant vectors")] --> dense
    dense --> recheck["PostgreSQL re-check<br/>tenant, deletion, current version, index ready"]
    recheck --> results["Chunks with exact offsets,<br/>pages, headings, scores"]
    results -.-> hybrid["BM25 + fusion + rerank"]
    hybrid -.-> generate["Grounded answer or abstain"]

    classDef working fill:#f0fdfa,stroke:#0f766e,color:#134e4a
    classDef planned fill:#fffbeb,stroke:#b45309,color:#78350f,stroke-dasharray:5 5
    class question,scope,embed,dense,vectors,recheck,results working
    class hybrid,generate planned
```

Qdrant proposes candidates; PostgreSQL decides what is returned, and only re-checked
chunks carry text. Scores are cosine similarities for one query, not probabilities.
See [vector indexing and search](docs/architecture/vector-indexing.md) for the failure and
recovery matrix.

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
$setup = kip bootstrap --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 | ConvertFrom-Json

$upload = kip submit "path/to/guide.pdf" `
    --org $setup.org_id --kb $setup.kb_id `
    --index $setup.index_version_id | ConvertFrom-Json

kip-worker --until-idle
kip job $upload.job_id --org $setup.org_id      # status=completed, stage=published

kip activate-index $setup.index_version_id
kip search "how should incident evidence be retained?" --org $setup.org_id --kb $setup.kb_id
```

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

The first retrieval benchmark uses five public NIST publications, pinned by SHA-256, and 38
questions labelled with quotes. Labels are located in the stored canonical text, so they
stay valid across chunking configurations. Run
[`20261027T100544Z-dense-120db434`](evals/reports/20261027T100544Z-dense-120db434/report.md)
measured **dense retrieval only** (MiniLM-L6-v2, 200-token structure-aware chunks, CPU):

| Metric | @1 | @5 | @10 |
| :--- | :---: | :---: | :---: |
| Recall | 0.25 [0.12, 0.39] | 0.55 [0.39, 0.70] | 0.71 [0.57, 0.86] |
| MRR | 0.26 | 0.37 | 0.39 |
| nDCG | 0.26 | 0.39 | 0.44 |

- **Weak spots:** paraphrase questions (Recall@5 0.25, n=8) and exact identifiers such as
  `RC.RP-01` (0.33, n=3). These motivate BM25 and reranking in Phase 4.
- **Superseded guidance:** in 2 of 3 version-sensitive questions with a superseded
  source, the withdrawn SP 800-61r2 or CSF 1.1 appeared in the top 5. Nothing ranks by
  recency or authority yet.
- **Isolation:** 0 results came from a second tenant holding identical copies of two
  documents.
- **Latency (p50/p95):** 46/62 ms per query on a laptop CPU, excluding the one-time
  6.4 s model load.
- **Reproducibility:** a repeated run produced identical metrics and identical top-5
  lists for every question.

Brackets are 95% bootstrap intervals; with 38 questions they are wide. The questions were
written by an AI assistant and have **not been independently human-reviewed**. Treat the
numbers as a development baseline, not a quality claim. Provenance, licence notes, and
reproduction steps are in the [dataset README](evals/datasets/nist-security/README.md).

## Verification

These results describe the **last recorded local run** (2026-10-27), rather than live CI
status.

| Check | Recorded result |
| :--- | :--- |
| Test suite, real services | **228 passed**, 5 skipped, against PostgreSQL 17.11 and Qdrant 1.18.3 |
| Test suite, no services | 181 passed, 52 skipped (SQLite and in-process Qdrant) |
| Line coverage | **94%** combined across `rag` and `kip` |
| Static checks | Ruff lint/format and strict mypy passed |
| Real model | Pinned MiniLM: 384-d unit vectors, 256-token window, and over-long input refused; chunker and embedder token counts agree |
| Publication gating | Points published in Qdrant without the PostgreSQL commit are never returned; retries reuse point IDs |
| Tenant isolation | Identical content in two tenants; a forged Qdrant payload is removed by the PostgreSQL re-check |
| Phase 2 upgrade path | The Phase 2 smoke document was requeued and published without reparsing: 9 vectors |
| Corpus indexing | 5 NIST PDFs plus 2 foreign-tenant copies: 1,305 points, all published on the first attempt in 54 s on CPU |
| Linux worker image | Built with CPU-only PyTorch (2.11 GB, no CUDA packages); processed, published, and searched a document inside Compose. This found and fixed a parser memory-limit defect; see the [indexing guide](docs/architecture/vector-indexing.md#container-verification) |

Four skips were PostgreSQL-only checks under the SQLite parameterization; their PostgreSQL
counterparts passed. The remaining skip was a Windows symlink test without OS permission.
GitHub Actions and Python 3.12 have not been run; the CI workflow now also starts a Qdrant
service container and installs CPU-only PyTorch, but that configuration is unverified.
`scripts/fetch_corpus.py` and `scripts/evaluate_retrieval.py` were exercised by the runs
above, not by unit tests.

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
| Model | `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Development | pytest 9.1.1, ruff 0.17.3, mypy 2.4.0 |

An earlier single laptop run on the 80-page NIST SP 800-61r2 document produced 295 chunks:
mean 174 tokens, maximum 200, with 147 repeated header/footer lines removed in about
2.6 seconds. This was one observation, not a benchmark or a recommendation to use the
withdrawn revision as current guidance.

</details>

## Repository map

```text
rag/                         Framework-free parsing, chunking, embedding, Qdrant
                             indexing, dense retrieval, and evaluation metrics
apps/api/src/kip/            CLI, persistence, storage, worker, and scoped search
apps/api/migrations/         Alembic environment and schema migration
infrastructure/docker/      Worker container image (CPU-only PyTorch)
scripts/                    Preview, corpus fetch (SHA-256 pinned), retrieval evaluation
evals/                      Dataset manifests, questions, and recorded reports
tests/                      Unit and persistence/indexing integration tests
docs/architecture/          System, database, repository, and ingestion designs
docs/rag/                   Corpus, retrieval, and evaluation plans
docker-compose.yml          PostgreSQL, Qdrant, and optional worker profile
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
| [NIST evaluation dataset](evals/datasets/nist-security/README.md) | Corpus provenance and licence notes, labels, reproduction steps, baseline results |
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
| 4 | BM25, reciprocal rank fusion, reranking, and comparisons | Next |
| 5 | Generation, context selection, citations, and abstention | Planned |
| 6 | Knowledge-base management, conversations, authorization, and REST API | Planned |
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
  Deleted documents disappear from search immediately, but their vectors remain until a
  cleanup job exists.
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
