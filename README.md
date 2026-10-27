# Enterprise Knowledge Intelligence Platform

A tenant-aware, citation-grounded Retrieval-Augmented Generation platform for
organizational documents: secure ingestion, hybrid dense + BM25 retrieval, cross-encoder
reranking, evidence-grounded answers with verifiable citations, and an evaluation
framework that measures retrieval and answer quality instead of asserting it.

> **Status: early development (Phase 2 of 11).** The ingestion processing core,
> persistence and background worker are implemented. The API, vector indexing, retrieval,
> generation and dashboard are designed (see [docs](#documentation)) but **not yet
> implemented**. This README
> only claims what has been run.

## Implementation status

| Capability | State |
|---|---|
| Upload validation: extension vs magic bytes, size limits, zip-bomb and macro rejection, filename sanitization | ✅ implemented, tested |
| Parsing: PDF (pages, encryption handling, running header/footer removal), DOCX (headings, lists, tables in reading order), TXT (form-feed pages), Markdown (headings, lists, tables, code, front matter) | ✅ implemented, tested |
| Canonical text with exact block/chunk offsets for citation highlighting | ✅ implemented, tested |
| Chunking: structure-aware (section → block → sentence → token window) and fixed-token baseline, budgeted in the embedding model's own word pieces | ✅ implemented, tested |
| Content hashes (raw bytes, normalized text, per chunk) and deterministic vector point IDs for idempotent re-indexing | ✅ implemented, tested |
| PostgreSQL models + Alembic, local object storage, transactional submissions, leased worker, administration CLI and Docker Compose | implemented; see persistence guide for validation |
| Qdrant indexing, BM25, hybrid fusion, reranking, generation, API, dashboard, evaluation runs | ⏳ designed, not implemented |

No retrieval-quality or answer-quality metrics exist yet, so none are reported.

## Quick start

Requires Python ≥ 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

# Quality gates (the same commands CI runs)
ruff check . && ruff format --check .
mypy rag apps/api/src/kip apps/api/migrations scripts
pytest --cov=rag --cov=kip
```

The first test run downloads the `all-MiniLM-L6-v2` tokenizer (~1 MB) from Hugging Face;
tests that need it are skipped when offline.

### Inspect how a document will be chunked

```bash
python scripts/ingest_preview.py path/to/file.pdf --show 3
python scripts/ingest_preview.py docs/*.md --strategy fixed_tokens --max-tokens 128
```

It prints a JSON report per file: detected type, title, pages, hashes, chunk count, token
statistics (in the embedding model's tokenizer), warnings and sample chunks with page
ranges, heading paths and character offsets. Invalid files are reported with a stable
error code (`encrypted_document`, `content_type_mismatch`, `empty_document`, ...).

Example (single run, laptop CPU, NIST SP 800-61r2, 80 pages, public domain):
295 chunks, mean 174 / max 200 tokens, 147 running header/footer lines removed, ~2.6 s.
This is an observation, not a benchmark.

### Store documents and run background processing

See the [ingestion persistence guide](docs/architecture/ingestion-persistence.md) for
environment setup, commands, state transitions and recovery details.

```bash
docker compose up -d postgres qdrant
alembic upgrade head
kip bootstrap --revision <hugging-face-model-commit>
kip submit path/to/guide.pdf --org <org_id> --kb <kb_id> --index <index_version_id>
kip-worker --once
kip job <job_id> --org <org_id>
```

Processed chunks are persisted with exact citation offsets. Jobs finish at
`stage=processed`; versions remain `processed` and document indexes remain `building`.
Vector publication and searchable documents arrive in Phase 3. The commands above are
trusted local administration tools; authentication and the public API arrive later.
The worker uses a killable parser subprocess with a deadline (plus CPU/memory limits
on Linux); full OS sandboxing remains deployment hardening work.

## Documentation

| Document | Contents |
|---|---|
| [System architecture](docs/architecture/system-architecture.md) | Components, ingestion and query lifecycles, security boundaries, consistency policy, ADRs |
| [Database design](docs/architecture/database-design.md) | PostgreSQL schema, tenant integrity, permissions, relationship to Qdrant, failure recovery |
| [Ingestion persistence](docs/architecture/ingestion-persistence.md) | Working PostgreSQL schema, migrations, object storage, leased jobs, CLI and Compose |
| [Repository architecture](docs/architecture/repository-architecture.md) | Module layout, dependency rules, deviations from the initial proposal |
| [Corpus strategy](docs/rag/corpus-strategy.md) | Public, reusable documents for development and evaluation; licences; limitations |
| [Retrieval design](docs/rag/retrieval-design.md) | Normalization, chunking, embeddings, Qdrant config, BM25, fusion, reranking, citations |
| [Evaluation plan](docs/rag/evaluation-plan.md) | Gold dataset, metrics, statistical comparison, security invariants, experiment matrix |

## Roadmap

1. ✅ Requirements, architecture, ingestion processing core
2. Implemented: PostgreSQL models + Alembic, local object storage, Postgres-backed processing worker, Docker Compose
3. Embeddings, Qdrant indexing, dense retrieval, corpus download + first retrieval benchmark
4. BM25 sparse vectors, RRF fusion, cross-encoder reranking, comparison experiments
5. Generation (Ollama provider), context selection, citations, abstention
6. Knowledge bases, conversations, authorization, REST API
7. Next.js dashboard · 8. Retrieval Explorer · 9. Evaluation lab + CI regression gate
10. Observability (OpenTelemetry, Prometheus, Grafana) · 11. Hardening, security testing, deployment

## Tested versions

Verified locally on Windows 11, Python 3.13.3: pydantic 2.14.1, pypdf 6.21.2,
python-docx 1.2.0, tokenizers 0.23.3, pytest 9.1.1, ruff 0.17.3, mypy 2.4.0.
Python 3.12 is declared supported and is covered by the CI matrix.

Phase 2 was verified with SQLAlchemy 2.0.55, Alembic 1.20.1, psycopg 3.3.6,
pydantic-settings 2.16.1 and uuid6 2025.0.1. The local run passed **148 tests**, with
**95% combined line coverage of `rag` and `kip`**, and passed ruff and strict mypy.
Five tests skipped: four PostgreSQL-only cases in the SQLite parameterization (their
PostgreSQL counterparts passed), and a Windows symlink test lacking OS permission.
PostgreSQL migration upgrade/downgrade, model/schema drift and concurrency were tested
against the Compose database. The Linux worker image built and processed a smoke
document with the actual pinned tokenizer: 9 chunks, max 197 tokens, all offsets valid.
Qdrant's readiness endpoint responded. Python 3.12 and GitHub Actions have not been run
locally/on GitHub; the configured CI matrix includes both supported Python versions.

## Data policy

No documents are committed to this repository: neither public corpora (downloaded into
the git-ignored `data/` directory with a provenance manifest) nor any private or
enterprise content. Test fixtures are generated in memory by `tests/builders.py`.

## Licence

MIT (see [LICENSE](LICENSE)). Third-party corpora keep their own licences; see the corpus strategy.
