# Ingestion persistence (Phase 2)

The `kip` package in `apps/api/src/` wraps the processing core with PostgreSQL, atomic
local object storage and a durable worker. These are trusted administration commands.
User accounts, permission checks and the HTTP API are not implemented yet.

## Run locally

Use Python 3.12 or 3.13 and Docker Desktop with Linux containers. From the repository root:

```bash
pip install -e ".[dev]"
# Copy .env.example to .env (PowerShell: Copy-Item .env.example .env).
docker compose up -d postgres qdrant
alembic upgrade head
kip bootstrap --revision <hugging-face-model-commit>
```

The bootstrap command prints organization, knowledge-base and index-version UUIDs. It
creates index metadata only; it does not download or embed a model. Use a commit hash
for reproducibility. A mutable revision such as `main` is acceptable for a disposable
development experiment, but can resolve to different tokenizer contents later.

```bash
kip submit path/to/guide.pdf --org <org_id> --kb <kb_id> --index <index_version_id>
kip-worker --once
kip job <job_id> --org <org_id>
# Run continuously (Ctrl-C finishes the current attempt before stopping):
kip-worker
```

The worker downloads the configured tokenizer on first use. For offline operation, warm
the Hugging Face cache first. Raw files and canonical UTF-8 text live in `data/objects/`;
all runtime data stays git-ignored. `--document <document_id>` on `submit` creates another
version of an existing document. Re-uploading the same bytes within a KB returns the
existing document/version; submitting the same version against another index enqueues
another processing job. An existing failed job is returned as failed, rather than silently
resetting its attempt budget. Operator retry/reset tooling is a later addition.

`KIP_DATABASE_URL` and `KIP_STORAGE_ROOT` are read from the environment or `.env`. The
remaining `KIP_` resource/lease settings are listed in `.env.example`. When overriding the
Postgres port or password, update the host-side database URL too; URL-encode passwords
containing reserved URL characters. Default credentials are for local development.
Host examples use `127.0.0.1` explicitly because Windows can stall when `localhost`
resolves to IPv6 while Docker publishes only an IPv4 loopback port.

## Run the worker in a container

```bash
docker compose --profile worker up -d --build
# Administration commands inside the same image and storage mount:
docker compose run --rm worker kip bootstrap --revision <model-commit>
docker compose run --rm worker kip submit /app/data/guide.pdf --org <org> --kb <kb> --index <index>
docker compose logs worker
docker compose stop
```

The worker profile first runs migrations after Postgres is healthy. Postgres and Qdrant
use named volumes, and `./data` is shared with the host for object storage and tokenizer
caching. On Linux, ensure this directory is writable by container UID 10001. The worker
runs without root privileges. Qdrant starts as an infrastructure dependency for Phase 3;
this worker does not write vectors or require Qdrant to process documents. Only localhost
ports are exposed. Container removal does not remove named database volumes unless
explicitly requested.

## State and recovery

1. Validate bytes and filename, lock the KB to serialize duplicate/new-version submission,
   write the original using generated keys, then commit document, version and job together.
   A failed commit may leave an orphan file; no unsafe automatic cleanup runs yet.
2. Claim a due job using `FOR UPDATE SKIP LOCKED` in a short transaction. Increment its
   attempt and assign a fresh random lease token and expiration. Parsing holds no DB lock.
3. Verify the stored original's size and hash. Load the tokenizer in the parent, then
   parse/chunk in a spawned child. Heartbeats extend only an unexpired owned lease.
4. Write canonical text under a content-addressed name. In one transaction, lock and
   validate the owned lease, check the document still exists, replace chunks with stable
   UUIDv5 IDs, persist processing metadata and complete the job with `stage=processed`.

After step 4, the document version is **processed**, the document index is **building**,
`published_at` is null and `documents.current_version_id` is unchanged. No version or index
is marked ready. Phase 3 must write vectors and implement the publication transaction.
Embedding text is retained separately from the exact citation text so indexing can later
reuse the heading prefix without changing citation offsets.

Transient storage/database errors retry with capped exponential backoff. Typed input
errors fail immediately. Unknown failures have bounded retries and store a generic code;
exception messages and document contents are not persisted in job errors or worker logs.
Expired processing jobs are reclaimable. A final expired attempt is failed, so repeated
worker crashes cannot retry indefinitely. A stale attempt cannot heartbeat, fail or commit
over its successor: all these transitions check the token, status and lease expiration.
Database outages can prevent recording failure; the lease then expires for later recovery.

The child has a wall-clock deadline on all platforms and CPU/address-space limits on
Linux. Windows currently enforces the deadline only. **The parser subprocess is not an
OS security sandbox:** network namespaces, syscall restrictions and equivalent Windows
resource isolation remain deployment hardening work. The local storage root must be
private to the application OS account; checks reject traversal and existing filesystem
links, but cannot protect against another OS process concurrently rewriting that root.
An orphan-object sweeper and S3 adapter are not implemented in this phase.

## Schema and verification

The initial frozen Alembic migration implements organizations, knowledge bases, documents,
versions, embedding-model/index metadata, document indexes, chunks and ingestion jobs.
Composite FKs enforce tenant and document ancestry, including the current-version pointer.
The full identity/access/conversation/evaluation design remains a roadmap, not deployed
schema. JSON lists use JSONB in Postgres; the SQLite test adapter uses JSON.

```bash
ruff check .
ruff format --check .
mypy rag apps/api/src/kip apps/api/migrations scripts
pytest --cov=rag --cov=kip
docker compose config --quiet
```

Set `TEST_DATABASE_URL` to a disposable PostgreSQL database to run the same persistence
tests against real Postgres, including migration upgrade/downgrade and schema drift,
concurrent duplicate uploads, and independent job claims while rows are locked. Tests
create and remove randomly named schemas; they do not truncate application tables. The
test database user needs permission to create schemas. For example in PowerShell:

```powershell
$env:TEST_DATABASE_URL = 'postgresql+psycopg://kip:kip_dev_only@127.0.0.1:5432/kip'
pytest --cov=rag --cov=kip
```

Without this variable, Postgres tests skip explicitly; SQLite does not verify Postgres
locking semantics or production migrations. CI supplies PostgreSQL on both Python 3.12
and 3.13. Linux CI also exercises the parser's resource limits. Local execution does not
establish that the GitHub workflow has run.

Implementation references: [SQLAlchemy locking](https://docs.sqlalchemy.org/en/20/core/selectable.html#sqlalchemy.sql.expression.Select.with_for_update),
[Alembic environments](https://alembic.sqlalchemy.org/en/latest/tutorial.html), and
[Compose startup dependencies](https://docs.docker.com/compose/how-tos/startup-order/).
Development service releases: [PostgreSQL 17.11](https://www.postgresql.org/docs/release/17.11/)
and [Qdrant 1.18.3](https://github.com/qdrant/qdrant/releases/tag/v1.18.3).
