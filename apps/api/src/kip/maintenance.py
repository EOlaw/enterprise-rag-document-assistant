"""Operator maintenance: failed-job retry, reindexing into a new index version, index
activation coverage, and retryable cleanup of obsolete vectors.

These are trusted administration functions (no user authorization yet). Every function
is scoped to one organization except the activation check, which spans the index version.
"""

from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import and_, exists, func, select
from sqlalchemy.orm import Session, aliased, sessionmaker

from kip.db.models import (
    Document,
    DocumentIndex,
    DocumentVersion,
    EmbeddingModel,
    IndexVersion,
    IngestionJob,
    KnowledgeBase,
    utcnow,
)
from kip.ingestion import job_key
from kip.vectors import VectorIndexFactory
from rag.indexing import VectorStoreError

ACTIVE_JOB_STATUSES = ("queued", "processing", "retrying")


# ------------------------------------------------------------------ failed jobs


@dataclass
class RetryOutcome:
    retried: list[UUID] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)  # job id -> reason


def retry_failed(
    session: Session,
    org_id: UUID,
    *,
    job_id: UUID | None = None,
    error_code: str | None = None,
) -> RetryOutcome:
    """Move ``failed`` jobs back to ``queued`` with a fresh attempt budget.

    Only the transition failed -> queued is performed (row-locked), so completed, running
    or cancelled jobs are never disturbed. Jobs that cannot succeed are skipped with a
    reason: deleted document or knowledge base, or a retired index version.
    """
    query = (
        select(IngestionJob, Document, KnowledgeBase, IndexVersion)
        .join(DocumentVersion, DocumentVersion.id == IngestionJob.document_version_id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .join(KnowledgeBase, KnowledgeBase.id == Document.kb_id)
        .join(IndexVersion, IndexVersion.id == IngestionJob.index_version_id)
        .where(IngestionJob.org_id == org_id, IngestionJob.status == "failed")
        .order_by(IngestionJob.created_at, IngestionJob.id)
        .with_for_update(of=IngestionJob)
    )
    if job_id is not None:
        query = query.where(IngestionJob.id == job_id)
    if error_code is not None:
        query = query.where(IngestionJob.error_code == error_code)
    outcome = RetryOutcome()
    now = utcnow()
    for job, document, kb, index in session.execute(query).all():
        if document.deleted_at is not None or kb.deleted_at is not None:
            outcome.skipped[str(job.id)] = "document or knowledge base deleted"
            continue
        if index.status == "retired":
            outcome.skipped[str(job.id)] = "index version retired"
            continue
        job.status = "queued"
        job.stage = "queued"
        job.attempt = 0
        job.run_after = now
        job.error_code = None
        job.error_message = None
        job.finished_at = None
        job.lease_token = None
        job.lease_expires_at = None
        version = session.get(DocumentVersion, job.document_version_id)
        if version is not None and version.status == "failed":
            version.status = "pending"
        outcome.retried.append(job.id)
    return outcome


# ------------------------------------------------------------------- reindexing


def enqueue_reindex(session: Session, org_id: UUID, index_version_id: UUID) -> int:
    """Queue the current, published version of every live document into an index version.

    Idempotent: versions that already have a job for that index version are skipped. The
    worker reparses, re-embeds and (for hybrid indexes) BM25-encodes each version.
    """
    index = session.get(IndexVersion, index_version_id)
    if index is None or index.status == "retired":
        raise LookupError("usable index version not found")
    versions = session.scalars(
        select(DocumentVersion)
        .join(Document, Document.current_version_id == DocumentVersion.id)
        .join(KnowledgeBase, KnowledgeBase.id == Document.kb_id)
        .where(
            Document.org_id == org_id,
            Document.deleted_at.is_(None),
            KnowledgeBase.deleted_at.is_(None),
            DocumentVersion.status == "ready",
        )
        .order_by(DocumentVersion.created_at, DocumentVersion.id)
    ).all()
    created = 0
    for version in versions:
        key = job_key(org_id, version.id, index.id)
        if session.scalar(select(IngestionJob.id).where(IngestionJob.idempotency_key == key)):
            continue
        session.add(
            IngestionJob(
                org_id=org_id,
                document_version_id=version.id,
                index_version_id=index.id,
                idempotency_key=key,
            )
        )
        created += 1
    session.flush()
    return created


def missing_from_index(session: Session, index_version_id: UUID) -> int:
    """Live documents whose current version is not ready in this index version."""
    ready_here = exists().where(
        DocumentIndex.document_version_id == Document.current_version_id,
        DocumentIndex.index_version_id == index_version_id,
        DocumentIndex.status == "ready",
    )
    return int(
        session.scalar(
            select(func.count())
            .select_from(Document)
            .join(KnowledgeBase, KnowledgeBase.id == Document.kb_id)
            .where(
                Document.current_version_id.is_not(None),
                Document.deleted_at.is_(None),
                KnowledgeBase.deleted_at.is_(None),
                ~ready_here,
            )
        )
        or 0
    )


# ---------------------------------------------------------------- vector cleanup


def mark_obsolete(session: Session, org_id: UUID) -> int:
    """Phase 1 of cleanup: mark obsolete document indexes ``deleting`` (row-locked).

    Obsolete means the document (or its KB) is deleted, or the version is older than the
    document's current version. Indexes with a queued/running/retrying job are left alone;
    the publication transaction re-checks status, so a racing attempt cannot resurrect
    an index marked here.
    """
    current = aliased(DocumentVersion)
    active_job = exists().where(
        IngestionJob.document_version_id == DocumentIndex.document_version_id,
        IngestionJob.index_version_id == DocumentIndex.index_version_id,
        IngestionJob.status.in_(ACTIVE_JOB_STATUSES),
    )
    targets = session.scalars(
        select(DocumentIndex)
        .join(DocumentVersion, DocumentVersion.id == DocumentIndex.document_version_id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .join(KnowledgeBase, KnowledgeBase.id == Document.kb_id)
        .outerjoin(current, current.id == Document.current_version_id)
        .where(
            DocumentIndex.org_id == org_id,
            DocumentIndex.status.in_(("ready", "building", "failed")),
            (Document.deleted_at.is_not(None))
            | (KnowledgeBase.deleted_at.is_not(None))
            | and_(
                current.id.is_not(None),
                DocumentVersion.version_number < current.version_number,
            ),
            ~active_job,
        )
        .with_for_update(of=DocumentIndex)
    ).all()
    for document_index in targets:
        document_index.status = "deleting"
    return len(targets)


@dataclass
class PurgeOutcome:
    purged: int = 0
    failed: int = 0


def purge_vectors(
    sessions: sessionmaker[Session], vector_index_factory: VectorIndexFactory, org_id: UUID
) -> PurgeOutcome:
    """Phase 2: delete Qdrant points of ``deleting`` indexes, then mark them ``deleted``.

    Each index is deleted by filter and verified empty before its status changes. A
    vector-store failure leaves the index ``deleting``; running cleanup again retries it.
    Chunk rows stay in PostgreSQL (history and citations); they are never visible again
    because search requires a ``ready`` index.
    """
    with sessions() as session:
        targets = session.execute(
            select(DocumentIndex.id, IndexVersion, EmbeddingModel)
            .join(IndexVersion, IndexVersion.id == DocumentIndex.index_version_id)
            .join(EmbeddingModel, EmbeddingModel.id == IndexVersion.embedding_model_id)
            .where(DocumentIndex.org_id == org_id, DocumentIndex.status == "deleting")
            .order_by(DocumentIndex.id)
        ).all()
    outcome = PurgeOutcome()
    for document_index_id, index, model in targets:
        try:
            vectors = vector_index_factory(index, model)
            vectors.delete_document_index(document_index_id)
            if vectors.count(document_index_id) != 0:
                raise VectorStoreError("points remain after delete")
        except VectorStoreError:
            outcome.failed += 1
            continue
        with sessions.begin() as session:
            row = session.get(DocumentIndex, document_index_id, with_for_update=True)
            if row is not None and row.status == "deleting":
                row.status = "deleted"
                outcome.purged += 1
    return outcome
