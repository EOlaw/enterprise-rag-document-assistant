"""Internal ingestion service. Callers must supply an authorized organization/KB.

This is infrastructure for the future API, not an authentication or authorization layer.
All document/version/job rows commit together; an object written before a failed commit
may be orphaned and is retained for a future retention-aware sweeper.
"""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from kip.db.models import Document, DocumentVersion, IndexVersion, IngestionJob, KnowledgeBase
from kip.storage import ObjectStore, object_key
from rag.config import IngestionLimits
from rag.ingestion.identity import sha256_hex
from rag.ingestion.validation import detect_file_type, safe_display_name


@dataclass(frozen=True)
class Submission:
    document_id: UUID
    version_id: UUID
    job_id: UUID
    duplicate: bool


def submit_document(
    sessions: sessionmaker[Session],
    store: ObjectStore,
    *,
    org_id: UUID,
    kb_id: UUID,
    index_version_id: UUID,
    filename: str,
    data: bytes,
    document_id: UUID | None = None,
    limits: IngestionLimits | None = None,
) -> Submission:
    limits = limits or IngestionLimits()
    filename = safe_display_name(filename)
    file_type = detect_file_type(filename, data, limits)
    digest = sha256_hex(data)
    with sessions.begin() as session:
        # Serialize submissions within a KB, including concurrent first uploads of the
        # same bytes. The worker never locks this row; processing is not serialized.
        kb = session.scalar(
            select(KnowledgeBase)
            .where(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.org_id == org_id,
                KnowledgeBase.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if kb is None:
            raise LookupError("knowledge base not found in organization")
        index = session.get(IndexVersion, index_version_id)
        if index is None or index.status == "retired":
            raise LookupError("usable index version not found")

        version: DocumentVersion | None = None
        if document_id is None:
            version = session.scalar(
                select(DocumentVersion)
                .join(
                    Document,
                    Document.id == DocumentVersion.document_id,
                )
                .where(
                    Document.org_id == org_id,
                    Document.kb_id == kb_id,
                    Document.deleted_at.is_(None),
                    DocumentVersion.raw_sha256 == digest,
                )
                .order_by(DocumentVersion.created_at, DocumentVersion.id)
                .limit(1)
            )
            if version is not None:
                document_id = version.document_id

        if document_id is None:
            document = Document(org_id=org_id, kb_id=kb_id, title=filename)
            session.add(document)
            session.flush()
        else:
            existing = session.scalar(
                select(Document)
                .where(
                    Document.id == document_id,
                    Document.org_id == org_id,
                    Document.kb_id == kb_id,
                    Document.deleted_at.is_(None),
                )
                .with_for_update()
            )
            if existing is None:
                raise LookupError("document not found in knowledge base")
            document = existing
            version = session.scalar(
                select(DocumentVersion).where(
                    DocumentVersion.document_id == document.id,
                    DocumentVersion.org_id == org_id,
                    DocumentVersion.raw_sha256 == digest,
                )
            )

        duplicate = version is not None
        if version is None:
            number = (
                session.scalar(
                    select(func.max(DocumentVersion.version_number)).where(
                        DocumentVersion.document_id == document.id,
                    )
                )
                or 0
            )
            version = DocumentVersion(
                org_id=org_id,
                document_id=document.id,
                version_number=number + 1,
                filename=filename,
                file_type=file_type.value,
                mime_type=file_type.mime_type,
                byte_size=len(data),
                raw_sha256=digest,
                storage_key="pending",
            )
            session.add(version)
            session.flush()
            version.storage_key = object_key(org_id, version.id, "original")
            store.put(version.storage_key, data)

        key = sha256_hex(f"ingest:{org_id}:{version.id}:{index.id}".encode())
        job = session.scalar(select(IngestionJob).where(IngestionJob.idempotency_key == key))
        if job is None:
            job = IngestionJob(
                org_id=org_id,
                document_version_id=version.id,
                index_version_id=index.id,
                idempotency_key=key,
            )
            session.add(job)
            session.flush()
        return Submission(document.id, version.id, job.id, duplicate)
