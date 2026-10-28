"""Phase 2 schema. Composite keys enforce tenant and document ancestry.

Processing is deliberately distinct from publication: processed versions and building
indexes cannot satisfy the retrieval gate, which requires a ready index and current version.
Identity, grants and conversations will be added with the authenticated API in Phase 6.
"""

from datetime import UTC, date, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from uuid6 import uuid7


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class Row:
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid7)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Organization(Row, Base):
    __tablename__ = "organizations"
    slug: Mapped[str] = mapped_column(String(100), unique=True)
    name: Mapped[str] = mapped_column(String(255))


class KnowledgeBase(Row, Base):
    __tablename__ = "knowledge_bases"
    __table_args__ = (UniqueConstraint("id", "org_id"),)
    org_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"))
    name: Mapped[str] = mapped_column(String(255))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Document(Row, Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("id", "org_id"),
        ForeignKeyConstraint(["kb_id", "org_id"], ["knowledge_bases.id", "knowledge_bases.org_id"]),
        ForeignKeyConstraint(
            ["current_version_id", "id", "org_id"],
            ["document_versions.id", "document_versions.document_id", "document_versions.org_id"],
            name="fk_documents_current_version",
            use_alter=True,
            deferrable=True,
            initially="DEFERRED",
        ),
        # Publication authority (Phase 5): a *different* document can replace this one.
        ForeignKeyConstraint(
            ["superseded_by_document_id", "org_id"],
            ["documents.id", "documents.org_id"],
            name="fk_documents_superseded_by",
        ),
        CheckConstraint(
            "publication_status IN ('current','superseded','withdrawn','unknown')",
            name="publication_status",
        ),
        CheckConstraint("superseded_by_document_id <> id", name="not_self_superseded"),
    )
    org_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"))
    kb_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    title: Mapped[str] = mapped_column(Text)
    current_version_id: Mapped[UUID | None] = mapped_column(Uuid)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Not the same as document_versions.status='superseded' (versions of THIS document).
    publication_status: Mapped[str] = mapped_column(
        String(20), default="unknown", server_default="unknown"
    )
    published_on: Mapped[date | None] = mapped_column(Date)  # NULL = undated
    superseded_by_document_id: Mapped[UUID | None] = mapped_column(Uuid)
    publication_note: Mapped[str | None] = mapped_column(String(500))
    publication_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DocumentVersion(Row, Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("id", "org_id", name="uq_document_versions_id_org"),
        UniqueConstraint(
            "id", "document_id", "org_id", name="uq_document_versions_id_document_org"
        ),
        UniqueConstraint("document_id", "version_number"),
        UniqueConstraint("org_id", "document_id", "raw_sha256"),
        ForeignKeyConstraint(["document_id", "org_id"], ["documents.id", "documents.org_id"]),
        CheckConstraint("version_number > 0 AND byte_size > 0", name="positive_sizes"),
        CheckConstraint(
            "status IN ('pending','processing','processed','ready','failed','superseded')",
            name="status",
        ),
    )
    org_id: Mapped[UUID] = mapped_column(Uuid)
    document_id: Mapped[UUID] = mapped_column(Uuid)
    version_number: Mapped[int]
    status: Mapped[str] = mapped_column(String(20), default="pending")
    filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(10))
    mime_type: Mapped[str] = mapped_column(String(100))
    byte_size: Mapped[int]
    raw_sha256: Mapped[str] = mapped_column(String(64), index=True)
    text_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    storage_key: Mapped[str] = mapped_column(Text)
    canonical_text_key: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None]
    parser_version: Mapped[str | None] = mapped_column(String(30))
    warnings: Mapped[list[str]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=list
    )


class EmbeddingModel(Row, Base):
    __tablename__ = "embedding_models"
    __table_args__ = (UniqueConstraint("name", "revision"),)
    name: Mapped[str] = mapped_column(String(255))
    revision: Mapped[str] = mapped_column(String(100))
    dimension: Mapped[int]
    max_seq_tokens: Mapped[int]
    normalize: Mapped[bool] = mapped_column(default=True)
    distance: Mapped[str] = mapped_column(String(20), default="Cosine")


class IndexVersion(Row, Base):
    __tablename__ = "index_versions"
    __table_args__ = (
        CheckConstraint("status IN ('building','active','retired')", name="status"),
        Index(
            "uq_index_versions_active",
            "status",
            unique=True,
            postgresql_where=text("status = 'active'"),
            sqlite_where=text("status = 'active'"),
        ),
    )
    embedding_model_id: Mapped[UUID] = mapped_column(ForeignKey("embedding_models.id"))
    parser_version: Mapped[str] = mapped_column(String(30))
    chunking_config: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql")
    )
    chunking_fingerprint: Mapped[str] = mapped_column(String(16))
    # BM25 parameters (rag.sparse.Bm25Config); NULL means a dense-only index version.
    sparse_config: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql")
    )
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    qdrant_collection: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="building")


class DocumentIndex(Row, Base):
    __tablename__ = "document_indexes"
    __table_args__ = (
        UniqueConstraint("id", "org_id", "document_version_id"),
        UniqueConstraint("document_version_id", "index_version_id"),
        ForeignKeyConstraint(
            ["document_version_id", "org_id"], ["document_versions.id", "document_versions.org_id"]
        ),
        # deleting -> deleted: vectors of an obsolete index are being / have been removed.
        CheckConstraint(
            "status IN ('building','ready','failed','deleting','deleted')", name="status"
        ),
        CheckConstraint("chunk_count >= 0", name="chunk_count"),
    )
    org_id: Mapped[UUID] = mapped_column(Uuid)
    document_version_id: Mapped[UUID] = mapped_column(Uuid)
    index_version_id: Mapped[UUID] = mapped_column(ForeignKey("index_versions.id"))
    status: Mapped[str] = mapped_column(String(20), default="building")
    chunk_count: Mapped[int] = mapped_column(default=0)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DocumentChunk(Row, Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_index_id", "chunk_index"),
        ForeignKeyConstraint(
            ["document_index_id", "org_id", "document_version_id"],
            [
                "document_indexes.id",
                "document_indexes.org_id",
                "document_indexes.document_version_id",
            ],
        ),
        ForeignKeyConstraint(
            ["document_version_id", "document_id", "org_id"],
            ["document_versions.id", "document_versions.document_id", "document_versions.org_id"],
        ),
        CheckConstraint("char_start >= 0 AND char_end > char_start", name="offsets"),
        CheckConstraint("chunk_index >= 0 AND token_count > 0", name="counts"),
        CheckConstraint(
            "(page_start IS NULL AND page_end IS NULL) OR "
            "(page_start IS NOT NULL AND page_end IS NOT NULL "
            "AND page_start > 0 AND page_end >= page_start)",
            name="pages",
        ),
    )
    org_id: Mapped[UUID] = mapped_column(Uuid)
    document_index_id: Mapped[UUID] = mapped_column(Uuid)
    document_id: Mapped[UUID] = mapped_column(Uuid)
    document_version_id: Mapped[UUID] = mapped_column(Uuid)
    chunk_index: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    embedding_text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int]
    char_end: Mapped[int]
    page_start: Mapped[int | None]
    page_end: Mapped[int | None]
    heading_path: Mapped[list[str]] = mapped_column(JSON().with_variant(JSONB, "postgresql"))
    token_count: Mapped[int]
    content_hash: Mapped[str] = mapped_column(String(64), index=True)


class IngestionJob(Row, Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_version_id", "org_id"], ["document_versions.id", "document_versions.org_id"]
        ),
        CheckConstraint(
            "status IN ('queued','processing','retrying','completed','failed','cancelled')",
            name="status",
        ),
        CheckConstraint("attempt >= 0 AND max_attempts > 0", name="attempts"),
        CheckConstraint("kind = 'ingest'", name="kind"),
        Index(
            "ix_ingestion_jobs_claim",
            "run_after",
            "created_at",
            postgresql_where=text("status IN ('queued','retrying')"),
            sqlite_where=text("status IN ('queued','retrying')"),
        ),
        Index("ix_ingestion_jobs_lease", "status", "lease_expires_at"),
    )
    org_id: Mapped[UUID] = mapped_column(Uuid)
    document_version_id: Mapped[UUID] = mapped_column(Uuid)
    index_version_id: Mapped[UUID] = mapped_column(ForeignKey("index_versions.id"))
    kind: Mapped[str] = mapped_column(String(20), default="ingest")
    status: Mapped[str] = mapped_column(String(20), default="queued")
    attempt: Mapped[int] = mapped_column(default=0)
    max_attempts: Mapped[int] = mapped_column(default=3)
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True)
    stage: Mapped[str] = mapped_column(String(30), default="queued")
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(String(255))
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_token: Mapped[UUID | None] = mapped_column(Uuid)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AnswerRun(Row, Base):
    """Reproducibility trace of one grounded-answer request.

    Stores identifiers, configuration, scores, statuses and the generated answer. It does
    not store prompts or document text: the prompt is rebuilt from ``PROMPT_VERSION``,
    the question and the evidence chunk IDs. The question is stored unless disabled.
    """

    __tablename__ = "answer_runs"
    __table_args__ = (Index("ix_answer_runs_org_created", "org_id", "created_at"),)
    org_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"))
    index_version_id: Mapped[UUID] = mapped_column(ForeignKey("index_versions.id"))
    kb_ids: Mapped[list[str]] = mapped_column(JSON().with_variant(JSONB, "postgresql"))
    question: Mapped[str | None] = mapped_column(Text)
    question_sha256: Mapped[str] = mapped_column(String(64))
    mode: Mapped[str] = mapped_column(String(10))
    reranked: Mapped[bool] = mapped_column(Boolean)
    grounding: Mapped[str] = mapped_column(String(30))
    abstain_reason: Mapped[str | None] = mapped_column(String(500))
    answer: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(String(100))
    trace: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB, "postgresql"))
