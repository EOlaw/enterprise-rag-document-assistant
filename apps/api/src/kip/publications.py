"""Publication authority metadata (trusted operator commands; no user permissions yet).

Records whether a document is current, superseded by another document, or withdrawn,
plus its publication date. This is separate from document *versions*: uploading a new
version of the same document already hides older versions from search.
"""

from datetime import date
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from kip.db.models import Document, KnowledgeBase, utcnow
from rag.answering import PublicationStatus

MAX_NOTE = 500


def set_publication(
    session: Session,
    org_id: UUID,
    document_id: UUID,
    *,
    status: PublicationStatus,
    published_on: date | None,
    superseded_by: UUID | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    document = session.scalar(
        select(Document)
        .where(Document.id == document_id, Document.org_id == org_id, Document.deleted_at.is_(None))
        .with_for_update()
    )
    if document is None:
        raise LookupError("document not found in organization")
    if superseded_by is not None:
        if status is PublicationStatus.CURRENT or status is PublicationStatus.UNKNOWN:
            raise ValueError("only superseded or withdrawn documents can name a replacement")
        if superseded_by == document_id:
            raise ValueError("a document cannot replace itself")
        replacement = session.scalar(
            select(Document.id).where(
                Document.id == superseded_by,
                Document.org_id == org_id,
                Document.deleted_at.is_(None),
            )
        )
        if replacement is None:
            raise LookupError("replacement document not found in organization")
    if note is not None and len(note) > MAX_NOTE:
        raise ValueError(f"note is limited to {MAX_NOTE} characters")
    document.publication_status = status.value
    document.published_on = published_on
    document.superseded_by_document_id = superseded_by
    document.publication_note = note
    document.publication_updated_at = utcnow()
    return describe(document)


def describe(document: Document) -> dict[str, Any]:
    return {
        "document_id": str(document.id),
        "title": document.title,
        "publication_status": document.publication_status,
        "published_on": document.published_on.isoformat() if document.published_on else None,
        "superseded_by_document_id": (
            str(document.superseded_by_document_id) if document.superseded_by_document_id else None
        ),
        "note": document.publication_note,
    }


def list_documents(
    session: Session, org_id: UUID, kb_id: UUID | None = None
) -> list[dict[str, Any]]:
    query = (
        select(Document)
        .join(KnowledgeBase, KnowledgeBase.id == Document.kb_id)
        .where(
            Document.org_id == org_id,
            Document.deleted_at.is_(None),
            KnowledgeBase.deleted_at.is_(None),
        )
        .order_by(Document.title, Document.id)
    )
    if kb_id is not None:
        query = query.where(Document.kb_id == kb_id)
    return [
        {
            **describe(d),
            "kb_id": str(d.kb_id),
            "current_version_id": str(d.current_version_id) if d.current_version_id else None,
        }
        for d in session.scalars(query)
    ]
