"""Scoped dense search with a PostgreSQL re-check of every candidate.

Callers must supply an organization and knowledge bases they are authorized for; user
authentication and permission resolution arrive with the API (Phase 6). Qdrant narrows
candidates by tenant/KB payload filters; PostgreSQL then decides visibility:

* chunk and document belong to the organization, document and KB are not deleted;
* the chunk's version is the document's current version and is ``ready``;
* its document index is ``ready`` in the requested index version.

Only re-checked chunks return text. Scores are cosine similarities for this query, not
probabilities, and are not comparable across queries or models.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kip.db.models import (
    Document,
    DocumentChunk,
    DocumentIndex,
    DocumentVersion,
    EmbeddingModel,
    IndexVersion,
    KnowledgeBase,
)
from kip.vectors import EmbedderFactory, VectorIndexFactory
from rag.embeddings import check_embedder_matches
from rag.retrieval import dense_candidates

MAX_TOP_K = 50


class SearchScopeError(LookupError):
    """Organization, knowledge base or index version is not usable for this search."""


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    score: float
    vector_rank: int
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    version_number: int
    kb_id: UUID
    title: str
    filename: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    heading_path: list[str]


@dataclass(frozen=True)
class SearchResult:
    index_version_id: UUID
    model: str
    revision: str
    query_tokens: int
    candidates: int
    dropped_by_recheck: int
    results: list[RetrievedChunk]
    timings_ms: dict[str, float]


def resolve_index(session: Session, index_version_id: UUID | None) -> IndexVersion:
    if index_version_id is None:
        index = session.scalar(select(IndexVersion).where(IndexVersion.status == "active"))
        if index is None:
            raise SearchScopeError("no active index version; pass --index or activate one")
        return index
    index = session.get(IndexVersion, index_version_id)
    if index is None or index.status == "retired":
        raise SearchScopeError("index version not found or retired")
    return index


def dense_search(
    sessions: sessionmaker[Session],
    embedder_factory: EmbedderFactory,
    vector_index_factory: VectorIndexFactory,
    *,
    org_id: UUID,
    kb_ids: Sequence[UUID],
    query: str,
    top_k: int = 5,
    index_version_id: UUID | None = None,
    overfetch: int = 4,
) -> SearchResult:
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
    scope = sorted(set(kb_ids))
    if not scope:
        raise SearchScopeError("at least one knowledge base is required")
    started = time.perf_counter()
    with sessions() as session:
        index = resolve_index(session, index_version_id)
        model = session.get(EmbeddingModel, index.embedding_model_id)
        if model is None:
            raise SearchScopeError("index version has no embedding model")
        usable = set(
            session.scalars(
                select(KnowledgeBase.id).where(
                    KnowledgeBase.id.in_(scope),
                    KnowledgeBase.org_id == org_id,
                    KnowledgeBase.deleted_at.is_(None),
                )
            )
        )
    if usable != set(scope):
        # Refuse rather than silently narrowing: a typo must not look like "no results".
        raise SearchScopeError("knowledge base not found in organization")

    embedder = embedder_factory(model)
    check_embedder_matches(
        embedder,
        model_id=model.name,
        revision=model.revision,
        dimension=model.dimension,
        max_seq_length=model.max_seq_tokens,
    )
    vectors = vector_index_factory(index, model)
    resolved = time.perf_counter()
    dense = dense_candidates(
        query, embedder, vectors, org_id=org_id, kb_ids=scope, limit=top_k * overfetch
    )
    recheck_started = time.perf_counter()
    rows = recheck(sessions, org_id=org_id, kb_ids=scope, index=index,
                   chunk_ids=[c.point_id for c in dense.candidates])  # fmt: skip
    results: list[RetrievedChunk] = []
    for candidate in dense.candidates:
        row = rows.get(candidate.point_id)
        if row is None:
            continue
        chunk, title, version_number, filename, kb_id = row
        results.append(
            RetrievedChunk(
                rank=len(results) + 1,
                score=candidate.score,
                vector_rank=candidate.rank,
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_version_id=chunk.document_version_id,
                version_number=version_number,
                kb_id=kb_id,
                title=title,
                filename=filename,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                heading_path=list(chunk.heading_path),
            )
        )
        if len(results) == top_k:
            break
    finished = time.perf_counter()
    return SearchResult(
        index_version_id=index.id,
        model=model.name,
        revision=model.revision,
        query_tokens=dense.query_tokens,
        candidates=len(dense.candidates),
        dropped_by_recheck=len(dense.candidates) - len(rows),
        results=results,
        timings_ms={
            # setup includes loading the embedding model on first use in this process.
            "setup": round((resolved - started) * 1000, 2),
            "embed": round(dense.embed_ms, 2),
            "vector_search": round(dense.search_ms, 2),
            "recheck": round((finished - recheck_started) * 1000, 2),
            "total": round((finished - started) * 1000, 2),
        },
    )


Row = tuple[DocumentChunk, str, int, str, UUID]


def recheck(
    sessions: sessionmaker[Session],
    *,
    org_id: UUID,
    kb_ids: Sequence[UUID],
    index: IndexVersion,
    chunk_ids: Sequence[UUID],
) -> dict[UUID, Row]:
    """Return only the candidates PostgreSQL currently allows this scope to see."""
    if not chunk_ids:
        return {}
    statement = (
        select(
            DocumentChunk,
            Document.title,
            DocumentVersion.version_number,
            DocumentVersion.filename,
            Document.kb_id,
        )
        .join(
            DocumentIndex,
            (DocumentIndex.id == DocumentChunk.document_index_id)
            & (DocumentIndex.org_id == org_id)
            & (DocumentIndex.index_version_id == index.id)
            & (DocumentIndex.status == "ready"),
        )
        .join(
            DocumentVersion,
            (DocumentVersion.id == DocumentChunk.document_version_id)
            & (DocumentVersion.org_id == org_id)
            & (DocumentVersion.status == "ready"),
        )
        .join(
            Document,
            (Document.id == DocumentChunk.document_id)
            & (Document.org_id == org_id)
            & (Document.current_version_id == DocumentChunk.document_version_id)
            & Document.deleted_at.is_(None),
        )
        .join(
            KnowledgeBase,
            (KnowledgeBase.id == Document.kb_id)
            & (KnowledgeBase.org_id == org_id)
            & KnowledgeBase.deleted_at.is_(None),
        )
        .where(
            DocumentChunk.id.in_(list(chunk_ids)),
            DocumentChunk.org_id == org_id,
            Document.kb_id.in_(list(kb_ids)),
        )
    )
    with sessions() as session:
        return {
            row[0].id: (row[0], row[1], row[2], row[3], row[4])
            for row in session.execute(statement)
        }
