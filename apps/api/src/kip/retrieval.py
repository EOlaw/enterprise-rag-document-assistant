"""Scoped dense, BM25 and hybrid search with a PostgreSQL re-check of every candidate.

Callers must supply an organization and knowledge bases they are authorized for; user
authentication and permission resolution arrive with the API (Phase 6).

Pipeline (each retriever independently, then combined):

1. Qdrant returns candidates under the shared tenant/KB/``published`` filter.
2. PostgreSQL decides visibility: same organization; chunk's document index ``ready`` in
   this index version; version is the document's current version and ``ready``; document
   and KB not deleted; KB in the requested scope. Only re-checked chunks carry text.
3. **Replenishment:** if too few candidates survive, the retriever is queried again with a
   doubled limit until enough are visible, it runs out, or ``max_candidates`` is reached.
4. Hybrid mode fuses the *visible* lists with Reciprocal Rank Fusion, so stale or
   unauthorized points can never shift the ranks of visible ones.
5. Optional cross-encoder reranking reorders at most ``rerank_candidates`` results.

Scores are only comparable within one list for one query. Cosine, BM25, RRF and
cross-encoder logits are different scales, and none of them is a probability.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
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
from rag.reranking import MAX_RERANK_CANDIDATES, Reranker, rerank_order
from rag.retrieval import (
    Candidate,
    FusionConfig,
    embed_query_checked,
    reciprocal_rank_fusion,
    to_candidates,
)
from rag.sparse import encode_query

MAX_TOP_K = 50
Mode = Literal["dense", "sparse", "hybrid"]
MODES: tuple[Mode, ...] = ("dense", "sparse", "hybrid")


class SearchScopeError(LookupError):
    """Organization, knowledge base or index version is not usable for this search."""


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    score: float  # the score that produced this ranking; see score_kind
    score_kind: str  # cosine | bm25 | rrf | cross_encoder_logit
    retrieval_rank: int  # rank before reranking (equals rank without reranking)
    dense_rank: int | None
    dense_score: float | None
    sparse_rank: int | None
    sparse_score: float | None
    fused_score: float | None
    rerank_score: float | None
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
class RetrieverStats:
    wanted: int  # visible candidates requested from this retriever
    final_limit: int  # last Qdrant limit used
    returned: int  # candidates returned by Qdrant in the last round
    visible: int  # candidates that passed the PostgreSQL re-check
    dropped: int  # returned - visible
    rounds: int  # 1 = no replenishment needed
    exhausted: bool  # Qdrant returned fewer than the limit: nothing more to fetch
    capped: bool  # stopped at max_candidates with fewer visible than wanted


@dataclass(frozen=True)
class SearchResult:
    mode: Mode
    reranked: bool
    index_version_id: UUID
    model: str
    revision: str
    query_tokens: int | None  # embedding tokens (dense/hybrid)
    query_terms: int | None  # distinct BM25 terms (sparse/hybrid)
    candidates: int  # distinct candidates returned by Qdrant across retrievers
    dropped_by_recheck: int  # of those, rejected by PostgreSQL
    retrievers: dict[str, RetrieverStats]
    fusion: dict[str, Any] | None
    rerank: dict[str, Any] | None
    results: list[RetrievedChunk]
    timings_ms: dict[str, float]
    trace: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


Row = tuple[DocumentChunk, str, int, str, UUID]


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


def search(
    sessions: sessionmaker[Session],
    embedder_factory: EmbedderFactory,
    vector_index_factory: VectorIndexFactory,
    *,
    org_id: UUID,
    kb_ids: Sequence[UUID],
    query: str,
    mode: Mode = "dense",
    top_k: int = 5,
    index_version_id: UUID | None = None,
    overfetch: int = 4,
    max_candidates: int = 200,
    hybrid_candidates: int = 50,
    fusion: FusionConfig | None = None,
    reranker: Reranker | None = None,
    rerank_candidates: int = 30,
    explain: bool = False,
) -> SearchResult:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
    if reranker is not None and not top_k <= rerank_candidates <= MAX_RERANK_CANDIDATES:
        raise ValueError(f"rerank_candidates must be between top_k and {MAX_RERANK_CANDIDATES}")
    if not " ".join(query.split()):
        raise ValueError("query is empty")
    scope = sorted(set(kb_ids))
    if not scope:
        raise SearchScopeError("at least one knowledge base is required")
    fusion = fusion or FusionConfig()
    timings: dict[str, float] = {}
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
    if mode != "dense" and index.sparse_config is None:
        raise SearchScopeError(
            f"index version {index.id} has no BM25 vectors; use mode=dense or a hybrid "
            "index version (see docs/architecture/hybrid-retrieval.md)"
        )
    vectors = vector_index_factory(index, model)
    timings["setup"] = _ms(started)

    # Visible candidates wanted from each retriever.
    final_depth = max(top_k, rerank_candidates) if reranker is not None else top_k
    wanted = max(final_depth, hybrid_candidates) if mode == "hybrid" else final_depth
    if wanted > max_candidates:
        raise ValueError(f"{wanted} candidates wanted but max_candidates is {max_candidates}")
    start_limit = min(max_candidates, wanted * overfetch)

    rows: dict[UUID, Row] = {}
    returned_ids: set[UUID] = set()
    lists: dict[str, list[Candidate]] = {}
    stats: dict[str, RetrieverStats] = {}
    trace: dict[str, list[dict[str, Any]]] = {}
    query_tokens: int | None = None
    query_terms: int | None = None

    def check(ids: Sequence[UUID]) -> None:
        unseen = [i for i in ids if i not in returned_ids]
        returned_ids.update(ids)
        if unseen:
            rows.update(
                recheck(sessions, org_id=org_id, kb_ids=scope, index=index, chunk_ids=unseen)
            )

    def gather(name: str, fetch: Callable[[int], list[Candidate]]) -> None:
        limit, rounds = start_limit, 0
        while True:
            rounds += 1
            found = fetch(limit)
            check([c.point_id for c in found])
            visible = [c for c in found if c.point_id in rows]
            exhausted = len(found) < limit
            if len(visible) >= wanted or exhausted or limit >= max_candidates:
                break
            limit = min(max_candidates, limit * 2)
        # Ranks are recomputed over visible candidates only.
        lists[name] = [Candidate(c.point_id, c.score, r) for r, c in enumerate(visible, 1)]
        stats[name] = RetrieverStats(
            wanted=wanted,
            final_limit=limit,
            returned=len(found),
            visible=len(visible),
            dropped=len(found) - len(visible),
            rounds=rounds,
            exhausted=exhausted,
            capped=not exhausted and len(visible) < wanted,
        )
        if explain:
            trace[name] = [
                {
                    "qdrant_rank": c.rank,
                    "chunk_id": str(c.point_id),
                    "score": c.score,
                    "visible": c.point_id in rows,
                }
                for c in found
            ]

    if mode in ("dense", "hybrid"):
        embedder = embedder_factory(model)
        check_embedder_matches(
            embedder,
            model_id=model.name,
            revision=model.revision,
            dimension=model.dimension,
            max_seq_length=model.max_seq_tokens,
        )
        moment = time.perf_counter()
        vector, query_tokens = embed_query_checked(query, embedder)
        timings["embed"] = _ms(moment)
        moment = time.perf_counter()
        gather(
            "dense",
            lambda limit: to_candidates(
                vectors.search(vector, org_id=org_id, kb_ids=scope, limit=limit)
            ),
        )
        timings["dense"] = _ms(moment)
    if mode in ("sparse", "hybrid"):
        moment = time.perf_counter()
        terms = encode_query(query)
        query_terms = len(terms)
        gather(
            "sparse",
            lambda limit: to_candidates(
                vectors.search_sparse(terms, org_id=org_id, kb_ids=scope, limit=limit)
            ),
        )
        timings["sparse"] = _ms(moment)

    # Combine: the single list, or RRF over the visible lists.
    moment = time.perf_counter()
    fusion_info: dict[str, Any] | None = None
    fused_scores: dict[UUID, float] = {}
    if mode == "hybrid":
        fused = reciprocal_rank_fusion(lists, fusion)
        ordered = [(f.point_id, f.score) for f in fused]
        ranks = {f.point_id: f.ranks for f in fused}
        raw = {f.point_id: f.scores for f in fused}
        fused_scores = {f.point_id: f.score for f in fused}
        dense_ids = {c.point_id for c in lists["dense"]}
        sparse_ids = {c.point_id for c in lists["sparse"]}
        fusion_info = {
            "method": "rrf",
            **fusion.model_dump(),
            "fused": len(fused),
            "overlap": len(dense_ids & sparse_ids),
        }
        score_kind = "rrf"
    else:
        only = lists[mode]
        ordered = [(c.point_id, c.score) for c in only]
        ranks = {c.point_id: {mode: c.rank} for c in only}
        raw = {c.point_id: {mode: c.score} for c in only}
        score_kind = "cosine" if mode == "dense" else "bm25"
    timings["fuse"] = _ms(moment)
    retrieval_rank = {pid: position for position, (pid, _) in enumerate(ordered, 1)}

    # Optional cross-encoder over a bounded head of the list.
    rerank_info: dict[str, Any] | None = None
    rerank_scores: dict[UUID, float] = {}
    if reranker is not None and ordered:
        moment = time.perf_counter()
        head = ordered[:rerank_candidates]
        passages = [rows[pid][0].embedding_text for pid, _ in head]
        pair_tokens = reranker.pair_tokens(query, passages)
        order = rerank_order(reranker.score(query, passages))
        rerank_scores = {head[r.index][0]: r.score for r in order}
        ordered = [(head[r.index][0], r.score) for r in order] + ordered[len(head) :]
        score_kind = "cross_encoder_logit"
        rerank_info = {
            "model": reranker.model_id,
            "revision": reranker.revision,
            "candidates": len(head),
            "max_pair_tokens": max(pair_tokens),
            "max_length": reranker.max_length,
        }
        timings["rerank"] = _ms(moment)

    results: list[RetrievedChunk] = []
    for pid, score in ordered[:top_k]:
        chunk, title, version_number, filename, kb_id = rows[pid]
        results.append(
            RetrievedChunk(
                rank=len(results) + 1,
                score=score,
                score_kind=score_kind,
                retrieval_rank=retrieval_rank[pid],
                dense_rank=ranks[pid].get("dense"),
                dense_score=raw[pid].get("dense"),
                sparse_rank=ranks[pid].get("sparse"),
                sparse_score=raw[pid].get("sparse"),
                fused_score=fused_scores.get(pid),
                rerank_score=rerank_scores.get(pid),
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
    timings["total"] = _ms(started)
    return SearchResult(
        mode=mode,
        reranked=reranker is not None,
        index_version_id=index.id,
        model=model.name,
        revision=model.revision,
        query_tokens=query_tokens,
        query_terms=query_terms,
        candidates=len(returned_ids),
        dropped_by_recheck=sum(1 for i in returned_ids if i not in rows),
        retrievers=stats,
        fusion=fusion_info,
        rerank=rerank_info,
        results=results,
        timings_ms={name: round(value, 2) for name, value in timings.items()},
        trace=trace,
    )


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
    """Phase 3 entry point: dense mode, no reranking."""
    return search(
        sessions,
        embedder_factory,
        vector_index_factory,
        org_id=org_id,
        kb_ids=kb_ids,
        query=query,
        mode="dense",
        top_k=top_k,
        index_version_id=index_version_id,
        overfetch=overfetch,
    )


def _ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000


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
