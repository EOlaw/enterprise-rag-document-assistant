"""Dense candidate retrieval: embed the query, search one scoped vector collection.

This returns *candidates* only. They are not evidence until the caller re-checks each one
against the system of record (tenant, deletion, current version, index readiness).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from rag.embeddings.base import Embedder, validate_vectors
from rag.indexing.base import VectorHit


class QueryTooLongError(ValueError):
    pass


class VectorSearcher(Protocol):
    def search(
        self, vector: Sequence[float], *, org_id: UUID, kb_ids: Sequence[UUID], limit: int
    ) -> list[VectorHit]: ...


@dataclass(frozen=True, slots=True)
class DenseCandidate:
    point_id: UUID
    score: float
    rank: int  # 1-based


@dataclass(frozen=True, slots=True)
class DenseCandidates:
    candidates: list[DenseCandidate]
    query_tokens: int
    embed_ms: float
    search_ms: float


def dense_candidates(
    query: str,
    embedder: Embedder,
    searcher: VectorSearcher,
    *,
    org_id: UUID,
    kb_ids: Sequence[UUID],
    limit: int,
) -> DenseCandidates:
    query = " ".join(query.split())
    if not query:
        raise ValueError("query is empty")
    if not kb_ids:
        raise ValueError("at least one knowledge base is required")
    [tokens] = embedder.count_tokens([query])
    if tokens > embedder.max_input_tokens:
        # Refuse rather than let the model silently embed a truncated query.
        raise QueryTooLongError(
            f"query has {tokens} tokens; the embedding model accepts {embedder.max_input_tokens}"
        )
    started = time.perf_counter()
    vector = embedder.embed_query(query)
    validate_vectors([vector], count=1, dimension=embedder.dimension)
    embedded = time.perf_counter()
    hits = searcher.search(vector, org_id=org_id, kb_ids=kb_ids, limit=limit)
    searched = time.perf_counter()
    return DenseCandidates(
        candidates=[DenseCandidate(hit.id, hit.score, rank) for rank, hit in enumerate(hits, 1)],
        query_tokens=tokens,
        embed_ms=(embedded - started) * 1000,
        search_ms=(searched - embedded) * 1000,
    )
