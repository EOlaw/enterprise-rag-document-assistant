"""BM25 candidate retrieval through the vector store's sparse index.

Like dense retrieval, this returns *candidates* that the caller must re-check against the
system of record. Scores are IDF-weighted BM25 for this query only.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from rag.indexing.base import VectorHit
from rag.retrieval.dense import Candidate
from rag.sparse.bm25 import SparseTerms, encode_query


class SparseSearcher(Protocol):
    def search_sparse(
        self, terms: SparseTerms, *, org_id: UUID, kb_ids: Sequence[UUID], limit: int
    ) -> list[VectorHit]: ...


@dataclass(frozen=True, slots=True)
class SparseCandidates:
    candidates: list[Candidate]
    query_terms: int  # distinct hashed terms after stop-word removal
    encode_ms: float
    search_ms: float


def sparse_candidates(
    query: str,
    searcher: SparseSearcher,
    *,
    org_id: UUID,
    kb_ids: Sequence[UUID],
    limit: int,
) -> SparseCandidates:
    query = " ".join(query.split())
    if not query:
        raise ValueError("query is empty")
    if not kb_ids:
        raise ValueError("at least one knowledge base is required")
    started = time.perf_counter()
    terms = encode_query(query)
    encoded = time.perf_counter()
    hits = searcher.search_sparse(terms, org_id=org_id, kb_ids=kb_ids, limit=limit) if terms else []
    searched = time.perf_counter()
    return SparseCandidates(
        candidates=[Candidate(hit.id, hit.score, rank) for rank, hit in enumerate(hits, 1)],
        query_terms=len(terms),
        encode_ms=(encoded - started) * 1000,
        search_ms=(searched - encoded) * 1000,
    )
