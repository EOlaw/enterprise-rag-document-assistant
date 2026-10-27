from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest

from rag.indexing import VectorHit
from rag.retrieval import QueryTooLongError, dense_candidates
from tests.fakes import HashingEmbedder


class RecordingSearcher:
    def __init__(self, hits: list[VectorHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[UUID, list[UUID], int]] = []

    def search(
        self, vector: Sequence[float], *, org_id: UUID, kb_ids: Sequence[UUID], limit: int
    ) -> list[VectorHit]:
        self.calls.append((org_id, list(kb_ids), limit))
        return self.hits[:limit]


def test_candidates_are_ranked_and_scope_is_passed_through() -> None:
    org, kb = uuid4(), uuid4()
    hits = [VectorHit(uuid4(), 0.9), VectorHit(uuid4(), 0.5)]
    searcher = RecordingSearcher(hits)
    result = dense_candidates("  incident   response ", HashingEmbedder(), searcher,
                              org_id=org, kb_ids=[kb], limit=10)  # fmt: skip
    assert [(c.point_id, c.rank) for c in result.candidates] == [(hits[0].id, 1), (hits[1].id, 2)]
    assert searcher.calls == [(org, [kb], 10)]
    assert result.query_tokens == 2


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_is_rejected(query: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        dense_candidates(query, HashingEmbedder(), RecordingSearcher([]),
                         org_id=uuid4(), kb_ids=[uuid4()], limit=5)  # fmt: skip


def test_unscoped_search_is_rejected_before_any_vector_call() -> None:
    searcher = RecordingSearcher([])
    with pytest.raises(ValueError, match="knowledge base"):
        dense_candidates("x", HashingEmbedder(), searcher, org_id=uuid4(), kb_ids=[], limit=5)
    assert searcher.calls == []


def test_overlong_query_is_refused_not_truncated() -> None:
    searcher = RecordingSearcher([])
    with pytest.raises(QueryTooLongError):
        dense_candidates("token " * 300, HashingEmbedder(), searcher,
                         org_id=uuid4(), kb_ids=[uuid4()], limit=5)  # fmt: skip
    assert searcher.calls == []
