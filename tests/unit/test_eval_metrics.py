from __future__ import annotations

import math

import pytest

from rag.evaluation.metrics import bootstrap, query_metrics
from rag.evaluation.relevance import (
    QuoteNotFoundError,
    RetrievedSpan,
    Span,
    locate_quote,
    result_gains,
)

CANONICAL = "Alpha beta gamma.\n\nIncident records are retained for three years. Delta epsilon."


def test_locate_quote_ignores_whitespace_and_rejects_missing_or_ambiguous() -> None:
    start, end = locate_quote("records are\n retained   for three years.", CANONICAL)
    assert CANONICAL[start:end] == "records are retained for three years."
    with pytest.raises(QuoteNotFoundError, match="not found"):
        locate_quote("retained for five years", CANONICAL)
    with pytest.raises(QuoteNotFoundError, match="ambiguous"):
        locate_quote("a", "a b a")


SPAN = Span("doc", 100, 200, grade=2)


def hit(start: int, end: int, document: str = "doc") -> RetrievedSpan:
    return RetrievedSpan(document, start, end)


def test_metrics_for_a_hand_computed_ranking() -> None:
    spans = [SPAN, Span("doc", 500, 600, grade=1)]
    ranked = [hit(0, 50), hit(90, 160), hit(480, 560), hit(510, 700)]  # miss, cover, cover, cover
    metrics = query_metrics(spans, ranked, k=3)
    assert metrics.recall == 1.0
    assert metrics.hit == 1.0
    assert metrics.reciprocal_rank == 0.5
    assert metrics.precision == pytest.approx(2 / 3)
    dcg = 2 / math.log2(3) + 1 / math.log2(4)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert metrics.ndcg == pytest.approx(dcg / idcg)


def test_miss_and_wrong_document_score_zero() -> None:
    metrics = query_metrics([SPAN], [hit(100, 200, document="other"), hit(0, 10)], k=2)
    assert (metrics.recall, metrics.hit, metrics.reciprocal_rank, metrics.ndcg) == (0, 0, 0, 0)


def test_overlapping_chunks_cannot_push_ndcg_above_one() -> None:
    ranked = [hit(160, 300), hit(170, 300), hit(100, 200), hit(100, 200)]  # 40%, 30%, 100%
    gains = result_gains([SPAN], ranked)
    assert gains == [1, 0, 1, 0]  # partial, (no new credit), cover remaining, duplicate
    assert query_metrics([SPAN], ranked, k=4).ndcg <= 1.0


def test_coverage_threshold_is_half_the_span() -> None:
    assert query_metrics([SPAN], [hit(150, 400)], k=1).recall == 1.0  # exactly 50%
    assert query_metrics([SPAN], [hit(151, 400)], k=1).recall == 0.0


def test_bootstrap_is_seeded_and_brackets_the_mean() -> None:
    values = [0.0, 1.0] * 20
    first, second = bootstrap(values, seed=7), bootstrap(values, seed=7)
    assert first == second
    assert first.ci_low <= first.mean == 0.5 <= first.ci_high
    assert bootstrap([1.0] * 5).ci_low == 1.0
