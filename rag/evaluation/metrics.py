"""Retrieval metrics over span labels, with percentile bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from rag.evaluation.relevance import (
    COVER_FRACTION,
    RetrievedSpan,
    Span,
    covered_spans,
    first_covering_rank,
    overlap_fraction,
    result_gains,
)


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    recall: float  # labelled spans covered in top-k / labelled spans
    precision: float  # top-k results covering any labelled span / k
    hit: float  # 1 if any top-k result covers a labelled span
    reciprocal_rank: float  # 1 / rank of first covering result (0 if none in top-k)
    ndcg: float  # graded, each span credited once


def query_metrics(spans: Sequence[Span], ranked: Sequence[RetrievedSpan], k: int) -> QueryMetrics:
    if not spans:
        raise ValueError("retrieval metrics require at least one labelled span")
    if k <= 0:
        raise ValueError("k must be positive")
    top = list(ranked[:k])
    recall = len(covered_spans(spans, top)) / len(spans)
    relevant_results = sum(
        1 for result in top if any(overlap_fraction(s, result) >= COVER_FRACTION for s in spans)
    )
    first = first_covering_rank(spans, top)
    gains = result_gains(spans, top)
    dcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))
    ideal = sorted((s.grade for s in spans), reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal, start=1))
    return QueryMetrics(
        recall=recall,
        precision=relevant_results / k,
        hit=1.0 if first is not None else 0.0,
        reciprocal_rank=1.0 / first if first else 0.0,
        ndcg=dcg / idcg if idcg else 0.0,
    )


@dataclass(frozen=True, slots=True)
class Estimate:
    mean: float
    ci_low: float
    ci_high: float
    n: int


def bootstrap(values: Sequence[float], *, resamples: int = 1000, seed: int = 0) -> Estimate:
    """Mean with a 95% percentile bootstrap interval over queries (seeded, reproducible)."""
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(resamples))
    low = means[int(0.025 * (resamples - 1))]
    high = means[math.ceil(0.975 * (resamples - 1))]
    return Estimate(statistics.fmean(values), low, high, n)
