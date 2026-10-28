"""Retrieval-only evidence sufficiency: decide *before* calling a model whether the
selected evidence plausibly answers the question.

Signals (none is a probability):

* ``coverage``: share of the question's content terms (BM25 tokenizer, stop words removed)
  that occur anywhere in the selected evidence;
* ``top_rerank``: the highest cross-encoder logit among selected evidence, when reranking
  ran.

Decision rule: when reranking ran and ``min_rerank`` is set, the top logit alone decides
(it separated the tuning set far better than coverage); otherwise coverage decides. Each
threshold is calibrated in the configuration where it applies.

Thresholds are calibrated on a **tuning set** (the retrieval questions plus separate
unanswerable tuning questions), never on the answer evaluation set. See
``evals/datasets/nist-answers/README.md``. The check is deliberately simple. It catches
off-topic questions; it cannot tell whether evidence answers a question that shares its
vocabulary but asks something the corpus does not say.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from rag.answering.evidence import Evidence
from rag.sparse.bm25 import tokenize


class SufficiencyThresholds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_coverage: float = Field(ge=0, le=1)
    min_rerank: float | None = None  # applied only when reranking ran


@dataclass(frozen=True, slots=True)
class Sufficiency:
    sufficient: bool
    coverage: float
    top_rerank: float | None
    reason: str  # "ok" | "no_evidence" | "low_coverage" | "low_rerank"


def assess(
    question: str,
    evidence: Sequence[Evidence],
    rerank_scores: Sequence[float | None],
    thresholds: SufficiencyThresholds,
) -> Sufficiency:
    scores = [s for s in rerank_scores if s is not None]
    top_rerank = max(scores) if scores else None
    if not evidence:
        return Sufficiency(False, 0.0, top_rerank, "no_evidence")
    terms = set(tokenize(question))
    corpus = set(
        tokenize(
            " ".join(" ".join(e.candidate.heading_path) + " " + e.candidate.text for e in evidence)
        )
    )
    coverage = round(len(terms & corpus) / len(terms) if terms else 0.0, 3)
    minimum = thresholds.min_rerank
    if minimum is not None and top_rerank is not None:
        # Reranking ran: its logit decides; coverage is reported but not used.
        if top_rerank < minimum:
            return Sufficiency(False, coverage, top_rerank, "low_rerank")
    elif coverage < thresholds.min_coverage:
        return Sufficiency(False, coverage, top_rerank, "low_coverage")
    return Sufficiency(True, coverage, top_rerank, "ok")
