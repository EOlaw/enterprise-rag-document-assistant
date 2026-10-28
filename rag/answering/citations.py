"""Validate a model's cited answer against the evidence it was actually given.

Guarantees (deterministic, independent of the model):

* Only labels that were handed to the model (``S1..Sn``) can become citations. Any other
  label, in the citation list or inline in the text, is removed and counted as
  ``fabricated``. Removed labels never reach the caller as citations.
* Every citation resolves to a stored chunk: document, version, chunk, pages and exact
  character offsets, plus the publication status at answer time.

Support checking is a **lexical heuristic, not entailment**. For each answer sentence it
measures how many of the sentence's content terms appear in the text it cites, and
requires every number in the sentence to appear there too:

* ``supported``: term coverage >= 0.7 and all numbers present;
* ``partial``: coverage >= 0.4, or coverage >= 0.7 with a missing number;
* ``unsupported``: lower coverage, or no valid citation at all;
* ``unchecked``: fewer than 3 content terms, too short to judge lexically.

Known failure modes: a paraphrase with different words can be marked ``partial`` or
``unsupported`` although it is correct (false negative), and a sentence that reuses the
source's words while inverting its meaning ("must not" versus "must") can be marked
``supported`` (false positive). Thresholds were fixed before evaluation, not tuned.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from rag.answering.evidence import Evidence
from rag.sparse.bm25 import tokenize

LABEL = re.compile(r"\[(S\d+)\]")
ANY_LABEL = re.compile(r"\[([A-Za-z]+\d+)\]")
LABEL_WITH_SPACE = re.compile(r"\s*\[[A-Za-z]+\d+\]")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
MIN_TERMS = 3
SUPPORTED_COVERAGE = 0.7
PARTIAL_COVERAGE = 0.4


class Support(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNCHECKED = "unchecked"


class Grounding(StrEnum):
    GROUNDED = "grounded"  # every checkable sentence supported by its citations
    PARTIALLY_GROUNDED = "partially_grounded"
    UNGROUNDED = "ungrounded"  # nothing supported: the answer is withheld
    UNVERIFIED = "unverified"  # only uncheckable sentences
    ABSTAINED_RETRIEVAL = "abstained_retrieval"  # insufficient evidence, no model call
    ABSTAINED_MODEL = "abstained_model"  # the model declined to answer
    INVALID_OUTPUT = "invalid_output"  # unparseable or empty model output: withheld
    ERROR = "error"  # provider failure or context overflow: no answer


@dataclass(frozen=True, slots=True)
class Claim:
    text: str
    labels: tuple[str, ...]
    support: Support
    coverage: float | None
    missing_numbers: tuple[str, ...] = ()


@dataclass
class Validation:
    claims: list[Claim] = field(default_factory=list)
    fabricated: list[str] = field(default_factory=list)  # labels not given to the model
    cited_labels: list[str] = field(default_factory=list)  # valid labels, first-use order

    @property
    def grounding(self) -> Grounding:
        if not self.claims:
            return Grounding.INVALID_OUTPUT
        checked = [c for c in self.claims if c.support is not Support.UNCHECKED]
        if not checked:
            return Grounding.UNVERIFIED
        supported = sum(1 for c in checked if c.support is Support.SUPPORTED)
        if supported == len(checked) and not self.fabricated:
            return Grounding.GROUNDED
        if supported or any(c.support is Support.PARTIAL for c in checked):
            return Grounding.PARTIALLY_GROUNDED
        return Grounding.UNGROUNDED


def strip_labels(text: str) -> str:
    """Remove inline ``[S1]``-style markers together with the space before them."""
    return " ".join(LABEL_WITH_SPACE.sub("", text).split())


def validate_claims(
    sentences: Sequence[tuple[str, Sequence[str]]], evidence: Sequence[Evidence]
) -> Validation:
    """``sentences`` are (text, cited labels) pairs from the model's structured output."""
    by_label = {e.label: e for e in evidence}
    result = Validation()
    for raw_text, raw_labels in sentences:
        labels: list[str] = []
        for raw_label in [*raw_labels, *ANY_LABEL.findall(raw_text)]:
            label = raw_label.strip().strip("[]")
            if label in by_label:
                if label not in labels:
                    labels.append(label)
            elif label and label not in result.fabricated:
                result.fabricated.append(label)
        text = strip_labels(raw_text)
        if not text:
            continue
        for label in labels:
            if label not in result.cited_labels:
                result.cited_labels.append(label)
        result.claims.append(_check(text, [by_label[label] for label in labels]))
    return result


def _check(text: str, cited: Sequence[Evidence]) -> Claim:
    labels = tuple(e.label for e in cited)
    terms = set(tokenize(text))
    if not cited:
        return Claim(text, labels, Support.UNSUPPORTED, None)
    if len(terms) < MIN_TERMS:
        return Claim(text, labels, Support.UNCHECKED, None)
    source = " ".join(" ".join(e.candidate.heading_path) + " " + e.candidate.text for e in cited)
    source_terms = set(tokenize(source))
    coverage = len(terms & source_terms) / len(terms)
    source_numbers = set(NUMBER.findall(source))
    missing = tuple(n for n in dict.fromkeys(NUMBER.findall(text)) if n not in source_numbers)
    if coverage >= SUPPORTED_COVERAGE and not missing:
        support = Support.SUPPORTED
    elif coverage >= PARTIAL_COVERAGE:
        support = Support.PARTIAL
    else:
        support = Support.UNSUPPORTED
    return Claim(text, labels, support, round(coverage, 3), missing)
