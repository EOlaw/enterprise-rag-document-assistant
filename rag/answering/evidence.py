"""Evidence selection under a strict token budget, with authority treatment applied.

Input: re-checked search results in retrieval order. Output: labelled evidence ``S1..Sn``
plus a record of every candidate that was *not* used and why (``excluded_by_policy``,
``duplicate``, ``document_cap``, ``over_budget``). The budget is a hard limit on the sum
of rendered source-block tokens. A block that does not fit is skipped, never truncated,
because a truncated passage could no longer be cited at exact offsets.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from rag.answering.authority import Authority, AuthorityPolicy, Treatment

DUPLICATE_OVERLAP = 0.5  # share of the shorter span covered by an already-selected span


@dataclass(frozen=True, slots=True)
class Candidate:
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    version_number: int
    title: str
    filename: str
    text: str
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    heading_path: tuple[str, ...]
    retrieval_rank: int
    score: float
    score_kind: str
    authority: Authority


@dataclass(frozen=True, slots=True)
class Evidence:
    label: str  # S1, S2, ...
    candidate: Candidate
    tokens: int
    treatment: str  # include | demote


@dataclass(frozen=True, slots=True)
class Skipped:
    chunk_id: UUID
    reason: str
    detail: str = ""


@dataclass
class Selection:
    evidence: list[Evidence] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    tokens_used: int = 0
    budget: int = 0


def render_source(label: str, candidate: Candidate) -> str:
    """The exact block shown to the model. Document text is data, never instructions:
    attribute values are escaped and closing tags inside the text are neutralised."""
    pages = (
        f"{candidate.page_start}-{candidate.page_end}"
        if candidate.page_start is not None
        else "n/a"
    )
    heading = " > ".join(candidate.heading_path)
    attributes = {
        "id": label,
        "title": candidate.title,
        "version": str(candidate.version_number),
        "pages": pages,
        "section": heading,
        "status": candidate.authority.label,
    }
    rendered = " ".join(f'{key}="{_attr(value)}"' for key, value in attributes.items() if value)
    body = candidate.text.replace("</source", "</ source").replace("<source", "< source")
    return f"<source {rendered}>\n{body}\n</source>"


def _attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def select_evidence(
    candidates: Sequence[Candidate],
    *,
    policy: AuthorityPolicy,
    budget_tokens: int,
    count_tokens: Callable[[str], int],
    max_sources: int = 8,
    max_per_document: int = 3,
) -> Selection:
    if budget_tokens <= 0 or max_sources <= 0:
        raise ValueError("budget_tokens and max_sources must be positive")
    selection = Selection(budget=budget_tokens)
    ordered: list[tuple[int, Candidate, Treatment]] = []
    for candidate in candidates:
        treatment = policy.treatment(candidate.authority.status, candidate.authority.published_on)
        if treatment == "exclude":
            selection.skipped.append(
                Skipped(candidate.chunk_id, "excluded_by_policy", candidate.authority.label)
            )
            continue
        ordered.append((1 if treatment == "demote" else 0, candidate, treatment))
    # Stable: demoted evidence after all included evidence, retrieval order within each.
    ordered.sort(key=lambda item: (item[0], item[1].retrieval_rank))

    per_document: dict[UUID, int] = {}
    for _, candidate, treatment in ordered:
        if len(selection.evidence) >= max_sources:
            selection.skipped.append(Skipped(candidate.chunk_id, "max_sources"))
            continue
        if _duplicates(candidate, selection.evidence):
            selection.skipped.append(Skipped(candidate.chunk_id, "duplicate"))
            continue
        if per_document.get(candidate.document_id, 0) >= max_per_document:
            selection.skipped.append(Skipped(candidate.chunk_id, "document_cap"))
            continue
        label = f"S{len(selection.evidence) + 1}"
        tokens = count_tokens(render_source(label, candidate))
        if selection.tokens_used + tokens > budget_tokens:
            selection.skipped.append(Skipped(candidate.chunk_id, "over_budget", f"{tokens} tokens"))
            continue
        selection.evidence.append(Evidence(label, candidate, tokens, treatment))
        selection.tokens_used += tokens
        per_document[candidate.document_id] = per_document.get(candidate.document_id, 0) + 1
    return selection


def _duplicates(candidate: Candidate, chosen: Sequence[Evidence]) -> bool:
    """Overlapping chunks of the same version (chunks overlap by design) add little."""
    for evidence in chosen:
        other = evidence.candidate
        if other.document_version_id != candidate.document_version_id:
            continue
        overlap = min(other.char_end, candidate.char_end) - max(
            other.char_start, candidate.char_start
        )
        shorter = min(other.char_end - other.char_start, candidate.char_end - candidate.char_start)
        if overlap > 0 and overlap / shorter >= DUPLICATE_OVERLAP:
            return True
    return False
