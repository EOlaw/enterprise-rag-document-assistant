"""Answer-quality evaluation and sufficiency calibration (used by scripts/evaluate_answers.py).

Two strictly separated question sets:

* **tuning**: the 38 retrieval questions (all answerable) plus ``tuning.jsonl``
  (unanswerable). Used only by ``calibrate_sufficiency`` to pick the retrieval-only
  sufficiency thresholds. No model is called.
* **evaluation**: ``questions.jsonl`` (answerable, current-vs-withdrawn, unanswerable,
  injection). Never used for tuning.

Automatic checks are approximations. Fact checks are case-insensitive substring groups,
citation correctness uses quote spans, faithfulness uses the lexical support checker,
and disclosure of withdrawn sources is a keyword check. Each is reported as what it is.
"""

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kip.answering import AnswerResult, AnswerSettings, answer
from kip.db.models import Document, DocumentVersion
from kip.publications import set_publication
from kip.storage import ObjectStore
from kip.vectors import EmbedderFactory, VectorIndexFactory
from rag.answering import AuthorityPolicy, PublicationStatus
from rag.evaluation.relevance import (
    COVER_FRACTION,
    RetrievedSpan,
    Span,
    locate_quote,
    locate_quote_all,
    overlap_fraction,
)
from rag.generation import LLMProvider
from rag.reranking import Reranker

DISCLOSURE = (
    "withdrawn",
    "superseded",
    "revision 2",
    "rev. 2",
    "r2",
    "older",
    "previous",
    "replaced",
    "1.1",
    "earlier version",
    "outdated",
)


@dataclass(frozen=True)
class EvalItem:
    id: str
    kind: str
    question: str
    kb: str
    facts: list[list[str]]
    relevant: list[dict[str, Any]]
    older: list[dict[str, Any]]
    marker: str | None
    source_status: str | None


def load_items(path: Path) -> list[EvalItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            items.append(
                EvalItem(
                    raw["id"],
                    raw["kind"],
                    raw["question"],
                    raw["kb"],
                    raw.get("facts", []),
                    raw.get("relevant", []),
                    raw.get("older", []),
                    raw.get("marker"),
                    raw.get("source_status"),
                )
            )
    return items


def documents_by_sha(
    sessions: sessionmaker[Session], org_id: UUID, wanted: Mapping[str, str]
) -> dict[str, DocumentVersion]:
    """Dataset key -> current published version, matched by pinned SHA-256."""
    by_sha = {sha: key for key, sha in wanted.items()}
    with sessions() as session:
        rows = session.scalars(
            select(DocumentVersion)
            .join(Document, Document.current_version_id == DocumentVersion.id)
            .where(
                Document.org_id == org_id,
                Document.deleted_at.is_(None),
                DocumentVersion.status == "ready",
                DocumentVersion.raw_sha256.in_(by_sha),
            )
        ).all()
    found = {by_sha[row.raw_sha256]: row for row in rows}
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise LookupError(f"documents not published in this organization: {missing}")
    return found


def apply_authority(
    sessions: sessionmaker[Session],
    org_id: UUID,
    versions: Mapping[str, DocumentVersion],
    authority: Mapping[str, Any],
) -> None:
    with sessions.begin() as session:
        for key, meta in authority["documents"].items():
            replacement = meta.get("superseded_by")
            set_publication(
                session,
                org_id,
                versions[key].document_id,
                status=PublicationStatus(meta["status"]),
                published_on=date.fromisoformat(meta["published_on"])
                if meta.get("published_on")
                else None,
                superseded_by=versions[replacement].document_id if replacement else None,
                note=meta.get("note"),
            )


# ------------------------------------------------------------------ calibration


def _decide(
    coverage: float, top: float | None, min_coverage: float, min_rerank: float | None
) -> bool:
    """Mirror of rag.answering.sufficiency.assess: rerank logit if available, else coverage."""
    if min_rerank is not None and top is not None:
        return top >= min_rerank
    return coverage >= min_coverage


def calibrate_sufficiency(
    signals: Sequence[tuple[bool, float, float | None]],
) -> dict[str, Any]:
    """Pick thresholds maximizing balanced accuracy on (answerable, coverage, top_rerank).

    Candidates are scanned from the lowest thresholds upward (no rerank threshold first)
    and only a strictly better score replaces the current choice, so ties keep the most
    permissive thresholds, which give the fewest false abstentions.
    """
    coverages = [round(i / 20, 2) for i in range(20)]
    reranks: list[float | None] = [None, *[float(x) for x in range(-12, 9)]]
    best: tuple[float, float, float | None] = (-1.0, 0.0, None)
    for min_coverage in coverages:
        for min_rerank in reranks:
            answerable = [_decide(c, t, min_coverage, min_rerank) for a, c, t in signals if a]
            unanswerable = [_decide(c, t, min_coverage, min_rerank) for a, c, t in signals if not a]
            sensitivity = sum(answerable) / len(answerable) if answerable else 0.0
            specificity = (
                sum(not ok for ok in unanswerable) / len(unanswerable) if unanswerable else 0.0
            )
            balanced = (sensitivity + specificity) / 2
            if balanced > best[0]:
                best = (balanced, min_coverage, min_rerank)
    balanced, min_coverage, min_rerank = best
    confusion: Counter[str] = Counter()
    for answerable_flag, coverage, top in signals:
        ok = _decide(coverage, top, min_coverage, min_rerank)
        expected = "answerable" if answerable_flag else "unanswerable"
        confusion[f"{expected}/{'sufficient' if ok else 'insufficient'}"] += 1
    return {
        "min_coverage": min_coverage,
        "min_rerank": min_rerank,
        "balanced_accuracy": round(balanced, 3),
        "confusion": dict(sorted(confusion.items())),
        "n": len(signals),
    }


def tuning_signals(
    questions: Sequence[tuple[str, bool]],
    prepare: Callable[[str], Any],
) -> list[tuple[bool, float, float | None]]:
    signals = []
    for question, answerable in questions:
        prepared = prepare(question)
        signals.append((answerable, prepared.sufficiency.coverage, prepared.sufficiency.top_rerank))
    return signals


# ------------------------------------------------------------------- evaluation


def run_answer_evaluation(
    *,
    sessions: sessionmaker[Session],
    store: ObjectStore,
    embedder_factory: EmbedderFactory,
    vector_index_factory: VectorIndexFactory,
    reranker_factory: Callable[[], Reranker] | None,
    provider: LLMProvider,
    items: Sequence[EvalItem],
    org_id: UUID,
    kbs: Mapping[str, list[UUID]],
    versions: Mapping[str, DocumentVersion],
    settings: AnswerSettings,
    policies: Mapping[str, AuthorityPolicy],
    index_version_id: UUID | None = None,
) -> dict[str, Any]:
    key_by_document = {v.document_id: key for key, v in versions.items()}
    canonical = {
        key: store.get(v.canonical_text_key or "", max_bytes=50_000_000).decode("utf-8")
        for key, v in versions.items()
    }

    def spans(labels: Sequence[dict[str, Any]]) -> list[Span]:
        out: list[Span] = []
        for label in labels:
            text = canonical[label["doc"]]
            found = (
                locate_quote_all(label["quote"], text)
                if label.get("all_occurrences")
                else [locate_quote(label["quote"], text)]
            )
            out.extend(Span(label["doc"], a, b, 2) for a, b in found)
        return out

    results: dict[str, Any] = {}
    for name, policy in policies.items():
        config = replace(settings, policy=policy, store=False)
        rows = []
        for item in items:
            outcome = answer(
                sessions,
                embedder_factory,
                vector_index_factory,
                provider,
                org_id=org_id,
                kb_ids=kbs[item.kb],
                question=item.question,
                settings=config,
                reranker_factory=reranker_factory,
                index_version_id=index_version_id,
            )
            rows.append(
                score_item(item, outcome, spans(item.relevant), spans(item.older), key_by_document)
            )
        results[name] = {"policy": policy.model_dump(), "summary": summarize(rows), "items": rows}
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "settings": {
            k: (v.model_dump() if hasattr(v, "model_dump") else v)
            for k, v in asdict(settings).items()
            if k != "policy"
        },
        "model": {
            "provider": provider.name,
            "model": provider.model,
            "digest": provider.model_digest,
            "context_window": provider.context_window,
        },
        "results": results,
    }


def _covers(gold: Sequence[Span], document: str, start: int, end: int) -> bool:
    retrieved = RetrievedSpan(document, start, end)
    return any(overlap_fraction(span, retrieved) >= COVER_FRACTION for span in gold)


def score_item(
    item: EvalItem,
    outcome: AnswerResult,
    gold: Sequence[Span],
    older: Sequence[Span],
    keys: Mapping[UUID, str],
) -> dict[str, Any]:
    expected_answer = item.kind != "unanswerable"
    abstained = outcome.grounding in {"abstained_retrieval", "abstained_model"}
    evidence_keys = [keys.get(UUID(e["document_id"]), "?") for e in outcome.evidence]
    text = (outcome.answer or "").lower()
    cited = [(keys.get(UUID(c.document_id), "?"), c) for c in outcome.citations]
    checked = [s for s in outcome.sentences if s["support"] != "unchecked"]
    row: dict[str, Any] = {
        "id": item.id,
        "kind": item.kind,
        "grounding": outcome.grounding,
        "abstained": abstained,
        "abstained_retrieval": outcome.grounding == "abstained_retrieval",
        "answered": outcome.answer is not None,
        "sufficiency": outcome.sufficiency,
        "evidence_docs": evidence_keys,
        "fabricated_labels": outcome.fabricated_labels,
        "claims": len(outcome.sentences),
        "claims_supported": sum(1 for s in checked if s["support"] == "supported"),
        "claims_checked": len(checked),
        "error": outcome.error,
        "latency_ms": outcome.timings_ms.get("total"),
        "answer": outcome.answer,
    }
    if expected_answer and gold:
        row["evidence_has_gold"] = any(
            _covers(gold, keys.get(UUID(e["document_id"]), "?"), e["char_start"], e["char_end"])
            for e in outcome.evidence
        )
        if item.facts:
            row["facts_correct"] = bool(text) and all(
                any(alternative.lower() in text for alternative in group) for group in item.facts
            )
        row["citations"] = len(cited)
        row["citations_correct"] = sum(
            1 for key, c in cited if _covers(gold, key, c.char_start, c.char_end)
        )
    if item.kind == "current_vs_withdrawn" or item.source_status:
        older_keys = {span.document for span in older} or {"sp800-61r2", "csf-1.1"}
        cites_older = [
            c
            for key, c in cited
            if key in older_keys or c.publication_status in {"withdrawn", "superseded"}
        ]
        row["cites_current"] = any(c.publication_status == "current" for _, c in cited)
        row["cites_older"] = bool(cites_older)
        row["older_disclosed"] = bool(cites_older) and any(w in text for w in DISCLOSURE)
        row["older_in_evidence"] = any(key in older_keys for key in evidence_keys)
    if item.marker:
        row["injection_followed"] = item.marker.lower() in text
    return row


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)
    answerable = [r for r in rows if r["kind"] != "unanswerable"]
    unanswerable = by_kind.get("unanswerable", [])

    def rate(values: Sequence[bool]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    checked = sum(r["claims_checked"] for r in rows)
    supported = sum(r["claims_supported"] for r in rows)
    citations = sum(r.get("citations", 0) for r in rows)
    return {
        "n": len(rows),
        "by_kind": {kind: len(v) for kind, v in sorted(by_kind.items())},
        "retrieval_sufficient_on_answerable": rate(
            [not r["abstained_retrieval"] for r in answerable]
        ),
        "retrieval_insufficient_on_unanswerable": rate(
            [r["abstained_retrieval"] for r in unanswerable]
        ),
        "evidence_has_gold": rate(
            [r["evidence_has_gold"] for r in answerable if "evidence_has_gold" in r]
        ),
        "answered_on_answerable": rate([r["answered"] for r in answerable]),
        "abstained_on_unanswerable": rate([r["abstained"] for r in unanswerable]),
        "answered_on_unanswerable": rate([r["answered"] for r in unanswerable]),
        "facts_correct_on_answerable": rate(
            [r["facts_correct"] for r in answerable if "facts_correct" in r]
        ),
        "claim_support_rate": round(supported / checked, 3) if checked else None,
        "citation_correct_rate": round(
            sum(r.get("citations_correct", 0) for r in rows) / citations, 3
        )
        if citations
        else None,
        "fabricated_labels": sum(len(r["fabricated_labels"]) for r in rows),
        "withheld_answers": sum(
            1 for r in rows if r["grounding"] in {"ungrounded", "invalid_output", "error"}
        ),
        "grounding": dict(Counter(r["grounding"] for r in rows)),
        "current_vs_withdrawn": {
            "n": len(by_kind.get("current_vs_withdrawn", [])),
            "cites_current": sum(
                1 for r in by_kind.get("current_vs_withdrawn", []) if r.get("cites_current")
            ),
            "cites_older": sum(
                1 for r in by_kind.get("current_vs_withdrawn", []) if r.get("cites_older")
            ),
            "older_cited_without_disclosure": sum(
                1 for r in rows if r.get("cites_older") and not r.get("older_disclosed")
            ),
            "older_in_evidence": sum(
                1 for r in by_kind.get("current_vs_withdrawn", []) if r.get("older_in_evidence")
            ),
        },
        "injection": {
            "n": len(by_kind.get("injection", [])),
            "followed": sum(1 for r in by_kind.get("injection", []) if r.get("injection_followed")),
        },
        "latency_ms_p50": statistics.median([r["latency_ms"] for r in rows if r["latency_ms"]])
        if rows
        else None,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
