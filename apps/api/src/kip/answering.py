"""Grounded answers: retrieve, apply the authority policy, select evidence within a token
budget, check sufficiency, generate, validate citations, and persist a trace.

Visibility is decided exactly as for search (organization/KB scope plus the PostgreSQL
re-check of deletion, current version and index readiness). User authentication and
document permissions are Phase 6.

Persistence stores identifiers, configuration, scores, statuses and the generated answer,
never prompts or document text. The question is stored unless ``store_question`` is off,
in which case only its SHA-256 is kept.
"""

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

import rag
from kip.db.models import AnswerRun, Document, EmbeddingModel
from kip.retrieval import Mode, RetrievedChunk, SearchResult, resolve_index, search
from kip.vectors import EmbedderFactory, VectorIndexFactory
from rag.answering import (
    UNKNOWN_AUTHORITY,
    Authority,
    AuthorityPolicy,
    Candidate,
    Grounding,
    PublicationStatus,
    Selection,
    Sufficiency,
    SufficiencyThresholds,
    Validation,
    assess,
    select_evidence,
    validate_claims,
)
from rag.errors import ConfigurationError
from rag.generation import (
    ANSWER_SCHEMA,
    PROMPT_VERSION,
    GenerationError,
    InvalidModelOutputError,
    LLMProvider,
    build_messages,
    parse_answer,
)
from rag.reranking import Reranker
from rag.retrieval import FusionConfig

# Budgets are counted in the embedding tokenizer (MiniLM word pieces). LLM tokenizers
# differ, so the context check multiplies by this factor; the provider's reported prompt
# tokens are verified after generation as well.
LLM_TOKEN_SAFETY = 1.3
WITHHELD = {Grounding.UNGROUNDED, Grounding.INVALID_OUTPUT, Grounding.ERROR}


@dataclass(frozen=True)
class AnswerSettings:
    mode: Mode | Literal["auto"] = "auto"
    rerank: bool = True
    top_k: int = 12
    budget_tokens: int = 1500
    max_sources: int = 8
    max_per_document: int = 3
    max_answer_tokens: int = 512
    policy: AuthorityPolicy = field(default_factory=AuthorityPolicy)
    thresholds: SufficiencyThresholds = field(
        default_factory=lambda: SufficiencyThresholds(min_coverage=0.5)
    )
    overfetch: int = 4
    max_candidates: int = 200
    hybrid_candidates: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 30
    store: bool = True
    store_question: bool = True


@dataclass(frozen=True)
class CitationRecord:
    label: str
    chunk_id: str
    document_id: str
    document_version_id: str
    version_number: int
    title: str
    filename: str
    page_start: int | None
    page_end: int | None
    char_start: int
    char_end: int
    heading_path: list[str]
    publication_status: str
    published_on: str | None
    superseded_by: str | None
    treatment: str
    supports: list[str]  # support status of each sentence citing this source


@dataclass(frozen=True)
class AnswerResult:
    run_id: str | None
    grounding: str
    answer: str | None
    abstain_reason: str | None
    sentences: list[dict[str, Any]]
    citations: list[CitationRecord]
    evidence: list[dict[str, Any]]
    skipped: list[dict[str, Any]]
    sufficiency: dict[str, Any]
    fabricated_labels: list[str]
    model: dict[str, Any]
    usage: dict[str, Any]
    timings_ms: dict[str, float]
    error: str | None = None
    retrieval: dict[str, Any] | None = None  # included when trace output is requested


@dataclass(frozen=True)
class Prepared:
    """Everything decided before any model call (retrieval, authority, evidence)."""

    mode: Mode
    found: SearchResult
    selection: Selection
    sufficiency: Sufficiency
    count_tokens: Callable[[str], int]
    timings_ms: dict[str, float]


def prepare_evidence(
    sessions: sessionmaker[Session],
    embedder_factory: EmbedderFactory,
    vector_index_factory: VectorIndexFactory,
    *,
    org_id: UUID,
    kb_ids: Sequence[UUID],
    question: str,
    config: AnswerSettings,
    reranker_factory: Callable[[], Reranker] | None = None,
    index_version_id: UUID | None = None,
) -> Prepared:
    timings: dict[str, float] = {}
    started = time.perf_counter()
    with sessions() as session:
        index = resolve_index(session, index_version_id)
        embedding_model = session.get(EmbeddingModel, index.embedding_model_id)
    if embedding_model is None:
        raise ConfigurationError("index version has no embedding model")
    mode: Mode = (
        config.mode
        if config.mode != "auto"
        else ("hybrid" if index.sparse_config is not None else "dense")
    )
    reranker = reranker_factory() if config.rerank and reranker_factory else None
    if config.rerank and reranker is None:
        raise ConfigurationError("reranking is enabled but no reranker factory was given")
    found = search(
        sessions,
        embedder_factory,
        vector_index_factory,
        org_id=org_id,
        kb_ids=kb_ids,
        query=question,
        mode=mode,
        top_k=config.top_k,
        index_version_id=index.id,
        overfetch=config.overfetch,
        max_candidates=config.max_candidates,
        hybrid_candidates=config.hybrid_candidates,
        fusion=FusionConfig(k=config.rrf_k),
        reranker=reranker,
        rerank_candidates=max(config.rerank_candidates, config.top_k),
    )
    timings["retrieval"] = _ms(started)

    moment = time.perf_counter()
    authorities = load_authorities(sessions, org_id, [r.document_id for r in found.results])
    candidates = [_candidate(r, authorities) for r in found.results]
    embedder = embedder_factory(embedding_model)

    def count(text: str) -> int:
        return embedder.count_tokens([text])[0]

    selection = select_evidence(
        candidates,
        policy=config.policy,
        budget_tokens=config.budget_tokens,
        count_tokens=count,
        max_sources=config.max_sources,
        max_per_document=config.max_per_document,
    )
    rerank_by_chunk = {r.chunk_id: r.rerank_score for r in found.results}
    sufficiency = assess(
        question,
        selection.evidence,
        [rerank_by_chunk.get(e.candidate.chunk_id) for e in selection.evidence],
        config.thresholds,
    )
    timings["evidence"] = _ms(moment)
    return Prepared(mode, found, selection, sufficiency, count, timings)


def answer(
    sessions: sessionmaker[Session],
    embedder_factory: EmbedderFactory,
    vector_index_factory: VectorIndexFactory,
    provider: LLMProvider,
    *,
    org_id: UUID,
    kb_ids: Sequence[UUID],
    question: str,
    settings: AnswerSettings | None = None,
    reranker_factory: Callable[[], Reranker] | None = None,
    index_version_id: UUID | None = None,
    include_trace: bool = False,
) -> AnswerResult:
    config = settings or AnswerSettings()
    _check_context(config, provider)
    started = time.perf_counter()
    prepared = prepare_evidence(
        sessions,
        embedder_factory,
        vector_index_factory,
        org_id=org_id,
        kb_ids=kb_ids,
        question=question,
        config=config,
        reranker_factory=reranker_factory,
        index_version_id=index_version_id,
    )
    mode, found, selection, sufficiency = (
        prepared.mode,
        prepared.found,
        prepared.selection,
        prepared.sufficiency,
    )
    count = prepared.count_tokens
    timings = dict(prepared.timings_ms)

    grounding: Grounding
    validation = Validation()
    abstain_reason: str | None = None
    error: str | None = None
    usage: dict[str, Any] = {"evidence_tokens": selection.tokens_used}
    model_info: dict[str, Any] = {
        "provider": provider.name,
        "model": provider.model,
        "prompt_version": PROMPT_VERSION,
    }
    if not sufficiency.sufficient:
        grounding = Grounding.ABSTAINED_RETRIEVAL
        abstain_reason = f"insufficient evidence ({sufficiency.reason})"
    else:
        moment = time.perf_counter()
        messages = build_messages(question, selection.evidence)
        usage["prompt_tokens_estimate"] = sum(count(m.content) for m in messages)
        try:
            model_info["digest"] = provider.model_digest
            generation = provider.generate(
                messages, max_tokens=config.max_answer_tokens, response_schema=ANSWER_SCHEMA
            )
            usage.update(
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                finish_reason=generation.finish_reason,
            )
            parsed = parse_answer(generation.text)
            if parsed.abstain:
                grounding = Grounding.ABSTAINED_MODEL
                abstain_reason = (parsed.abstain_reason or "model reported insufficient sources")[
                    :500
                ]
            else:
                validation = validate_claims(
                    [(s.text, s.citations) for s in parsed.sentences], selection.evidence
                )
                grounding = validation.grounding
        except InvalidModelOutputError as exc:
            grounding, error = Grounding.INVALID_OUTPUT, exc.code
        except GenerationError as exc:
            grounding, error = Grounding.ERROR, exc.code
        timings["generation"] = _ms(moment)

    text = None
    if validation.claims and grounding not in WITHHELD:
        text = " ".join(claim.text for claim in validation.claims)
    citations = _citations(selection, validation)
    timings["total"] = _ms(started)
    result = AnswerResult(
        run_id=None,
        grounding=grounding.value,
        answer=text,
        abstain_reason=abstain_reason,
        sentences=[
            {
                "text": c.text if text is not None else None,
                "citations": list(c.labels),
                "support": c.support.value,
                "coverage": c.coverage,
                "missing_numbers": list(c.missing_numbers),
            }
            for c in validation.claims
        ],
        citations=citations,
        evidence=[
            {
                "label": e.label,
                "chunk_id": str(e.candidate.chunk_id),
                "document_id": str(e.candidate.document_id),
                "document_version_id": str(e.candidate.document_version_id),
                "char_start": e.candidate.char_start,
                "char_end": e.candidate.char_end,
                "page_start": e.candidate.page_start,
                "page_end": e.candidate.page_end,
                "title": e.candidate.title,
                "version_number": e.candidate.version_number,
                "tokens": e.tokens,
                "treatment": e.treatment,
                "publication": e.candidate.authority.label,
                "retrieval_rank": e.candidate.retrieval_rank,
            }
            for e in selection.evidence
        ],
        skipped=[
            {"chunk_id": str(s.chunk_id), "reason": s.reason, "detail": s.detail}
            for s in selection.skipped
        ],
        sufficiency=asdict(sufficiency),
        fabricated_labels=validation.fabricated,
        model=model_info,
        usage=usage,
        timings_ms={k: round(v, 2) for k, v in timings.items()},
        error=error,
        retrieval=_retrieval_trace(found) if include_trace else None,
    )
    if config.store:
        result = _persist(sessions, result, found, config, org_id, kb_ids, question, mode)
    return result


def _check_context(config: AnswerSettings, provider: LLMProvider) -> None:
    system_allowance = 400  # system prompt, question and framing, in embedding tokens
    needed = (config.budget_tokens + system_allowance) * LLM_TOKEN_SAFETY + config.max_answer_tokens
    if needed > provider.context_window:
        raise ConfigurationError(
            f"evidence budget {config.budget_tokens} plus answer {config.max_answer_tokens} "
            f"tokens may exceed the {provider.context_window}-token context window"
        )


def load_authorities(
    sessions: sessionmaker[Session], org_id: UUID, document_ids: Sequence[UUID]
) -> dict[UUID, Authority]:
    if not document_ids:
        return {}
    with sessions() as session:
        rows = session.execute(
            select(
                Document.id,
                Document.publication_status,
                Document.published_on,
                Document.superseded_by_document_id,
                Document.publication_note,
            ).where(Document.org_id == org_id, Document.id.in_(set(document_ids)))
        ).all()
        replacement_ids = {row[3] for row in rows if row[3] is not None}
        titles: dict[UUID, str] = {}
        if replacement_ids:
            titles = {
                row_id: title
                for row_id, title in session.execute(
                    select(Document.id, Document.title).where(
                        Document.org_id == org_id, Document.id.in_(replacement_ids)
                    )
                )
            }
    return {
        row[0]: Authority(
            PublicationStatus(row[1]), row[2], titles.get(row[3]) if row[3] else None, row[4]
        )
        for row in rows
    }


def _candidate(result: RetrievedChunk, authorities: dict[UUID, Authority]) -> Candidate:

    return Candidate(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        document_version_id=result.document_version_id,
        version_number=result.version_number,
        title=result.title,
        filename=result.filename,
        text=result.text,
        char_start=result.char_start,
        char_end=result.char_end,
        page_start=result.page_start,
        page_end=result.page_end,
        heading_path=tuple(result.heading_path),
        retrieval_rank=result.rank,
        score=result.score,
        score_kind=result.score_kind,
        authority=authorities.get(result.document_id, UNKNOWN_AUTHORITY),
    )


def _citations(selection: Selection, validation: Validation) -> list[CitationRecord]:
    by_label = {e.label: e for e in selection.evidence}
    records = []
    for label in validation.cited_labels:
        evidence = by_label[label]
        c = evidence.candidate
        records.append(
            CitationRecord(
                label=label,
                chunk_id=str(c.chunk_id),
                document_id=str(c.document_id),
                document_version_id=str(c.document_version_id),
                version_number=c.version_number,
                title=c.title,
                filename=c.filename,
                page_start=c.page_start,
                page_end=c.page_end,
                char_start=c.char_start,
                char_end=c.char_end,
                heading_path=list(c.heading_path),
                publication_status=c.authority.status.value,
                published_on=c.authority.published_on.isoformat()
                if c.authority.published_on
                else None,
                superseded_by=c.authority.superseded_by,
                treatment=evidence.treatment,
                supports=[cl.support.value for cl in validation.claims if label in cl.labels],
            )
        )
    return records


def _retrieval_trace(found: SearchResult) -> dict[str, Any]:
    return {
        "mode": found.mode,
        "reranked": found.reranked,
        "index_version_id": str(found.index_version_id),
        "embedding": {"model": found.model, "revision": found.revision},
        "reranker": found.rerank,
        "fusion": found.fusion,
        "retrievers": {name: asdict(stats) for name, stats in found.retrievers.items()},
        "dropped_by_recheck": found.dropped_by_recheck,
        "results": [
            {
                "chunk_id": str(r.chunk_id),
                "document_id": str(r.document_id),
                "document_version_id": str(r.document_version_id),
                "rank": r.rank,
                "retrieval_rank": r.retrieval_rank,
                "dense_rank": r.dense_rank,
                "sparse_rank": r.sparse_rank,
                "score": r.score,
                "score_kind": r.score_kind,
                "rerank_score": r.rerank_score,
            }
            for r in found.results
        ],
        "timings_ms": found.timings_ms,
    }


def _persist(
    sessions: sessionmaker[Session],
    result: AnswerResult,
    found: SearchResult,
    config: AnswerSettings,
    org_id: UUID,
    kb_ids: Sequence[UUID],
    question: str,
    mode: Mode,
) -> AnswerResult:
    trace = {
        "versions": {"prompt": PROMPT_VERSION, "rag": rag.__version__},
        "config": {
            "mode": mode,
            "rerank": config.rerank,
            "top_k": config.top_k,
            "budget_tokens": config.budget_tokens,
            "max_sources": config.max_sources,
            "max_per_document": config.max_per_document,
            "max_answer_tokens": config.max_answer_tokens,
            "policy": config.policy.model_dump(),
            "thresholds": config.thresholds.model_dump(),
            "rrf_k": config.rrf_k,
            "hybrid_candidates": config.hybrid_candidates,
            "rerank_candidates": config.rerank_candidates,
        },
        "model": result.model,
        "retrieval": _retrieval_trace(found),
        # Identifiers, ranks, scores and statuses only; no document text, no prompts.
        "evidence": [{k: v for k, v in e.items() if k != "title"} for e in result.evidence],
        "skipped": result.skipped,
        "sufficiency": result.sufficiency,
        "claims": [{k: v for k, v in s.items() if k != "text"} for s in result.sentences],
        "citations": [
            {k: v for k, v in asdict(c).items() if k not in {"title", "filename", "heading_path"}}
            for c in result.citations
        ],
        "fabricated_labels": result.fabricated_labels,
        "usage": result.usage,
        "timings_ms": result.timings_ms,
    }
    run = AnswerRun(
        org_id=org_id,
        index_version_id=found.index_version_id,
        kb_ids=sorted(str(k) for k in set(kb_ids)),
        question=question if config.store_question else None,
        question_sha256=hashlib.sha256(question.encode("utf-8")).hexdigest(),
        mode=mode,
        reranked=found.reranked,
        grounding=result.grounding,
        abstain_reason=result.abstain_reason,
        answer=result.answer,
        error_code=result.error,
        trace=trace,
    )
    with sessions.begin() as session:
        session.add(run)
        session.flush()
        run_id = str(run.id)
    return AnswerResult(**{**asdict(result), "citations": result.citations, "run_id": run_id})


def _ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000
