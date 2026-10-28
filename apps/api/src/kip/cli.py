"""Trusted local administration commands; no public API or user authorization yet."""

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from kip.answering import answer
from kip.db.models import (
    AnswerRun,
    DocumentIndex,
    EmbeddingModel,
    IndexVersion,
    IngestionJob,
    KnowledgeBase,
    Organization,
    utcnow,
)
from kip.db.session import make_engine, session_factory
from kip.ingestion import submit_document
from kip.llm import answer_settings, make_provider
from kip.maintenance import (
    enqueue_reindex,
    mark_obsolete,
    missing_from_index,
    purge_vectors,
    retry_failed,
)
from kip.publications import list_documents, set_publication
from kip.retrieval import SearchScopeError, resolve_index, search
from kip.settings import Settings
from kip.storage import LocalObjectStore
from kip.vectors import EmbedderCache, QdrantIndexFactory, RerankerCache
from rag.answering import AuthorityPolicy, PublicationStatus
from rag.config import ChunkingConfig, IngestionLimits
from rag.ingestion.identity import sha256_hex
from rag.parsing import PARSER_VERSION
from rag.reranking import RerankInputTooLongError
from rag.retrieval import FusionConfig
from rag.sparse import Bm25Config


def create_index(
    session: Session,
    *,
    model_name: str,
    revision: str,
    dimension: int,
    max_seq_tokens: int,
    chunking: ChunkingConfig,
    sparse: Bm25Config | None = None,
) -> IndexVersion:
    if dimension <= 0 or max_seq_tokens < chunking.max_tokens + 2:
        raise ValueError("invalid model dimension or chunk budget exceeds model sequence limit")
    model = session.scalar(
        select(EmbeddingModel).where(
            EmbeddingModel.name == model_name,
            EmbeddingModel.revision == revision,
        )
    )
    if model is None:
        model = EmbeddingModel(
            name=model_name, revision=revision, dimension=dimension, max_seq_tokens=max_seq_tokens
        )
        session.add(model)
        session.flush()
    elif (model.dimension, model.max_seq_tokens) != (dimension, max_seq_tokens):
        raise ValueError("existing model revision has different dimensions or sequence limit")
    material = json.dumps(
        {
            "parser": PARSER_VERSION,
            "chunking": chunking.model_dump(),
            "model": model_name,
            "revision": revision,
            "dimension": dimension,
            "max_seq_tokens": max_seq_tokens,
            "normalize": model.normalize,
            "distance": model.distance,
            # Only present for hybrid indexes, so dense-only fingerprints are unchanged.
            **({"sparse": sparse.model_dump()} if sparse is not None else {}),
        },
        sort_keys=True,
    )
    fingerprint = sha256_hex(material.encode())
    index = session.scalar(select(IndexVersion).where(IndexVersion.fingerprint == fingerprint))
    if index is None:
        index = IndexVersion(
            embedding_model_id=model.id,
            parser_version=PARSER_VERSION,
            chunking_config=chunking.model_dump(),
            chunking_fingerprint=chunking.fingerprint(),
            sparse_config=sparse.model_dump() if sparse is not None else None,
            fingerprint=fingerprint,
            qdrant_collection=f"kip_{fingerprint[:24]}",
        )
        session.add(index)
        session.flush()
    return index


def _bootstrap(sessions: sessionmaker[Session], args: argparse.Namespace) -> dict[str, str]:
    with sessions.begin() as session:
        org = session.scalar(select(Organization).where(Organization.slug == args.slug))
        if org is None:
            org = Organization(slug=args.slug, name=args.name)
            session.add(org)
            session.flush()
        kb = session.scalar(
            select(KnowledgeBase).where(
                KnowledgeBase.org_id == org.id,
                KnowledgeBase.name == args.kb_name,
                KnowledgeBase.deleted_at.is_(None),
            )
        )
        if kb is None:
            kb = KnowledgeBase(org_id=org.id, name=args.kb_name)
            session.add(kb)
            session.flush()
        index = create_index(
            session,
            model_name=args.model,
            revision=args.revision,
            dimension=args.dimension,
            max_seq_tokens=args.max_seq_tokens,
            chunking=ChunkingConfig(),
            sparse=Bm25Config() if args.sparse == "bm25" else None,
        )
        return {"org_id": str(org.id), "kb_id": str(kb.id), "index_version_id": str(index.id)}


def activate_index(
    session: Session, index_version_id: UUID, *, force: bool = False
) -> dict[str, str | int | None]:
    """Make one index version the default for search; the previous one returns to building.

    Refuses (unless forced) while live documents are missing from the target index version,
    because activation would silently hide them from default searches.
    """
    index = session.get(IndexVersion, index_version_id, with_for_update=True)
    if index is None or index.status == "retired":
        raise SystemExit("index version not found or retired")
    missing = missing_from_index(session, index.id)
    if missing and not force:
        raise SystemExit(
            f"{missing} live document(s) are not ready in this index version; run "
            "`kip reindex` and the worker first, or pass --force"
        )
    previous = session.scalar(
        select(IndexVersion).where(IndexVersion.status == "active").with_for_update()
    )
    if previous is not None and previous.id != index.id:
        previous.status = "building"
        session.flush()
    index.status = "active"
    return {
        "active": str(index.id),
        "previous": str(previous.id) if previous else None,
        "missing_documents": missing,
    }


def requeue_processed(session: Session, org_id: UUID) -> int:
    """Queue Phase 2 jobs that stopped at stage=processed so they continue to indexing.

    The worker reuses the persisted chunks (no reparse). Only completed jobs whose document
    index is still building are touched, so published or failed work is never reset.
    """
    building = select(DocumentIndex.document_version_id).where(
        DocumentIndex.org_id == org_id,
        DocumentIndex.index_version_id == IngestionJob.index_version_id,
        DocumentIndex.status == "building",
    )
    result = session.execute(
        update(IngestionJob)
        .where(
            IngestionJob.org_id == org_id,
            IngestionJob.status == "completed",
            IngestionJob.stage == "processed",
            IngestionJob.document_version_id.in_(building),
        )
        .values(
            status="queued",
            stage="queued",
            attempt=0,
            run_after=utcnow(),
            finished_at=None,
            error_code=None,
            error_message=None,
        )
        .execution_options(synchronize_session=False)
    )
    return int(getattr(result, "rowcount", 0) or 0)


def _search(sessions: sessionmaker[Session], settings: Settings, args: argparse.Namespace) -> Any:
    mode = args.mode
    if mode == "auto":
        with sessions() as session:
            mode = "hybrid" if resolve_index(session, args.index).sparse_config else "dense"
    result = search(
        sessions,
        EmbedderCache(settings),
        QdrantIndexFactory(settings),
        org_id=args.org,
        kb_ids=args.kb,
        query=args.query,
        mode=mode,
        top_k=args.top_k,
        index_version_id=args.index,
        overfetch=settings.search_overfetch,
        max_candidates=settings.search_max_candidates,
        hybrid_candidates=args.candidates or settings.hybrid_candidates,
        fusion=FusionConfig(k=args.rrf_k or settings.rrf_k),
        reranker=RerankerCache(settings)() if args.rerank else None,
        rerank_candidates=args.rerank_candidates or settings.rerank_candidates,
        explain=args.explain,
    )
    payload = asdict(result)
    for item in payload["results"]:
        if args.max_chars and len(item["text"]) > args.max_chars:
            item["text"] = item["text"][: args.max_chars] + "..."
    return payload


def _ask(sessions: sessionmaker[Session], settings: Settings, args: argparse.Namespace) -> Any:
    overrides: dict[str, Any] = {
        "mode": args.mode,
        "rerank": args.rerank,
        "top_k": args.top_k,
        "budget_tokens": args.budget,
        "policy": AuthorityPolicy.parse(args.policy) if args.policy else None,
        "store": False if args.no_store else None,
    }
    config = answer_settings(settings, **overrides)
    result = answer(
        sessions,
        EmbedderCache(settings),
        QdrantIndexFactory(settings),
        make_provider(settings),
        org_id=args.org,
        kb_ids=args.kb,
        question=args.question,
        settings=config,
        reranker_factory=RerankerCache(settings),
        index_version_id=args.index,
        include_trace=args.trace,
    )
    return asdict(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap = commands.add_parser("bootstrap", help="Create local org, KB and building index.")
    bootstrap.add_argument("--slug", default="local")
    bootstrap.add_argument("--name", default="Local development")
    bootstrap.add_argument("--kb-name", default="Documents")
    bootstrap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    bootstrap.add_argument(
        "--revision", required=True, help="Use a Hugging Face commit for stability."
    )
    bootstrap.add_argument("--dimension", type=int, default=384)
    bootstrap.add_argument("--max-seq-tokens", type=int, default=256)
    bootstrap.add_argument(
        "--sparse",
        choices=["none", "bm25"],
        default="none",
        help="bm25 creates a hybrid index version (new collection: dense + BM25).",
    )
    submit = commands.add_parser("submit", help="Validate, store and enqueue a document.")
    submit.add_argument("path", type=Path)
    submit.add_argument("--org", type=UUID, required=True)
    submit.add_argument("--kb", type=UUID, required=True)
    submit.add_argument("--index", type=UUID, required=True)
    submit.add_argument("--document", type=UUID, help="Create another version of this document.")
    job = commands.add_parser("job", help="Inspect a job within an organization.")
    job.add_argument("id", type=UUID)
    job.add_argument("--org", type=UUID, required=True)
    search = commands.add_parser("search", help="Search within an organization's KBs.")
    search.add_argument("query")
    search.add_argument("--org", type=UUID, required=True)
    search.add_argument("--kb", type=UUID, action="append", required=True, help="Repeatable.")
    search.add_argument("--index", type=UUID, help="Index version (default: the active one).")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--max-chars", type=int, default=600, help="0 prints full chunk text.")
    search.add_argument(
        "--mode",
        choices=["auto", "dense", "sparse", "hybrid"],
        default="auto",
        help="auto = hybrid when the index version has BM25 vectors, else dense.",
    )
    search.add_argument("--rerank", action="store_true", help="Cross-encoder reranking.")
    search.add_argument("--rerank-candidates", type=int, help="Results reranked (<= 100).")
    search.add_argument("--candidates", type=int, help="Hybrid: visible candidates per retriever.")
    search.add_argument("--rrf-k", type=int, help="Reciprocal rank fusion constant.")
    search.add_argument("--explain", action="store_true", help="Include raw retriever lists.")
    activate = commands.add_parser("activate-index", help="Make an index version the default.")
    activate.add_argument("index", type=UUID)
    activate.add_argument(
        "--force", action="store_true", help="Activate despite missing documents."
    )
    reindex = commands.add_parser(
        "reindex", help="Queue current document versions into another index version."
    )
    reindex.add_argument("--org", type=UUID, required=True)
    reindex.add_argument("--index", type=UUID, required=True)
    retry = commands.add_parser("retry-failed", help="Requeue failed jobs in an organization.")
    retry.add_argument("--org", type=UUID, required=True)
    retry.add_argument("--job", type=UUID, help="Only this job.")
    retry.add_argument("--error-code", help="Only jobs that failed with this code.")
    cleanup = commands.add_parser(
        "cleanup-vectors", help="Remove vectors of deleted documents and superseded versions."
    )
    cleanup.add_argument("--org", type=UUID, required=True)
    requeue = commands.add_parser(
        "requeue-processed", help="Continue Phase 2 processed documents through indexing."
    )
    requeue.add_argument("--org", type=UUID, required=True)
    ask = commands.add_parser("ask", help="Answer a question from an organization's KBs.")
    ask.add_argument("question")
    ask.add_argument("--org", type=UUID, required=True)
    ask.add_argument("--kb", type=UUID, action="append", required=True, help="Repeatable.")
    ask.add_argument("--index", type=UUID, help="Index version (default: the active one).")
    ask.add_argument("--mode", choices=["auto", "dense", "sparse", "hybrid"], default="auto")
    ask.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    ask.add_argument("--top-k", type=int, help="Retrieved results considered as evidence.")
    ask.add_argument("--budget", type=int, help="Evidence token budget (embedding tokens).")
    ask.add_argument(
        "--policy", help="Authority policy, e.g. 'withdrawn=exclude,superseded=demote'."
    )
    ask.add_argument("--trace", action="store_true", help="Include retrieval trace output.")
    ask.add_argument("--no-store", action="store_true", help="Do not persist an answer run.")
    publication = commands.add_parser(
        "set-publication", help="Record a document's publication status (authority)."
    )
    publication.add_argument("--org", type=UUID, required=True)
    publication.add_argument("--document", type=UUID, required=True)
    publication.add_argument(
        "--status", choices=[s.value for s in PublicationStatus], required=True
    )
    publication.add_argument("--published-on", help="YYYY-MM-DD; omit for undated.")
    publication.add_argument("--superseded-by", type=UUID, help="Replacement document ID.")
    publication.add_argument("--note", help="Evidence for the status (max 500 characters).")
    documents = commands.add_parser("documents", help="List documents and publication status.")
    documents.add_argument("--org", type=UUID, required=True)
    documents.add_argument("--kb", type=UUID)
    run = commands.add_parser("answer-run", help="Show a stored answer trace.")
    run.add_argument("id", type=UUID)
    run.add_argument("--org", type=UUID, required=True)
    args = parser.parse_args()
    settings = Settings()
    engine = make_engine(settings.database_url.get_secret_value())
    sessions = session_factory(engine)
    try:
        if args.command == "bootstrap":
            print(json.dumps(_bootstrap(sessions, args)))
        elif args.command == "search":
            try:
                print(json.dumps(_search(sessions, settings, args), default=str, indent=2))
            except (SearchScopeError, RerankInputTooLongError, ValueError) as exc:
                raise SystemExit(f"search refused: {exc}") from None
            except OSError as exc:  # model files missing offline, or download failure
                raise SystemExit(
                    f"search failed: a pinned model could not be loaded ({type(exc).__name__}); "
                    "check network access or warm the Hugging Face cache"
                ) from None
        elif args.command == "ask":
            try:
                print(json.dumps(_ask(sessions, settings, args), default=str, indent=2))
            except (SearchScopeError, RerankInputTooLongError, ValueError) as exc:
                raise SystemExit(f"ask refused: {exc}") from None
            except OSError as exc:
                raise SystemExit(
                    f"ask failed: a pinned model could not be loaded ({type(exc).__name__})"
                ) from None
        elif args.command == "set-publication":
            published = date.fromisoformat(args.published_on) if args.published_on else None
            with sessions.begin() as session:
                try:
                    record = set_publication(
                        session,
                        args.org,
                        args.document,
                        status=PublicationStatus(args.status),
                        published_on=published,
                        superseded_by=args.superseded_by,
                        note=args.note,
                    )
                except (LookupError, ValueError) as exc:
                    raise SystemExit(f"set-publication refused: {exc}") from None
            print(json.dumps(record))
        elif args.command == "documents":
            with sessions() as session:
                print(json.dumps(list_documents(session, args.org, args.kb), indent=2))
        elif args.command == "answer-run":
            with sessions() as session:
                stored = session.scalar(
                    select(AnswerRun).where(AnswerRun.id == args.id, AnswerRun.org_id == args.org)
                )
                if stored is None:
                    raise SystemExit("answer run not found")
                print(
                    json.dumps(
                        {
                            "run_id": str(stored.id),
                            "created_at": stored.created_at.isoformat(),
                            "question": stored.question,
                            "question_sha256": stored.question_sha256,
                            "kb_ids": stored.kb_ids,
                            "index_version_id": str(stored.index_version_id),
                            "mode": stored.mode,
                            "reranked": stored.reranked,
                            "grounding": stored.grounding,
                            "abstain_reason": stored.abstain_reason,
                            "answer": stored.answer,
                            "error_code": stored.error_code,
                            "trace": stored.trace,
                        },
                        indent=2,
                    )
                )
        elif args.command == "activate-index":
            with sessions.begin() as session:
                print(json.dumps(activate_index(session, args.index, force=args.force)))
        elif args.command == "reindex":
            with sessions.begin() as session:
                print(json.dumps({"queued": enqueue_reindex(session, args.org, args.index)}))
        elif args.command == "retry-failed":
            with sessions.begin() as session:
                outcome = retry_failed(
                    session, args.org, job_id=args.job, error_code=args.error_code
                )
            print(json.dumps({"retried": [str(i) for i in outcome.retried],
                              "skipped": outcome.skipped}))  # fmt: skip
        elif args.command == "cleanup-vectors":
            with sessions.begin() as session:
                marked = mark_obsolete(session, args.org)
            purge = purge_vectors(sessions, QdrantIndexFactory(settings), args.org)
            print(json.dumps({"marked": marked, "purged": purge.purged, "failed": purge.failed}))
            if purge.failed:
                raise SystemExit("some vectors could not be removed; run cleanup-vectors again")
        elif args.command == "requeue-processed":
            with sessions.begin() as session:
                print(json.dumps({"requeued": requeue_processed(session, args.org)}))
        elif args.command == "submit":
            limits = IngestionLimits()
            with args.path.open("rb") as stream:
                data = stream.read(limits.max_file_bytes + 1)
            submission = submit_document(
                sessions,
                LocalObjectStore(settings.storage_root),
                org_id=args.org,
                kb_id=args.kb,
                index_version_id=args.index,
                filename=args.path.name,
                data=data,
                document_id=args.document,
                limits=limits,
            )
            print(
                json.dumps(
                    {
                        "document_id": str(submission.document_id),
                        "version_id": str(submission.version_id),
                        "job_id": str(submission.job_id),
                        "duplicate": submission.duplicate,
                    }
                )
            )
        else:
            with sessions() as session:
                record = session.scalar(
                    select(IngestionJob).where(
                        IngestionJob.id == args.id,
                        IngestionJob.org_id == args.org,
                    )
                )
                if record is None:
                    raise SystemExit("job not found")
                print(
                    json.dumps(
                        {
                            "job_id": str(record.id),
                            "status": record.status,
                            "stage": record.stage,
                            "attempt": record.attempt,
                            "max_attempts": record.max_attempts,
                            "error_code": record.error_code,
                        }
                    )
                )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
