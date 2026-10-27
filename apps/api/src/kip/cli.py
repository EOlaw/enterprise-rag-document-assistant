"""Trusted local administration commands; no public API or user authorization yet."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from kip.db.models import (
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
from kip.retrieval import SearchScopeError, dense_search
from kip.settings import Settings
from kip.storage import LocalObjectStore
from kip.vectors import EmbedderCache, QdrantIndexFactory
from rag.config import ChunkingConfig, IngestionLimits
from rag.ingestion.identity import sha256_hex
from rag.parsing import PARSER_VERSION


def create_index(
    session: Session,
    *,
    model_name: str,
    revision: str,
    dimension: int,
    max_seq_tokens: int,
    chunking: ChunkingConfig,
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
        )
        return {"org_id": str(org.id), "kb_id": str(kb.id), "index_version_id": str(index.id)}


def activate_index(session: Session, index_version_id: UUID) -> dict[str, str | None]:
    """Make one index version the default for search; the previous one returns to building."""
    index = session.get(IndexVersion, index_version_id, with_for_update=True)
    if index is None or index.status == "retired":
        raise SystemExit("index version not found or retired")
    previous = session.scalar(
        select(IndexVersion).where(IndexVersion.status == "active").with_for_update()
    )
    if previous is not None and previous.id != index.id:
        previous.status = "building"
        session.flush()
    index.status = "active"
    return {"active": str(index.id), "previous": str(previous.id) if previous else None}


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
    result = dense_search(
        sessions,
        EmbedderCache(settings),
        QdrantIndexFactory(settings),
        org_id=args.org,
        kb_ids=args.kb,
        query=args.query,
        top_k=args.top_k,
        index_version_id=args.index,
        overfetch=settings.search_overfetch,
    )
    payload = asdict(result)
    for item in payload["results"]:
        if args.max_chars and len(item["text"]) > args.max_chars:
            item["text"] = item["text"][: args.max_chars] + "..."
    return payload


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
    submit = commands.add_parser("submit", help="Validate, store and enqueue a document.")
    submit.add_argument("path", type=Path)
    submit.add_argument("--org", type=UUID, required=True)
    submit.add_argument("--kb", type=UUID, required=True)
    submit.add_argument("--index", type=UUID, required=True)
    submit.add_argument("--document", type=UUID, help="Create another version of this document.")
    job = commands.add_parser("job", help="Inspect a job within an organization.")
    job.add_argument("id", type=UUID)
    job.add_argument("--org", type=UUID, required=True)
    search = commands.add_parser("search", help="Dense search within an organization's KBs.")
    search.add_argument("query")
    search.add_argument("--org", type=UUID, required=True)
    search.add_argument("--kb", type=UUID, action="append", required=True, help="Repeatable.")
    search.add_argument("--index", type=UUID, help="Index version (default: the active one).")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--max-chars", type=int, default=600, help="0 prints full chunk text.")
    activate = commands.add_parser("activate-index", help="Make an index version the default.")
    activate.add_argument("index", type=UUID)
    requeue = commands.add_parser(
        "requeue-processed", help="Continue Phase 2 processed documents through indexing."
    )
    requeue.add_argument("--org", type=UUID, required=True)
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
            except (SearchScopeError, ValueError) as exc:
                raise SystemExit(f"search refused: {exc}") from None
        elif args.command == "activate-index":
            with sessions.begin() as session:
                print(json.dumps(activate_index(session, args.index)))
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
