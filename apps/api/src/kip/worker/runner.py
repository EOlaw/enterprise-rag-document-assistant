"""One ingestion attempt: parse -> persist chunks -> embed -> index -> publish.

Stages recorded on the job: processing, processed, embedding, indexing, publishing,
published. ``processed`` is an intermediate checkpoint: when a version's chunks are
already persisted for the target index (including documents processed before vector
indexing existed), a later attempt reuses them instead of reparsing.

Publication order (PostgreSQL stays the authority):

1. upsert every vector with ``published=false`` (deterministic ids: retries overwrite);
2. delete stale extra points, verify the exact point count;
3. flip ``published=true`` in Qdrant and verify the count again;
4. in one fenced PostgreSQL transaction mark the document index ``ready`` and advance
   ``documents.current_version_id``.

A crash between steps 3 and 4 leaves published points whose index is still ``building``;
the search re-check excludes them, and the next attempt completes publication.
"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import TypeVar
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from tokenizers import Tokenizer

from kip.db.models import (
    Document,
    DocumentChunk,
    DocumentIndex,
    DocumentVersion,
    EmbeddingModel,
    IndexVersion,
    IngestionJob,
    KnowledgeBase,
    utcnow,
)
from kip.db.queue import Lease, LeaseLostError, claim_job, fail_job, heartbeat, owned_job
from kip.settings import Settings
from kip.storage import ObjectStore, object_key
from kip.vectors import EmbedderCache, EmbedderFactory, QdrantIndexFactory, VectorIndexFactory
from kip.worker.processor import ProcessingError, Processor, SubprocessProcessor
from rag.config import ChunkingConfig, IngestionLimits
from rag.embeddings import check_embedder_matches, validate_vectors
from rag.errors import ConfigurationError, IngestionError
from rag.indexing import ChunkPoint, VectorCountMismatchError, VectorStoreError
from rag.ingestion.identity import chunk_point_id, sha256_hex
from rag.ingestion.pipeline import ProcessedDocument
from rag.parsing import PARSER_VERSION

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class Work:
    version: DocumentVersion
    index: IndexVersion
    model: EmbeddingModel


@dataclass(frozen=True)
class ChunkRow:
    id: UUID
    chunk_index: int
    embedding_text: str
    document_id: UUID
    document_version_id: UUID
    kb_id: UUID


class Worker:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        store: ObjectStore,
        settings: Settings,
        *,
        processor_factory: Callable[[EmbeddingModel], Processor] | None = None,
        embedder_factory: EmbedderFactory | None = None,
        vector_index_factory: VectorIndexFactory | None = None,
        limits: IngestionLimits | None = None,
    ) -> None:
        self.sessions = sessions
        self.store = store
        self.settings = settings
        self.limits = limits or IngestionLimits()
        self.processor_factory = processor_factory or self._processor
        self.embedder_factory = embedder_factory or EmbedderCache(settings)
        self.vector_index_factory = vector_index_factory or QdrantIndexFactory(settings)
        self.heartbeat_seconds = min(5.0, settings.lease_seconds / 3)
        self._processors: dict[tuple[str, str, int], Processor] = {}

    def _processor(self, model: EmbeddingModel) -> Processor:
        key = (model.name, model.revision, model.max_seq_tokens)
        if key not in self._processors:
            # Pin the tokenizer to the same revision recorded in the index identity.
            tokenizer = Tokenizer.from_pretrained(model.name, revision=model.revision)
            self._processors[key] = SubprocessProcessor(
                tokenizer_json=tokenizer.to_str(),
                tokenizer_name=model.name,
                max_seq_tokens=model.max_seq_tokens,
                timeout_seconds=self.settings.parse_timeout_seconds,
                memory_mb=self.settings.parse_memory_mb,
                heartbeat_seconds=self.heartbeat_seconds,
            )
        return self._processors[key]

    def _work(self, lease: Lease) -> Work:
        with self.sessions() as session:
            version = session.get(DocumentVersion, lease.version_id)
            index = session.get(IndexVersion, lease.index_version_id)
            if version is None or index is None or version.org_id != lease.org_id:
                raise ProcessingError("missing_work", retryable=False)
            model = session.get(EmbeddingModel, index.embedding_model_id)
            if model is None or index.parser_version != PARSER_VERSION:
                raise ProcessingError("incompatible_index", retryable=False)
            if index.status == "retired":
                raise ProcessingError("retired_index", retryable=False)
            return Work(version, index, model)

    # ------------------------------------------------------------------ main loop

    def run_once(self) -> bool:
        lease = claim_job(self.sessions, lease_seconds=self.settings.lease_seconds)
        if lease is None:
            return False
        try:
            work = self._work(lease)
            status, document_index_id = self._existing_index(lease, work)
            if status == "ready":
                self._finish(lease, stage="published")
                return True
            if document_index_id is None:
                document_index_id = self._process(lease, work)
            self._index_and_publish(lease, work, document_index_id)
        except LeaseLostError:
            # A newer attempt or cancellation owns the state. Discard this result.
            pass
        except IngestionError as exc:
            self._fail(lease, exc.code, retryable=exc.retryable)
        except ConfigurationError:
            # e.g. an unpinned model revision: retrying cannot succeed.
            self._fail(lease, "configuration_error", retryable=False)
        except (OSError, OperationalError):
            self._fail(lease, "infrastructure_error", retryable=True)
        except Exception:
            # Keep the loop alive, bound retries and exclude private exception details.
            logger.exception("unexpected worker error", extra={"job_id": str(lease.job_id)})
            self._fail(lease, "worker_error", retryable=True)
        return True

    def _pulse(self, lease: Lease) -> None:
        heartbeat(self.sessions, lease, lease_seconds=self.settings.lease_seconds)

    def _stage(self, lease: Lease, stage: str) -> None:
        with self.sessions.begin() as session:
            job = owned_job(session, lease)
            job.stage = stage
        self._pulse(lease)

    def _with_pulse(self, lease: Lease, operation: Callable[[], T]) -> T:
        """Run a blocking call (model load, embedding batch) while keeping the lease alive."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(operation)
            while True:
                try:
                    return future.result(timeout=self.heartbeat_seconds)
                except FutureTimeout:
                    self._pulse(lease)  # raises LeaseLostError if fenced out

    def _fail(self, lease: Lease, code: str, *, retryable: bool) -> None:
        with suppress(LeaseLostError):
            fail_job(self.sessions, lease, code=code, retryable=retryable)
        # A database outage may prevent recording failure. Leave the lease to expire;
        # the supervisor retries claiming after connectivity returns.

    def _finish(self, lease: Lease, *, stage: str) -> None:
        with self.sessions.begin() as session:
            job = owned_job(session, lease)
            _complete(job, stage)

    # ------------------------------------------------------- processing checkpoint

    def _existing_index(self, lease: Lease, work: Work) -> tuple[str | None, UUID | None]:
        """Return ("ready", id), ("processed", id) for reusable chunks, or (None, None)."""
        with self.sessions() as session:
            document_index = session.scalar(
                select(DocumentIndex).where(
                    DocumentIndex.document_version_id == lease.version_id,
                    DocumentIndex.index_version_id == lease.index_version_id,
                    DocumentIndex.org_id == lease.org_id,
                )
            )
            if document_index is None:
                return None, None
            if document_index.status == "ready":
                return "ready", document_index.id
            stored = session.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.document_index_id == document_index.id)
            )
            version = session.get(DocumentVersion, lease.version_id)
            reusable = (
                document_index.status == "building"
                and version is not None
                and version.canonical_text_key is not None
                and version.parser_version == work.index.parser_version
                and document_index.chunk_count > 0
                and stored == document_index.chunk_count
            )
            return ("processed", document_index.id) if reusable else (None, None)

    def _process(self, lease: Lease, work: Work) -> UUID:
        data = self.store.get(work.version.storage_key, max_bytes=self.limits.max_file_bytes)
        if len(data) != work.version.byte_size or sha256_hex(data) != work.version.raw_sha256:
            raise ProcessingError("object_integrity", retryable=False)
        processor = self.processor_factory(work.model)
        self._pulse(lease)
        result = processor.process(
            data,
            work.version.filename,
            chunking=ChunkingConfig.model_validate(work.index.chunking_config),
            limits=self.limits,
            pulse=lambda: self._pulse(lease),
        )
        self._pulse(lease)
        canonical_key = object_key(
            lease.org_id, lease.version_id, f"canonical-{result.text_sha256}.txt"
        )
        self.store.put(canonical_key, result.canonical_text.encode("utf-8"))
        return self._persist(lease, work, result, canonical_key)

    def _persist(
        self,
        lease: Lease,
        work: Work,
        result: ProcessedDocument,
        canonical_key: str,
    ) -> UUID:
        if (
            result.raw_sha256 != work.version.raw_sha256
            or result.parser_version != work.index.parser_version
            or result.chunking_fingerprint != work.index.chunking_fingerprint
        ):
            raise ProcessingError("processing_identity_mismatch", retryable=False)
        with self.sessions.begin() as session:
            job = owned_job(session, lease)
            document = self._live_document(session, lease, work)
            version = session.get(DocumentVersion, lease.version_id)
            assert version is not None
            if document.title == version.filename and result.title:
                # Submission stores the filename; adopt the parsed (PDF/DOCX/front-matter)
                # title unless someone already set a title explicitly.
                document.title = result.title
            document_index = session.scalar(
                select(DocumentIndex)
                .where(
                    DocumentIndex.document_version_id == lease.version_id,
                    DocumentIndex.index_version_id == lease.index_version_id,
                    DocumentIndex.org_id == lease.org_id,
                )
                .with_for_update()
            )
            if document_index is None:
                document_index = DocumentIndex(
                    org_id=lease.org_id,
                    document_version_id=lease.version_id,
                    index_version_id=lease.index_version_id,
                )
                session.add(document_index)
                session.flush()
            if document_index.status != "building":
                raise ProcessingError("index_not_building", retryable=False)
            session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_index_id == document_index.id,
                    DocumentChunk.org_id == lease.org_id,
                )
            )
            for chunk in result.chunks:
                if result.canonical_text[chunk.char_start : chunk.char_end] != chunk.text:
                    raise ProcessingError("invalid_chunk_offsets", retryable=False)
                session.add(
                    DocumentChunk(
                        id=chunk_point_id(lease.version_id, work.index.fingerprint, chunk.index),
                        org_id=lease.org_id,
                        document_index_id=document_index.id,
                        document_id=version.document_id,
                        document_version_id=version.id,
                        chunk_index=chunk.index,
                        text=chunk.text,
                        embedding_text=chunk.embedding_text,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        heading_path=list(chunk.heading_path),
                        token_count=chunk.token_count,
                        content_hash=chunk.content_hash,
                    )
                )
            document_index.chunk_count = len(result.chunks)
            version.canonical_text_key = canonical_key
            version.text_sha256 = result.text_sha256
            version.page_count = result.page_count
            version.parser_version = result.parser_version
            version.warnings = result.warnings
            if version.status not in {"ready", "superseded"}:
                version.status = "processed"
            # Checkpoint only: the job keeps its lease and continues to vector indexing.
            job.stage = "processed"
            return document_index.id

    @staticmethod
    def _live_document(session: Session, lease: Lease, work: Work) -> Document:
        document = session.scalar(
            select(Document)
            .join(KnowledgeBase, KnowledgeBase.id == Document.kb_id)
            .where(
                Document.id == work.version.document_id,
                Document.org_id == lease.org_id,
                Document.deleted_at.is_(None),
                KnowledgeBase.deleted_at.is_(None),
            )
            .with_for_update(of=Document)
        )
        if document is None:
            raise ProcessingError("document_deleted", retryable=False)
        return document

    # ------------------------------------------------------ embedding and indexing

    def _chunk_rows(self, lease: Lease, document_index_id: UUID) -> list[ChunkRow]:
        with self.sessions() as session:
            rows = session.execute(
                select(
                    DocumentChunk.id,
                    DocumentChunk.chunk_index,
                    DocumentChunk.embedding_text,
                    DocumentChunk.document_id,
                    DocumentChunk.document_version_id,
                    Document.kb_id,
                )
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(
                    DocumentChunk.document_index_id == document_index_id,
                    DocumentChunk.org_id == lease.org_id,
                    Document.org_id == lease.org_id,
                )
                .order_by(DocumentChunk.chunk_index)
            ).all()
            return [ChunkRow(*row) for row in rows]

    def _index_and_publish(self, lease: Lease, work: Work, document_index_id: UUID) -> None:
        self._stage(lease, "embedding")
        embedder = self._with_pulse(lease, partial(self.embedder_factory, work.model))
        check_embedder_matches(
            embedder,
            model_id=work.model.name,
            revision=work.model.revision,
            dimension=work.model.dimension,
            max_seq_length=work.model.max_seq_tokens,
        )
        vectors = self.vector_index_factory(work.index, work.model)
        self._with_pulse(lease, vectors.ensure_collection)

        rows = self._chunk_rows(lease, document_index_id)
        if not rows:
            raise ProcessingError("no_chunks", retryable=False)
        for row in rows:
            if row.id != chunk_point_id(lease.version_id, work.index.fingerprint, row.chunk_index):
                raise ProcessingError("chunk_identity_mismatch", retryable=False)

        batch_size = self.settings.embed_batch_size
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [row.embedding_text for row in batch]
            embedded = self._with_pulse(lease, partial(embedder.embed_documents, texts))
            validate_vectors(embedded, count=len(batch), dimension=work.model.dimension)
            if start == 0:
                self._stage(lease, "indexing")
            vectors.upsert(
                [
                    ChunkPoint(
                        id=row.id,
                        vector=vector,
                        org_id=lease.org_id,
                        kb_id=row.kb_id,
                        document_id=row.document_id,
                        document_version_id=row.document_version_id,
                        document_index_id=document_index_id,
                        chunk_index=row.chunk_index,
                    )
                    for row, vector in zip(batch, embedded, strict=True)
                ]
            )
            self._pulse(lease)

        vectors.delete_extra_chunks(document_index_id, chunk_count=len(rows))
        if (written := vectors.count(document_index_id)) != len(rows):
            raise VectorCountMismatchError(f"expected {len(rows)} points, found {written}")
        self._stage(lease, "publishing")
        vectors.set_published(document_index_id, True)
        if (visible := vectors.count(document_index_id, published=True)) != len(rows):
            raise VectorCountMismatchError(f"expected {len(rows)} published, found {visible}")

        try:
            hidden = self._commit_publication(lease, work, document_index_id, len(rows))
        except ProcessingError as exc:
            if exc.code == "document_deleted":
                with suppress(VectorStoreError):
                    vectors.delete_document_index(document_index_id)
            raise
        for stale in hidden:
            # Optimization only: these versions are already invisible via PostgreSQL.
            with suppress(VectorStoreError):
                vectors.set_published(stale, False)

    def _commit_publication(
        self, lease: Lease, work: Work, document_index_id: UUID, chunk_count: int
    ) -> list[UUID]:
        """Fenced PostgreSQL publication. Returns document indexes that became invisible."""
        with self.sessions.begin() as session:
            job = owned_job(session, lease)
            document = self._live_document(session, lease, work)
            version = session.get(DocumentVersion, lease.version_id)
            document_index = session.scalar(
                select(DocumentIndex)
                .where(
                    DocumentIndex.id == document_index_id,
                    DocumentIndex.org_id == lease.org_id,
                    DocumentIndex.document_version_id == lease.version_id,
                    DocumentIndex.index_version_id == lease.index_version_id,
                )
                .with_for_update()
            )
            if version is None or document_index is None:
                raise ProcessingError("missing_work", retryable=False)
            stored = session.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.document_index_id == document_index_id)
            )
            if not chunk_count == stored == document_index.chunk_count:
                raise ProcessingError("chunk_count_changed", retryable=True)
            if document_index.status != "building":
                raise ProcessingError("index_not_building", retryable=False)

            now = utcnow()
            document_index.status = "ready"
            document_index.published_at = now
            hidden: list[UUID] = []
            current = (
                session.get(DocumentVersion, document.current_version_id)
                if document.current_version_id is not None
                else None
            )
            if current is None or current.id == version.id:
                document.current_version_id = version.id
                version.status = "ready"
            elif version.version_number > current.version_number:
                document.current_version_id = version.id
                version.status = "ready"
                current.status = "superseded"
                hidden.extend(
                    session.scalars(
                        select(DocumentIndex.id).where(
                            DocumentIndex.document_version_id == current.id,
                            DocumentIndex.index_version_id == lease.index_version_id,
                            DocumentIndex.status == "ready",
                        )
                    )
                )
            else:
                # An older version finished after a newer one became current: index it,
                # but never move the document backwards.
                version.status = "superseded"
                hidden.append(document_index.id)
            _complete(job, "published")
            return hidden


def _complete(job: IngestionJob, stage: str) -> None:
    job.status = "completed"
    job.stage = stage
    job.finished_at = utcnow()
    job.lease_token = None
    job.lease_expires_at = None
