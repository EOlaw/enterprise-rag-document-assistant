"""One processing attempt, with fenced atomic persistence of citation-ready chunks."""

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

from sqlalchemy import delete, select
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
    KnowledgeBase,
    utcnow,
)
from kip.db.queue import Lease, LeaseLostError, claim_job, fail_job, heartbeat, owned_job
from kip.settings import Settings
from kip.storage import ObjectStore, object_key
from kip.worker.processor import ProcessingError, Processor, SubprocessProcessor
from rag.config import ChunkingConfig, IngestionLimits
from rag.errors import IngestionError
from rag.ingestion.identity import chunk_point_id, sha256_hex
from rag.ingestion.pipeline import ProcessedDocument
from rag.parsing import PARSER_VERSION


@dataclass(frozen=True)
class Work:
    version: DocumentVersion
    index: IndexVersion
    model: EmbeddingModel


class Worker:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        store: ObjectStore,
        settings: Settings,
        *,
        processor_factory: Callable[[EmbeddingModel], Processor] | None = None,
        limits: IngestionLimits | None = None,
    ) -> None:
        self.sessions = sessions
        self.store = store
        self.settings = settings
        self.limits = limits or IngestionLimits()
        self.processor_factory = processor_factory or self._processor
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
                heartbeat_seconds=min(5, self.settings.lease_seconds / 3),
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

    def run_once(self) -> bool:
        lease = claim_job(self.sessions, lease_seconds=self.settings.lease_seconds)
        if lease is None:
            return False
        try:
            work = self._work(lease)
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
            self._persist(lease, work, result, canonical_key)
        except LeaseLostError:
            # A newer attempt or cancellation owns the state. Discard this result.
            pass
        except IngestionError as exc:
            self._fail(lease, exc.code, retryable=exc.retryable)
        except (OSError, OperationalError):
            self._fail(lease, "infrastructure_error", retryable=True)
        except Exception:
            # Keep the loop alive, bound retries and exclude private exception details.
            self._fail(lease, "worker_error", retryable=True)
        return True

    def _pulse(self, lease: Lease) -> None:
        heartbeat(self.sessions, lease, lease_seconds=self.settings.lease_seconds)

    def _fail(self, lease: Lease, code: str, *, retryable: bool) -> None:
        with suppress(LeaseLostError):
            fail_job(self.sessions, lease, code=code, retryable=retryable)
        # A database outage may prevent recording failure. Leave the lease to expire;
        # the supervisor retries claiming after connectivity returns.

    def _persist(
        self,
        lease: Lease,
        work: Work,
        result: ProcessedDocument,
        canonical_key: str,
    ) -> None:
        if (
            result.raw_sha256 != work.version.raw_sha256
            or result.parser_version != work.index.parser_version
            or result.chunking_fingerprint != work.index.chunking_fingerprint
        ):
            raise ProcessingError("processing_identity_mismatch", retryable=False)
        with self.sessions.begin() as session:
            job = owned_job(session, lease)
            document = session.scalar(
                select(Document)
                .join(
                    KnowledgeBase,
                    KnowledgeBase.id == Document.kb_id,
                )
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
            version = session.get(DocumentVersion, lease.version_id)
            assert version is not None
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
            job.status = "completed"
            job.stage = "processed"
            job.finished_at = utcnow()
            job.lease_token = None
            job.lease_expires_at = None
            # No current_version_id / published_at / ready transition until vectors
            # have been written and published by the Phase 3 indexing implementation.
