from dataclasses import replace
from datetime import timedelta
from uuid import uuid4

import pytest
from kip.cli import create_index
from kip.db.models import (
    Document,
    DocumentChunk,
    DocumentIndex,
    DocumentVersion,
    IngestionJob,
    Organization,
    utcnow,
)
from kip.db.queue import claim_job
from kip.ingestion import submit_document
from kip.worker.processor import ProcessingError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from rag.config import ChunkingConfig
from rag.ingestion.identity import sha256_hex
from tests.integration.conftest import Harness, InlineProcessor


def test_roundtrip_preserves_offsets_and_keeps_publication_closed(harness: Harness) -> None:
    submission = harness.submit(("A sentence with useful recovery evidence. " * 130).encode())
    worker = harness.worker()
    assert worker.run_once()
    assert not worker.run_once()
    with harness.sessions() as session:
        version = session.get(DocumentVersion, submission.version_id)
        job = session.get(IngestionJob, submission.job_id)
        document = session.get(Document, submission.document_id)
        index = session.scalar(select(DocumentIndex))
        chunks = session.scalars(select(DocumentChunk).order_by(DocumentChunk.chunk_index)).all()
        assert version.status == "processed"
        assert job.status == "completed" and job.stage == "processed"
        assert document.current_version_id is None
        assert index.status == "building" and index.published_at is None
        assert index.chunk_count == len(chunks) > 1
        canonical = harness.store.get(version.canonical_text_key, max_bytes=1_000_000).decode()
        assert sha256_hex(canonical.encode()) == version.text_sha256
        for chunk in chunks:
            assert chunk.text == canonical[chunk.char_start : chunk.char_end]
            assert chunk.content_hash == sha256_hex(chunk.embedding_text.encode())
            assert chunk.token_count <= 200


def test_duplicate_submission_reuses_document_version_and_job(harness: Harness) -> None:
    first = harness.submit()
    duplicate = harness.submit()
    assert duplicate.duplicate
    assert (duplicate.document_id, duplicate.version_id, duplicate.job_id) == (
        first.document_id,
        first.version_id,
        first.job_id,
    )
    with harness.sessions() as session:
        for model in (Document, DocumentVersion, IngestionJob):
            assert session.scalar(select(func.count()).select_from(model)) == 1


def test_new_version_and_new_index_are_distinct_jobs(harness: Harness) -> None:
    first = harness.submit()
    with harness.sessions.begin() as session:
        other_index = create_index(
            session,
            model_name="regex-v1",
            revision="test-v1",
            dimension=384,
            max_seq_tokens=256,
            chunking=ChunkingConfig(max_tokens=128),
        )
    second = submit_document(
        harness.sessions,
        harness.store,
        org_id=harness.org.id,
        kb_id=harness.kb.id,
        index_version_id=harness.index_id,
        document_id=first.document_id,
        filename="guide.txt",
        data=b"A revised document with new evidence.",
    )
    reindex = submit_document(
        harness.sessions,
        harness.store,
        org_id=harness.org.id,
        kb_id=harness.kb.id,
        index_version_id=other_index.id,
        document_id=first.document_id,
        filename="guide.txt",
        data=b"Evidence for incident response and recovery.",
    )
    assert second.version_id != first.version_id
    assert reindex.version_id == first.version_id and reindex.job_id != first.job_id
    worker = harness.worker()
    assert all(worker.run_once() for _ in range(3))
    with harness.sessions() as session:
        assert session.get(DocumentVersion, second.version_id).version_number == 2
        assert session.scalar(select(func.count()).select_from(DocumentIndex)) == 3


def test_storage_failure_rolls_back_document_version_and_job(harness: Harness, monkeypatch) -> None:
    def broken_put(_key: str, _data: bytes) -> None:
        raise OSError("private storage detail")

    monkeypatch.setattr(harness.store, "put", broken_put)
    with pytest.raises(OSError):
        harness.submit()
    with harness.sessions() as session:
        for model in (Document, DocumentVersion, IngestionJob):
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_tenant_and_document_ancestry_enforced_by_database(harness: Harness) -> None:
    submission = harness.submit()
    other = harness.submit(b"Different bytes make a different document.")
    with harness.sessions.begin() as session:
        org = Organization(slug="other", name="Other tenant")
        session.add(org)
        session.flush()
        org_id = org.id
    with pytest.raises(IntegrityError), harness.sessions.begin() as session:
        session.add(Document(org_id=org_id, kb_id=harness.kb.id, title="Cross tenant"))
    with pytest.raises(IntegrityError), harness.sessions.begin() as session:
        document = session.get(Document, submission.document_id)
        document.current_version_id = other.version_id
    with pytest.raises(LookupError):
        submit_document(
            harness.sessions,
            harness.store,
            org_id=org_id,
            kb_id=harness.kb.id,
            index_version_id=harness.index_id,
            filename="guide.txt",
            data=b"hello",
        )


def test_job_tenant_reference_is_enforced(harness: Harness) -> None:
    submission = harness.submit()
    with harness.sessions.begin() as session:
        org = Organization(slug="other", name="Other")
        session.add(org)
        session.flush()
        org_id = org.id
    with pytest.raises(IntegrityError), harness.sessions.begin() as session:
        session.add(
            IngestionJob(
                org_id=org_id,
                document_version_id=submission.version_id,
                index_version_id=harness.index_id,
                idempotency_key="wrong",
            )
        )


def test_retry_replaces_chunks_with_same_ids(harness: Harness) -> None:
    submission = harness.submit()
    worker = harness.worker()
    assert worker.run_once()
    with harness.sessions.begin() as session:
        ids = set(session.scalars(select(DocumentChunk.id)))
        job = session.get(IngestionJob, submission.job_id)
        job.status = "queued"
    assert worker.run_once()
    with harness.sessions() as session:
        assert set(session.scalars(select(DocumentChunk.id))) == ids
        assert session.scalar(select(func.count()).select_from(DocumentIndex)) == 1


def test_tampered_original_fails_without_persisting_chunks(harness: Harness) -> None:
    submission = harness.submit()
    with harness.sessions() as session:
        key = session.get(DocumentVersion, submission.version_id).storage_key
    harness.store.put(key, b"tampered original")
    assert harness.worker().run_once()
    with harness.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert job.status == "failed" and job.error_code == "object_integrity"
        assert session.scalar(select(func.count()).select_from(DocumentChunk)) == 0


def test_empty_document_is_terminal_not_retried(harness: Harness) -> None:
    submission = harness.submit(b"    \n   ")
    assert harness.worker().run_once()
    with harness.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert job.status == "failed" and job.error_code == "empty_document"
        assert job.attempt == 1


def test_transient_failure_retries_without_private_error_text(harness: Harness) -> None:
    class BrokenProcessor(InlineProcessor):
        def process(self, *args, **kwargs):
            raise OSError("secret document and credential details")

    submission = harness.submit()
    assert harness.worker(BrokenProcessor()).run_once()
    with harness.sessions.begin() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert job.status == "retrying" and job.error_code == "infrastructure_error"
        assert "secret" not in job.error_message
        job.run_after = utcnow() - timedelta(seconds=1)
    assert harness.worker().run_once()
    with harness.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert job.status == "completed" and job.attempt == 2
        assert job.error_code is None and job.error_message is None


def test_stale_attempt_cannot_complete_or_fail_new_attempt(harness: Harness) -> None:
    submission = harness.submit()

    class ReclaimedProcessor(InlineProcessor):
        def process(self, *args, **kwargs):
            with harness.sessions.begin() as session:
                job = session.get(IngestionJob, submission.job_id)
                job.lease_expires_at = utcnow() - timedelta(seconds=1)
            assert claim_job(harness.sessions, lease_seconds=300).attempt == 2
            # No heartbeat here: simulate results arriving from a stalled process.
            kwargs["pulse"] = lambda: None
            return super().process(*args, **kwargs)

    assert harness.worker(ReclaimedProcessor()).run_once()
    with harness.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert job.attempt == 2 and job.status == "processing"
        assert session.scalar(select(func.count()).select_from(DocumentChunk)) == 0


def test_soft_deleted_document_cannot_be_completed(harness: Harness) -> None:
    submission = harness.submit()
    with harness.sessions.begin() as session:
        session.get(Document, submission.document_id).deleted_at = utcnow()
    assert harness.worker().run_once()
    with harness.sessions() as session:
        assert session.get(IngestionJob, submission.job_id).error_code == "document_deleted"
        assert session.scalar(select(func.count()).select_from(DocumentIndex)) == 0


def test_cancelled_attempt_discards_results(harness: Harness) -> None:
    submission = harness.submit()

    class CancelledProcessor(InlineProcessor):
        def process(self, *args, **kwargs):
            with harness.sessions.begin() as session:
                session.get(IngestionJob, submission.job_id).status = "cancelled"
            return super().process(*args, **kwargs)

    assert harness.worker(CancelledProcessor()).run_once()
    with harness.sessions() as session:
        assert session.get(IngestionJob, submission.job_id).status == "cancelled"
        assert session.scalar(select(func.count()).select_from(DocumentChunk)) == 0


def test_failed_persistence_rolls_back_partial_chunks(harness: Harness) -> None:
    submission = harness.submit()

    class BadOffsetsProcessor(InlineProcessor):
        def process(self, *args, **kwargs):
            result = super().process(*args, **kwargs)
            # A second invalid chunk ensures the first insertion is also rolled back.
            result.chunks.append(replace(result.chunks[0], index=1, text="wrong slice"))
            return result

    assert harness.worker(BadOffsetsProcessor()).run_once()
    with harness.sessions() as session:
        assert session.get(IngestionJob, submission.job_id).error_code == "invalid_chunk_offsets"
        assert session.scalar(select(func.count()).select_from(DocumentChunk)) == 0
        assert session.scalar(select(func.count()).select_from(DocumentIndex)) == 0


def test_unknown_target_fails_without_writing_objects(harness: Harness) -> None:
    with pytest.raises(LookupError):
        submit_document(
            harness.sessions,
            harness.store,
            org_id=harness.org.id,
            kb_id=harness.kb.id,
            index_version_id=uuid4(),
            filename="guide.txt",
            data=b"some evidence",
        )
    assert list(harness.store.root.iterdir()) == []


def test_nonretryable_processing_error_keeps_stable_code(harness: Harness) -> None:
    class BrokenProcessor(InlineProcessor):
        def process(self, *args, **kwargs):
            raise ProcessingError("parser_timeout", retryable=False)

    submission = harness.submit()
    harness.worker(BrokenProcessor()).run_once()
    with harness.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert job.status == "failed" and job.error_code == "parser_timeout"


def test_chunk_cannot_point_at_another_documents_version(harness: Harness) -> None:
    harness.submit()
    other = harness.submit(b"Another document in the same tenant.")
    worker = harness.worker()
    assert worker.run_once()
    assert worker.run_once()
    with harness.sessions() as session:
        chunk_id = session.scalar(
            select(DocumentChunk.id).where(
                DocumentChunk.document_id != other.document_id,
            )
        )
    with pytest.raises(IntegrityError), harness.sessions.begin() as session:
        chunk = session.get(DocumentChunk, chunk_id)
        chunk.document_id = other.document_id
    with pytest.raises(IntegrityError), harness.sessions.begin() as session:
        chunk = session.get(DocumentChunk, chunk_id)
        chunk.document_version_id = other.version_id
