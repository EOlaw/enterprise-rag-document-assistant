"""Postgres queue with expiring, fenced leases and bounded retry attempts."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from kip.db.models import DocumentVersion, IngestionJob, utcnow


class LeaseLostError(RuntimeError):
    """An expired, cancelled or reclaimed attempt must discard its results."""


@dataclass(frozen=True)
class Lease:
    job_id: UUID
    org_id: UUID
    version_id: UUID
    index_version_id: UUID
    token: UUID
    attempt: int


def claim_job(
    sessions: sessionmaker[Session],
    *,
    lease_seconds: int,
    now: datetime | None = None,
) -> Lease | None:
    now = now or utcnow()
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    with sessions.begin() as session:
        while True:
            job = session.scalar(
                select(IngestionJob)
                .where(
                    or_(
                        and_(
                            IngestionJob.status.in_(["queued", "retrying"]),
                            IngestionJob.run_after <= now,
                        ),
                        and_(
                            IngestionJob.status == "processing",
                            IngestionJob.lease_expires_at <= now,
                        ),
                    )
                )
                .order_by(IngestionJob.run_after, IngestionJob.created_at, IngestionJob.id)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if job is None:
                return None
            if job.attempt >= job.max_attempts:
                job.status = "failed"
                job.stage = "failed"
                job.error_code = "attempts_exhausted"
                job.error_message = "Worker lease expired on the final attempt."
                job.finished_at = now
                job.lease_token = None
                job.lease_expires_at = None
                _version_status(session, job, "failed")
                session.flush()
                continue
            job.attempt += 1
            job.status = "processing"
            job.stage = "processing"
            job.lease_token = uuid4()
            job.heartbeat_at = now
            job.lease_expires_at = now + timedelta(seconds=lease_seconds)
            job.started_at = job.started_at or now
            job.finished_at = None
            job.error_code = None
            job.error_message = None
            _version_status(session, job, "processing")
            return Lease(
                job.id,
                job.org_id,
                job.document_version_id,
                job.index_version_id,
                job.lease_token,
                job.attempt,
            )


def _version_status(session: Session, job: IngestionJob, status: str) -> None:
    # Reprocessing for a new index never downgrades a successfully processed or
    # published version when another index's attempt fails.
    session.execute(
        update(DocumentVersion)
        .where(
            DocumentVersion.id == job.document_version_id,
            DocumentVersion.org_id == job.org_id,
            DocumentVersion.status.not_in(["ready", "superseded", "processed"]),
        )
        .values(status=status)
    )


def owned_job(session: Session, lease: Lease, *, now: datetime | None = None) -> IngestionJob:
    job = session.scalar(
        select(IngestionJob)
        .where(
            IngestionJob.id == lease.job_id,
            IngestionJob.org_id == lease.org_id,
            IngestionJob.lease_token == lease.token,
            IngestionJob.status == "processing",
            IngestionJob.lease_expires_at > (now or utcnow()),
        )
        .with_for_update()
    )
    if job is None:
        raise LeaseLostError("job lease is no longer owned by this attempt")
    return job


def heartbeat(
    sessions: sessionmaker[Session],
    lease: Lease,
    *,
    lease_seconds: int,
    now: datetime | None = None,
) -> None:
    now = now or utcnow()
    with sessions.begin() as session:
        job = owned_job(session, lease, now=now)
        job.heartbeat_at = now
        job.lease_expires_at = now + timedelta(seconds=lease_seconds)


def fail_job(
    sessions: sessionmaker[Session],
    lease: Lease,
    *,
    code: str,
    retryable: bool,
    now: datetime | None = None,
) -> None:
    now = now or utcnow()
    with sessions.begin() as session:
        job = owned_job(session, lease, now=now)
        retry = retryable and job.attempt < job.max_attempts
        job.status = "retrying" if retry else "failed"
        job.stage = job.status
        job.error_code = code
        # Never persist exception text: parsers and storage can include private content.
        job.error_message = "Processing failed; retry scheduled." if retry else "Processing failed."
        job.run_after = now + timedelta(seconds=min(300, 2 ** min(job.attempt, 9)))
        job.finished_at = None if retry else now
        job.lease_token = None
        job.lease_expires_at = None
        _version_status(session, job, "pending" if retry else "failed")
