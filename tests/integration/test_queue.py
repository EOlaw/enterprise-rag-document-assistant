from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from kip.db.models import DocumentVersion, IngestionJob, utcnow
from kip.db.queue import LeaseLostError, claim_job, fail_job, heartbeat
from sqlalchemy import select

from tests.integration.conftest import Harness


def test_live_lease_cannot_be_claimed_twice(harness: Harness) -> None:
    harness.submit()
    lease = claim_job(harness.sessions, lease_seconds=60)
    assert lease is not None and lease.attempt == 1
    assert claim_job(harness.sessions, lease_seconds=60) is None


def test_expired_lease_is_fenced_and_reclaimed(harness: Harness) -> None:
    harness.submit()
    now = utcnow()
    first = claim_job(harness.sessions, lease_seconds=60, now=now)
    second = claim_job(harness.sessions, lease_seconds=60, now=now + timedelta(seconds=61))
    assert second.job_id == first.job_id and second.token != first.token
    assert second.attempt == 2
    with pytest.raises(LeaseLostError):
        heartbeat(harness.sessions, first, lease_seconds=60, now=now + timedelta(seconds=62))
    with pytest.raises(LeaseLostError):
        fail_job(
            harness.sessions, first, code="stale", retryable=False, now=now + timedelta(seconds=62)
        )
    with harness.sessions() as session:
        assert session.get(IngestionJob, second.job_id).lease_token == second.token


def test_heartbeat_extends_lease_but_cannot_resurrect_expired_one(harness: Harness) -> None:
    harness.submit()
    now = utcnow()
    lease = claim_job(harness.sessions, lease_seconds=60, now=now)
    heartbeat(harness.sessions, lease, lease_seconds=60, now=now + timedelta(seconds=40))
    assert claim_job(harness.sessions, lease_seconds=60, now=now + timedelta(seconds=80)) is None
    with pytest.raises(LeaseLostError):
        heartbeat(harness.sessions, lease, lease_seconds=60, now=now + timedelta(seconds=101))


def test_crashes_exhaust_attempts_without_infinite_reclaim(harness: Harness) -> None:
    submission = harness.submit()
    now = utcnow()
    for offset in (0, 61, 122):
        assert claim_job(harness.sessions, lease_seconds=60, now=now + timedelta(seconds=offset))
    assert claim_job(harness.sessions, lease_seconds=60, now=now + timedelta(seconds=183)) is None
    with harness.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert job.status == "failed" and job.attempt == job.max_attempts == 3
        assert job.error_code == "attempts_exhausted"
        assert session.get(DocumentVersion, submission.version_id).status == "failed"


def test_retry_backoff_and_attempt_limit(harness: Harness) -> None:
    submission = harness.submit()
    now = utcnow()
    for attempt in (1, 2, 3):
        lease = claim_job(harness.sessions, lease_seconds=60, now=now)
        assert lease.attempt == attempt
        fail_job(harness.sessions, lease, code="transient", retryable=True, now=now)
        if attempt < 3:
            assert claim_job(harness.sessions, lease_seconds=60, now=now) is None
        now += timedelta(seconds=30)
    assert claim_job(harness.sessions, lease_seconds=60, now=now) is None
    with harness.sessions() as session:
        assert session.get(IngestionJob, submission.job_id).status == "failed"


@pytest.mark.postgres
def test_skip_locked_does_not_wait_for_another_worker(harness: Harness) -> None:
    if harness.sessions.kw["bind"].dialect.name != "postgresql":
        pytest.skip("row locking requires PostgreSQL")
    submission = harness.submit()
    with harness.sessions.begin() as session:
        session.scalar(
            select(IngestionJob).where(IngestionJob.id == submission.job_id).with_for_update()
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(claim_job, harness.sessions, lease_seconds=60)
            assert result.result(timeout=5) is None
    assert claim_job(harness.sessions, lease_seconds=60) is not None


@pytest.mark.postgres
def test_parallel_workers_claim_different_jobs(harness: Harness) -> None:
    if harness.sessions.kw["bind"].dialect.name != "postgresql":
        pytest.skip("row locking requires PostgreSQL")
    harness.submit(b"First document.")
    harness.submit(b"Second document.")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(claim_job, harness.sessions, lease_seconds=60) for _ in range(2)]
        leases = [future.result(timeout=5) for future in futures]
    assert len({lease.job_id for lease in leases}) == 2


@pytest.mark.postgres
def test_parallel_duplicate_uploads_get_one_job(harness: Harness) -> None:
    if harness.sessions.kw["bind"].dialect.name != "postgresql":
        pytest.skip("row locking requires PostgreSQL")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(harness.submit) for _ in range(2)]
        submissions = [future.result(timeout=5) for future in futures]
    assert len({item.job_id for item in submissions}) == 1
    assert sorted(item.duplicate for item in submissions) == [False, True]
