import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from kip.cli import create_index
from kip.db.models import Base, KnowledgeBase, Organization
from kip.db.session import make_engine, session_factory
from kip.ingestion import Submission, submit_document
from kip.settings import Settings
from kip.storage import LocalObjectStore
from kip.worker.processor import Processor
from kip.worker.runner import Worker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from rag.chunking.tokens import RegexTokenCounter
from rag.config import ChunkingConfig, IngestionLimits
from rag.ingestion.pipeline import ProcessedDocument, process_document


class InlineProcessor:
    def process(
        self,
        data: bytes,
        filename: str,
        *,
        chunking: ChunkingConfig,
        limits: IngestionLimits,
        pulse: Callable[[], None],
    ) -> ProcessedDocument:
        pulse()
        return process_document(
            data, filename, chunking=chunking, limits=limits, token_counter=RegexTokenCounter()
        )


@pytest.fixture(params=["sqlite", "postgres"])
def sessions(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    if request.param == "sqlite":
        engine = make_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
        Base.metadata.create_all(engine)
        try:
            yield session_factory(engine)
        finally:
            engine.dispose()
        return
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("set TEST_DATABASE_URL to run PostgreSQL migrations/queue tests")
    schema = f"test_{uuid4().hex}"
    admin = create_engine(url)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    try:
        config = Config("alembic.ini")
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        yield session_factory(engine)
    finally:
        engine.dispose()
        with admin.begin() as connection:
            # Only the randomly generated schema created by this fixture is removed.
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@dataclass
class Harness:
    sessions: sessionmaker[Session]
    store: LocalObjectStore
    org: Organization
    kb: KnowledgeBase
    index_id: UUID
    settings: Settings

    def submit(self, data: bytes = b"Evidence for incident response and recovery.") -> Submission:
        return submit_document(
            self.sessions,
            self.store,
            org_id=self.org.id,
            kb_id=self.kb.id,
            index_version_id=self.index_id,
            filename="guide.txt",
            data=data,
        )

    def worker(self, processor: Processor | None = None) -> Worker:
        return Worker(
            self.sessions,
            self.store,
            self.settings,
            processor_factory=lambda _: processor or InlineProcessor(),
        )


@pytest.fixture
def harness(sessions: sessionmaker[Session], tmp_path: Path) -> Harness:
    with sessions.begin() as session:
        org = Organization(slug="test", name="Test org")
        session.add(org)
        session.flush()
        kb = KnowledgeBase(org_id=org.id, name="Knowledge")
        session.add(kb)
        session.flush()
        index = create_index(
            session,
            model_name="regex-v1",
            revision="test-v1",
            dimension=384,
            max_seq_tokens=256,
            chunking=ChunkingConfig(),
        )
    return Harness(
        sessions,
        LocalObjectStore(tmp_path / "objects"),
        org,
        kb,
        index.id,
        Settings(_env_file=None, storage_root=tmp_path / "objects"),
    )
