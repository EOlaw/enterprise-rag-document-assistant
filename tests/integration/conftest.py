import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from kip.cli import create_index
from kip.db.models import Base, EmbeddingModel, IndexVersion, KnowledgeBase, Organization
from kip.db.session import make_engine, session_factory
from kip.ingestion import Submission, submit_document
from kip.retrieval import SearchResult, dense_search
from kip.settings import Settings
from kip.storage import LocalObjectStore
from kip.vectors import VectorIndexFactory
from kip.worker.processor import Processor
from kip.worker.runner import Worker
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from rag.chunking.tokens import RegexTokenCounter
from rag.config import ChunkingConfig, IngestionLimits
from rag.embeddings import Embedder
from rag.indexing import VectorIndex
from rag.indexing.qdrant import QdrantVectorIndex
from rag.ingestion.pipeline import ProcessedDocument, process_document
from tests.fakes import HashingEmbedder


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


@pytest.fixture
def qdrant() -> Iterator[tuple[QdrantClient, str]]:
    """Real Qdrant when TEST_QDRANT_URL is set, otherwise qdrant-client's local mode.

    Each test gets a unique collection prefix; server collections it created are removed.
    """
    url = os.environ.get("TEST_QDRANT_URL")
    client = QdrantClient(url=url, timeout=30) if url else QdrantClient(location=":memory:")
    prefix = f"t{uuid4().hex[:12]}_"
    try:
        yield client, prefix
    finally:
        for collection in client.get_collections().collections:
            if collection.name.startswith(prefix):
                client.delete_collection(collection.name)
        client.close()


@dataclass
class Harness:
    sessions: sessionmaker[Session]
    store: LocalObjectStore
    org: Organization
    kb: KnowledgeBase
    index_id: UUID
    settings: Settings
    client: QdrantClient
    prefix: str
    embedder: HashingEmbedder = field(default_factory=HashingEmbedder)

    def submit(
        self,
        data: bytes = b"Evidence for incident response and recovery.",
        *,
        org_id: UUID | None = None,
        kb_id: UUID | None = None,
        document_id: UUID | None = None,
        filename: str = "guide.txt",
    ) -> Submission:
        return submit_document(
            self.sessions,
            self.store,
            org_id=org_id or self.org.id,
            kb_id=kb_id or self.kb.id,
            index_version_id=self.index_id,
            filename=filename,
            data=data,
            document_id=document_id,
        )

    def vectors(self, index: IndexVersion, model: EmbeddingModel) -> VectorIndex:
        return QdrantVectorIndex(
            self.client, self.prefix + index.qdrant_collection, model.dimension
        )

    def collection(self) -> str:
        with self.sessions() as session:
            index = session.get(IndexVersion, self.index_id)
            assert index is not None
            return self.prefix + index.qdrant_collection

    def worker(
        self,
        processor: Processor | None = None,
        *,
        embedder: Embedder | None = None,
        vector_index_factory: VectorIndexFactory | None = None,
    ) -> Worker:
        return Worker(
            self.sessions,
            self.store,
            self.settings,
            processor_factory=lambda _: processor or InlineProcessor(),
            embedder_factory=lambda _: embedder or self.embedder,
            vector_index_factory=vector_index_factory or self.vectors,
        )

    def search(
        self,
        query: str,
        *,
        org_id: UUID | None = None,
        kb_ids: list[UUID] | None = None,
        top_k: int = 5,
    ) -> SearchResult:
        return dense_search(
            self.sessions,
            lambda _: self.embedder,
            self.vectors,
            org_id=org_id or self.org.id,
            kb_ids=kb_ids or [self.kb.id],
            query=query,
            top_k=top_k,
            index_version_id=self.index_id,
        )

    def add_tenant(self, slug: str) -> tuple[UUID, UUID]:
        with self.sessions.begin() as session:
            org = Organization(slug=slug, name=slug)
            session.add(org)
            session.flush()
            kb = KnowledgeBase(org_id=org.id, name="Knowledge")
            session.add(kb)
            session.flush()
            return org.id, kb.id


@pytest.fixture
def harness(
    sessions: sessionmaker[Session], tmp_path: Path, qdrant: tuple[QdrantClient, str]
) -> Harness:
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
    client, prefix = qdrant
    return Harness(
        sessions,
        LocalObjectStore(tmp_path / "objects"),
        org,
        kb,
        index.id,
        Settings(_env_file=None, storage_root=tmp_path / "objects"),
        client,
        prefix,
    )
