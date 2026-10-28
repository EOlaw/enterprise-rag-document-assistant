"""Factories binding index metadata (PostgreSQL) to the embedder and vector store."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING

from kip.db.models import EmbeddingModel, IndexVersion
from kip.settings import Settings
from rag.embeddings import Embedder
from rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag.indexing import VectorIndex
from rag.reranking import Reranker

if TYPE_CHECKING:
    # qdrant_client is imported lazily: the parser subprocess re-imports this module graph.
    from qdrant_client import QdrantClient

EmbedderFactory = Callable[[EmbeddingModel], Embedder]
VectorIndexFactory = Callable[[IndexVersion, EmbeddingModel], VectorIndex]


class EmbedderCache:
    """Loads each pinned model revision once per process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: dict[tuple[str, str, int, int], Embedder] = {}
        self._lock = Lock()

    def __call__(self, model: EmbeddingModel) -> Embedder:
        key = (model.name, model.revision, model.dimension, model.max_seq_tokens)
        with self._lock:
            if key not in self._cache:
                self._cache[key] = SentenceTransformerEmbedder(
                    model.name,
                    model.revision,
                    expected_dimension=model.dimension,
                    expected_max_seq_length=model.max_seq_tokens,
                    batch_size=self.settings.embed_batch_size,
                    device=self.settings.embed_device,
                )
            return self._cache[key]


class QdrantIndexFactory:
    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self.settings = settings
        self._client = client

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            from rag.indexing.qdrant import make_client  # noqa: PLC0415 (lazy, see above)

            key = self.settings.qdrant_api_key
            self._client = make_client(
                self.settings.qdrant_url,
                api_key=key.get_secret_value() if key else None,
                timeout=self.settings.qdrant_timeout_seconds,
            )
        return self._client

    def __call__(self, index: IndexVersion, model: EmbeddingModel) -> VectorIndex:
        from rag.indexing.qdrant import QdrantVectorIndex  # noqa: PLC0415 (lazy, see above)

        return QdrantVectorIndex(
            self.client,
            index.qdrant_collection,
            model.dimension,
            sparse=index.sparse_config is not None,
        )


class RerankerCache:
    """Loads the configured, pinned cross-encoder once per process (only when used)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._reranker: Reranker | None = None
        self._lock = Lock()

    def __call__(self) -> Reranker:
        from rag.reranking.cross_encoder import CrossEncoderReranker  # noqa: PLC0415

        with self._lock:
            if self._reranker is None:
                self._reranker = CrossEncoderReranker(
                    self.settings.rerank_model,
                    self.settings.rerank_revision,
                    expected_max_length=self.settings.rerank_max_length,
                    device=self.settings.embed_device,
                )
            return self._reranker
