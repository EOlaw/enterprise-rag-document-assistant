"""Vector index contract. Import the Qdrant adapter from ``rag.indexing.qdrant``.

This package deliberately does not import ``qdrant_client`` (see ``base.py``).
"""

from rag.indexing.base import (
    ChunkPoint,
    VectorCollectionMismatchError,
    VectorCountMismatchError,
    VectorHit,
    VectorIndex,
    VectorStoreError,
    VectorStoreUnavailableError,
)

__all__ = [
    "ChunkPoint",
    "VectorCollectionMismatchError",
    "VectorCountMismatchError",
    "VectorHit",
    "VectorIndex",
    "VectorStoreError",
    "VectorStoreUnavailableError",
]
