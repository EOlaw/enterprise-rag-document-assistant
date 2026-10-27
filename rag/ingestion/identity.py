"""Content hashes and deterministic identifiers.

Deterministic point IDs are the mechanism that prevents duplicate vectors from retried
ingestion jobs: re-running a job upserts the same IDs, overwriting rather than appending.
"""

from __future__ import annotations

import hashlib
import uuid

# Fixed namespace for this platform's UUIDv5 identifiers. Never change it: doing so would
# orphan every existing vector point.
_POINT_NAMESPACE = uuid.UUID("6f1c2d4e-8a3b-5c7d-9e0f-1a2b3c4d5e6f")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_fingerprint(canonical_text: str) -> str:
    """Hash of normalized text: detects re-uploads whose bytes differ but content does not
    (e.g. a re-exported PDF), allowing embedding reuse."""
    return sha256_hex(canonical_text.encode("utf-8"))


def index_fingerprint(
    *, parser_version: str, chunking_fingerprint: str, embedding_model: str
) -> str:
    """Identity of everything that determines vector contents for a given document version."""
    material = (
        f"parser={parser_version}|chunking={chunking_fingerprint}|embedding={embedding_model}"
    )
    return sha256_hex(material.encode("utf-8"))[:16]


def chunk_point_id(
    document_version_id: uuid.UUID, index_fingerprint: str, chunk_index: int
) -> uuid.UUID:
    """Deterministic Qdrant point ID for one chunk of one document version in one index."""
    return uuid.uuid5(_POINT_NAMESPACE, f"{document_version_id}:{index_fingerprint}:{chunk_index}")
