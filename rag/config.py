"""Versioned configuration objects for ingestion.

Configurations are immutable and fingerprinted. The fingerprint becomes part of the index
version identity, so changing any parameter that alters chunk boundaries forces an explicit
reindex instead of silently mixing incompatible chunks in one index.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkingConfig(BaseModel):
    """Chunking parameters.

    Defaults are a starting point derived from a hard model constraint, not a tuned optimum:
    ``all-MiniLM-L6-v2`` truncates input beyond 256 word pieces (254 excluding [CLS]/[SEP]),
    so chunks must stay below that or their tails are silently never embedded. 200 tokens
    leaves headroom for the heading prefix. Final values are selected by the Phase 3 chunking
    benchmark (see docs/rag/evaluation-plan.md).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["structure_aware", "fixed_tokens"] = "structure_aware"
    max_tokens: int = Field(default=200, ge=32, le=8192)
    overlap_tokens: int = Field(default=32, ge=0)
    heading_aware: bool = True
    page_boundaries: bool = False
    include_heading_prefix: bool = True
    # Bounded so the body always keeps at least half of the budget (validated below).
    max_heading_prefix_tokens: int = Field(default=32, ge=0)

    @model_validator(mode="after")
    def _check_budgets(self) -> Self:
        if self.overlap_tokens * 2 > self.max_tokens:
            raise ValueError("overlap_tokens must be at most half of max_tokens")
        if self.max_heading_prefix_tokens * 2 > self.max_tokens:
            raise ValueError("max_heading_prefix_tokens must be at most half of max_tokens")
        return self

    def fingerprint(self) -> str:
        """Stable short hash of every parameter that affects chunk boundaries."""
        canonical = self.model_dump_json()  # field order is declaration order: stable
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class IngestionLimits(BaseModel):
    """Resource limits applied to untrusted uploads before and during parsing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_file_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    max_pdf_pages: int = Field(default=2000, gt=0)
    max_extracted_chars: int = Field(default=20_000_000, gt=0)
    max_zip_entries: int = Field(default=5000, gt=0)
    max_zip_uncompressed_bytes: int = Field(default=256 * 1024 * 1024, gt=0)
    max_zip_compression_ratio: float = Field(default=200.0, gt=1.0)
