"""Publication authority: which documents may serve as evidence, and how prominently.

This is about **separate publications** (for example, NIST SP 800-61r2 versus its
replacement, SP 800-61r3), recorded as document metadata. It is not about versions of one
document: a superseded *version* is never retrievable at all, which the search re-check
already guarantees.

Every decision is explicit and reported. ``exclude`` removes a candidate from evidence and
records why; ``demote`` keeps it but orders it after all non-demoted evidence and labels it
in the prompt; ``include`` treats it like current material. Nothing is dropped silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class PublicationStatus(StrEnum):
    CURRENT = "current"
    SUPERSEDED = "superseded"  # replaced by a later publication, not formally withdrawn
    WITHDRAWN = "withdrawn"  # formally withdrawn by its publisher
    UNKNOWN = "unknown"  # nobody has recorded a status


Treatment = Literal["include", "demote", "exclude"]


class AuthorityPolicy(BaseModel):
    """Per-status treatment of evidence. Defaults keep everything visible but rank
    withdrawn and superseded publications last and label them, so a question *about*
    an old publication can still be answered from it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    current: Treatment = "include"
    unknown: Treatment = "include"
    superseded: Treatment = "demote"
    withdrawn: Treatment = "demote"
    # Undated documents (no published_on) are treated by status alone; this flag demotes
    # undated documents of status current/unknown as well, if an organization wants it.
    demote_undated: bool = False

    def treatment(self, status: PublicationStatus, published_on: date | None) -> Treatment:
        chosen: Treatment = getattr(self, status.value)
        if chosen == "include" and self.demote_undated and published_on is None:
            return "demote"
        return chosen

    @classmethod
    def parse(cls, spec: str) -> AuthorityPolicy:
        """Parse ``"withdrawn=exclude,superseded=demote,demote_undated=true"``."""
        values: dict[str, object] = {}
        for item in filter(None, (part.strip() for part in spec.split(","))):
            key, _, value = item.partition("=")
            values[key.strip()] = (
                value.strip().lower() in {"1", "true", "yes"}
                if (key.strip() == "demote_undated")
                else value.strip()
            )
        return cls.model_validate(values)


@dataclass(frozen=True, slots=True)
class Authority:
    status: PublicationStatus
    published_on: date | None
    superseded_by: str | None  # replacement document title, for labels and citations
    note: str | None

    @property
    def label(self) -> str:
        """Human-readable status used in prompts and CLI output."""
        parts = [self.status.value.upper()]
        if self.published_on:
            parts.append(f"published {self.published_on.isoformat()}")
        else:
            parts.append("undated")
        if self.superseded_by:
            parts.append(f"replaced by {self.superseded_by}")
        return "; ".join(parts)


UNKNOWN_AUTHORITY = Authority(PublicationStatus.UNKNOWN, None, None, None)
