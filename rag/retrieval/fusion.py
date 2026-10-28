"""Reciprocal Rank Fusion over ranked candidate lists.

``score(d) = sum over lists L containing d of weight_L / (k + rank_L(d))``

Only ranks are used. Cosine similarities and BM25 scores live on different, unbounded
scales and are never compared or added. Deduplication is deterministic: within a list a
repeated id keeps its best (first) rank; ties in fused score are broken by best rank,
then by the number of lists containing the id, then by the id itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from rag.retrieval.dense import Candidate


class FusionConfig(BaseModel):
    """RRF settings. k=60 is the value from the original RRF paper (Cormack et al., 2009)
    and the retrieval design; weights are equal. Neither was tuned on evaluation data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(default=60, ge=1, le=1000)
    dense_weight: float = Field(default=1.0, ge=0)
    sparse_weight: float = Field(default=1.0, ge=0)

    def weight(self, retriever: str) -> float:
        return {"dense": self.dense_weight, "sparse": self.sparse_weight}[retriever]


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    point_id: UUID
    score: float
    rank: int
    ranks: dict[str, int] = field(default_factory=dict)  # retriever -> 1-based rank
    scores: dict[str, float] = field(default_factory=dict)  # retriever -> raw score


def reciprocal_rank_fusion(
    lists: Mapping[str, Sequence[Candidate]], config: FusionConfig
) -> list[FusedCandidate]:
    fused: dict[UUID, float] = {}
    ranks: dict[UUID, dict[str, int]] = {}
    scores: dict[UUID, dict[str, float]] = {}
    for retriever, candidates in lists.items():
        weight = config.weight(retriever)
        rank = 0
        for candidate in candidates:
            if retriever in ranks.get(candidate.point_id, {}):
                continue  # duplicate within one list: keep the first (best) occurrence
            rank += 1
            ranks.setdefault(candidate.point_id, {})[retriever] = rank
            scores.setdefault(candidate.point_id, {})[retriever] = candidate.score
            fused[candidate.point_id] = fused.get(candidate.point_id, 0.0) + weight / (
                config.k + rank
            )
    ordered = sorted(
        fused,
        key=lambda pid: (-fused[pid], min(ranks[pid].values()), -len(ranks[pid]), str(pid)),
    )
    return [
        FusedCandidate(pid, fused[pid], position, ranks[pid], scores[pid])
        for position, pid in enumerate(ordered, start=1)
    ]
