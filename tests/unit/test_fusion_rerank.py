from __future__ import annotations

from uuid import UUID

import pytest

from rag.errors import ConfigurationError
from rag.reranking import RerankInputTooLongError, rerank_order
from rag.reranking.cross_encoder import CrossEncoderReranker
from rag.retrieval import Candidate, FusionConfig, reciprocal_rank_fusion

A, B, C, D = (UUID(int=i) for i in (1, 2, 3, 4))
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


def ranked(*ids: UUID) -> list[Candidate]:
    return [Candidate(pid, 1.0 / rank, rank) for rank, pid in enumerate(ids, 1)]


def test_rrf_matches_hand_computation() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": ranked(A, B, C), "sparse": ranked(C, A)}, FusionConfig(k=60)
    )
    scores = {f.point_id: f.score for f in fused}
    assert scores[A] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[C] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[B] == pytest.approx(1 / 62)
    assert [f.point_id for f in fused] == [A, C, B]
    assert fused[0].ranks == {"dense": 1, "sparse": 2}
    assert [f.rank for f in fused] == [1, 2, 3]


def test_duplicates_within_a_list_keep_their_best_rank() -> None:
    fused = reciprocal_rank_fusion({"dense": ranked(A, A, B)}, FusionConfig(k=1))
    assert {f.point_id: f.ranks["dense"] for f in fused} == {A: 1, B: 2}
    assert fused[1].score == pytest.approx(1 / 3)  # B is rank 2 after deduplication


def test_ties_are_broken_deterministically() -> None:
    # B and D both appear once at rank 1: equal score, equal best rank, one list each.
    first = reciprocal_rank_fusion({"dense": ranked(D), "sparse": ranked(B)}, FusionConfig())
    second = reciprocal_rank_fusion({"sparse": ranked(B), "dense": ranked(D)}, FusionConfig())
    assert [f.point_id for f in first] == [f.point_id for f in second] == [B, D]


def test_weights_and_validation() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": ranked(A), "sparse": ranked(B)}, FusionConfig(sparse_weight=2.0)
    )
    assert [f.point_id for f in fused] == [B, A]
    with pytest.raises(ValueError):
        FusionConfig(k=0)


def test_rerank_order_is_stable_on_ties() -> None:
    order = rerank_order([0.5, 2.0, 0.5, -1.0])
    assert [r.index for r in order] == [1, 0, 2, 3]


def test_unpinned_reranker_revision_is_refused_before_loading() -> None:
    with pytest.raises(ConfigurationError, match="commit hash"):
        CrossEncoderReranker(RERANKER, "main")


@pytest.fixture(scope="module")
def cross_encoder() -> CrossEncoderReranker:
    try:
        return CrossEncoderReranker(RERANKER, RERANKER_REVISION)
    except OSError as exc:  # offline without a cached model
        pytest.skip(f"reranker unavailable: {exc}")


@pytest.mark.model
def test_real_cross_encoder_orders_relevant_passage_first(
    cross_encoder: CrossEncoderReranker,
) -> None:
    scores = cross_encoder.score(
        "how long are backup snapshots kept?",
        ["The cafeteria opens at nine.", "Backup snapshots are kept for 35 days."],
    )
    assert scores[1] > scores[0]
    assert cross_encoder.max_length == 512


@pytest.mark.model
def test_real_cross_encoder_refuses_instead_of_truncating(
    cross_encoder: CrossEncoderReranker,
) -> None:
    long_passage = "security " * 600
    assert cross_encoder.pair_tokens("q", [long_passage])[0] > 512
    with pytest.raises(RerankInputTooLongError):
        cross_encoder.score("q", [long_passage])
