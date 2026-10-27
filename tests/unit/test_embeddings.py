from __future__ import annotations

import math

import pytest

from rag.chunking.tokens import HuggingFaceTokenCounter
from rag.config import ChunkingConfig
from rag.embeddings import (
    EmbeddingInputTooLongError,
    EmbeddingModelMismatchError,
    InvalidEmbeddingError,
    check_embedder_matches,
    check_token_limits,
    validate_vectors,
)
from rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag.errors import ConfigurationError
from rag.ingestion.pipeline import process_document
from tests.builders import make_pdf
from tests.fakes import HashingEmbedder

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # pinned commit (see README)


def unit(n: int, at: int = 0) -> list[float]:
    vector = [0.0] * n
    vector[at] = 1.0
    return vector


def test_validate_vectors_accepts_unit_vectors() -> None:
    validate_vectors([unit(4), [0.6, 0.8, 0.0, 0.0]], count=2, dimension=4)


@pytest.mark.parametrize(
    ("vectors", "match"),
    [
        ([unit(4)], "expected 2 vectors"),
        ([unit(4), unit(3)], "dimension"),
        ([unit(4), [math.nan, 0, 0, 0]], "non-finite"),
        ([unit(4), [2.0, 0, 0, 0]], "not unit-normalized"),
        ([unit(4), [0.0, 0, 0, 0]], "not unit-normalized"),
    ],
)
def test_validate_vectors_rejects_bad_output(vectors: list[list[float]], match: str) -> None:
    with pytest.raises(InvalidEmbeddingError, match=match):
        validate_vectors(vectors, count=2, dimension=4)


def test_token_limit_is_enforced_not_truncated() -> None:
    embedder = HashingEmbedder(max_seq_length=12)  # 10 content tokens
    assert check_token_limits(embedder, ["one two three"]) == [3]
    with pytest.raises(EmbeddingInputTooLongError) as error:
        embedder.embed_documents(["word " * 11])
    assert error.value.code == "embedding_token_limit" and not error.value.retryable


def test_embedder_identity_must_match_index_metadata() -> None:
    embedder = HashingEmbedder()
    check_embedder_matches(
        embedder, model_id="regex-v1", revision="test-v1", dimension=384, max_seq_length=256
    )
    with pytest.raises(EmbeddingModelMismatchError):
        check_embedder_matches(
            embedder, model_id="regex-v1", revision="other", dimension=384, max_seq_length=256
        )


def test_unpinned_revision_is_rejected_before_loading() -> None:
    with pytest.raises(ConfigurationError, match="commit hash"):
        SentenceTransformerEmbedder(
            MINILM, "main", expected_dimension=384, expected_max_seq_length=256
        )


# ------------------------------------------------------------ real pinned model


@pytest.fixture(scope="module")
def minilm():  # type: ignore[no-untyped-def]
    try:
        return SentenceTransformerEmbedder(
            MINILM, MINILM_REVISION, expected_dimension=384, expected_max_seq_length=256
        )
    except OSError as exc:  # offline without a cached model
        pytest.skip(f"model unavailable: {exc}")


@pytest.mark.model
def test_minilm_properties_are_verified(minilm) -> None:  # type: ignore[no-untyped-def]
    assert (minilm.dimension, minilm.max_seq_length, minilm.max_input_tokens) == (384, 256, 254)
    vectors = minilm.embed_documents(["Incident response plan.", "Quarterly budget."])
    validate_vectors(vectors, count=2, dimension=384)
    again = minilm.embed_documents(["Incident response plan."])
    assert max(abs(a - b) for a, b in zip(vectors[0], again[0], strict=True)) < 1e-5


@pytest.mark.model
def test_minilm_semantic_sanity(minilm) -> None:  # type: ignore[no-untyped-def]
    query = minilm.embed_query("How should a ransomware attack be reported?")
    related, unrelated = minilm.embed_documents(
        [
            "Report suspected ransomware infections to the incident response team immediately.",
            "The cafeteria menu changes every Monday.",
        ]
    )

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cosine(query, related) > cosine(query, unrelated) + 0.2


@pytest.mark.model
def test_minilm_rejects_inputs_it_would_truncate(minilm) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(EmbeddingInputTooLongError):
        minilm.embed_documents(["security " * 300])


@pytest.mark.model
def test_chunker_and_embedder_tokenizers_agree_on_real_chunks(minilm) -> None:  # type: ignore[no-untyped-def]
    counter = HuggingFaceTokenCounter.from_pretrained(MINILM, 256)
    text = [
        [f"Control IR-{i}: the organization reviews incident handling capability annually, "
         "including ransomware playbooks, e.g. containment and recovery." for i in range(40)]
    ]  # fmt: skip
    document = process_document(
        make_pdf(text), "controls.pdf", chunking=ChunkingConfig(), token_counter=counter
    )
    texts = [chunk.embedding_text for chunk in document.chunks]
    assert minilm.count_tokens(texts) == [chunk.token_count for chunk in document.chunks]
    assert max(minilm.count_tokens(texts)) <= minilm.max_input_tokens
