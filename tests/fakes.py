"""Deterministic test doubles. The real MiniLM model is exercised in model-marked tests."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

from rag.chunking.tokens import RegexTokenCounter
from rag.embeddings import Vector, check_token_limits, validate_vectors
from rag.generation import Generation, GenerationError, Message


class HashingEmbedder:
    """Signed feature-hashing bag of words, unit-normalized.

    Lexically similar texts get similar vectors, which makes search assertions meaningful
    without loading torch. It matches the test index metadata (regex-v1 @ test-v1).
    """

    def __init__(
        self,
        *,
        model_id: str = "regex-v1",
        revision: str = "test-v1",
        dimension: int = 384,
        max_seq_length: int = 256,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._dimension = dimension
        self._max_seq = max_seq_length
        self._counter = RegexTokenCounter()
        self.embedded_texts = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_seq_length(self) -> int:
        return self._max_seq

    @property
    def max_input_tokens(self) -> int:
        return self._max_seq - 2

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return [self._counter.count(text) for text in texts]

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        check_token_limits(self, texts)
        vectors = [self._vector(text) for text in texts]
        validate_vectors(vectors, count=len(texts), dimension=self._dimension)
        self.embedded_texts += len(texts)
        return vectors

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]

    def _vector(self, text: str) -> Vector:
        values = [0.0] * self._dimension
        lowered = text.lower()
        for start, end in self._counter.spans(lowered):
            digest = hashlib.blake2b(lowered[start:end].encode(), digest_size=16).digest()
            # Three signed buckets per token so one hash collision cannot cancel a term.
            for offset in (0, 5, 10):
                bucket = int.from_bytes(digest[offset : offset + 4], "little") % self._dimension
                values[bucket] += 1.0 if digest[offset + 4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0:
            values[0], norm = 1.0, 1.0
        return [v / norm for v in values]


class OverlapReranker:
    """Deterministic stand-in for a cross-encoder: scores distinct shared terms.

    ``max_length`` lets tests exercise the explicit pair-length refusal.
    """

    def __init__(self, *, max_length: int = 512) -> None:
        self._max_length = max_length
        self._counter = RegexTokenCounter()
        self.calls: list[int] = []  # number of passages per call

    @property
    def model_id(self) -> str:
        return "overlap-test"

    @property
    def revision(self) -> str:
        return "test-v1"

    @property
    def max_length(self) -> int:
        return self._max_length

    def pair_tokens(self, query: str, passages: Sequence[str]) -> list[int]:
        q = self._counter.count(query)
        return [q + self._counter.count(p) + 3 for p in passages]

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        from rag.reranking import RerankInputTooLongError  # noqa: PLC0415

        if passages and max(self.pair_tokens(query, passages)) > self._max_length:
            raise RerankInputTooLongError("pair exceeds max_length")
        self.calls.append(len(passages))
        terms = set(query.lower().split())
        return [float(len(terms & set(p.lower().split()))) for p in passages]


SOURCE_BLOCK = re.compile(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', re.S)


def sources_in(messages: Sequence[Message]) -> dict[str, str]:
    """Label -> passage text, parsed from the user message exactly as a model sees it."""
    return dict(SOURCE_BLOCK.findall(messages[-1].content))


class ScriptedProvider:
    """LLM double. ``script`` maps the prompt messages to the raw text the model returns.

    The default script answers faithfully: it copies the first sentence of S1 and cites S1.
    """

    def __init__(
        self,
        script: Callable[[Sequence[Message]], str] | None = None,
        *,
        context_window: int = 4096,
        error: GenerationError | None = None,
    ) -> None:
        self.script = script or faithful
        self.calls: list[list[Message]] = []
        self._context = context_window
        self._error = error

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def model(self) -> str:
        return "scripted-test"

    @property
    def model_digest(self) -> str | None:
        return "sha256:test"

    @property
    def context_window(self) -> int:
        return self._context

    def count_tokens(self, messages: Sequence[Message]) -> int | None:
        return None

    def generate(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> Generation:
        self.calls.append(list(messages))
        if self._error is not None:
            raise self._error
        return Generation(self.script(messages), 100, 20, "stop", 1.0)


def answer_json(
    sentences: list[tuple[str, list[str]]], *, abstain: bool = False, reason: str | None = None
) -> str:
    return json.dumps(
        {
            "abstain": abstain,
            "abstain_reason": reason,
            "sentences": [{"text": t, "citations": c} for t, c in sentences],
        }
    )


def faithful(messages: Sequence[Message]) -> str:
    sources = sources_in(messages)
    if not sources:
        return answer_json([], abstain=True, reason="no sources")
    first = sources["S1"].split(". ")[0].rstrip(".") + "."
    return answer_json([(first, ["S1"])])
