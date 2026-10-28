"""LLM provider contract and the structured answer format.

Providers return raw text; ``parse_answer`` turns it into a validated ``ModelAnswer`` or
raises ``InvalidModelOutputError``. Malformed output is never repaired by guessing, and
the caller withholds the answer instead.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class GenerationError(RuntimeError):
    code = "generation_error"
    retryable = False


class ProviderUnavailableError(GenerationError):
    code = "llm_unavailable"
    retryable = True


class ContextOverflowError(GenerationError):
    """The prompt would not fit the model's context window with room for the answer."""

    code = "context_overflow"


class InvalidModelOutputError(GenerationError):
    code = "invalid_model_output"


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    prompt_tokens: int | None  # as reported by the provider
    completion_tokens: int | None
    finish_reason: str | None
    latency_ms: float


class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def model_digest(self) -> str | None:
        """Content digest of the loaded model, when the provider exposes one."""
        ...

    @property
    def context_window(self) -> int: ...

    def count_tokens(self, messages: Sequence[Message]) -> int | None:
        """Exact prompt tokens in the model's own tokenizer, or None if unavailable."""
        ...

    def generate(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> Generation: ...


class AnswerSentence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = Field(max_length=2000)
    citations: list[str] = Field(default_factory=list, max_length=20)


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    abstain: bool
    abstain_reason: str | None = Field(default=None, max_length=500)
    sentences: list[AnswerSentence] = Field(default_factory=list, max_length=40)


ANSWER_SCHEMA: dict[str, Any] = ModelAnswer.model_json_schema()


def parse_answer(text: str) -> ModelAnswer:
    try:
        return ModelAnswer.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise InvalidModelOutputError("model output is not a valid answer object") from exc
