"""Ollama provider (local inference over HTTP).

* Deterministic decoding by default (temperature 0, fixed seed). Identical prompts usually
  reproduce identical output on the same build and hardware; that is not guaranteed
  across Ollama versions or CPUs.
* The model is identified by name **and digest**. Tags such as ``llama3.2:3b`` are mutable,
  so when a digest is configured the provider refuses a model whose digest differs, and
  every answer trace records the digest actually used.
* ``num_ctx`` is set explicitly. Ollama silently truncates prompts longer than its
  context, so the caller budgets evidence beforehand, and the reported prompt-token count
  is checked afterwards (``ContextOverflowError``).
* Output is constrained with Ollama's JSON-schema ``format``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx

from rag.errors import ConfigurationError
from rag.generation.base import (
    ContextOverflowError,
    Generation,
    GenerationError,
    Message,
    ProviderUnavailableError,
)


class OllamaProvider:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        expected_digest: str | None = None,
        context_window: int = 4096,
        temperature: float = 0.0,
        seed: int = 7,
        timeout_seconds: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        if context_window < 512:
            raise ConfigurationError("context_window must be at least 512 tokens")
        self._model = model
        self._expected = expected_digest.removeprefix("sha256:") if expected_digest else None
        self._context = context_window
        self._options = {"temperature": temperature, "seed": seed, "num_ctx": context_window}
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_seconds)
        self._digest: str | None = None

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context

    @property
    def model_digest(self) -> str | None:
        if self._digest is None:
            self._digest = self._resolve_digest()
        return self._digest

    def _resolve_digest(self) -> str:
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderUnavailableError("Ollama is not reachable") from exc
        digest = next(
            (str(m.get("digest", "")) for m in models
             if m.get("name") == self._model or m.get("model") == self._model),
            None,
        )  # fmt: skip
        if not digest:
            raise ProviderUnavailableError(
                f"model {self._model!r} is not pulled into Ollama (run `ollama pull`)"
            )
        if self._expected and not digest.startswith(self._expected):
            raise ConfigurationError(
                f"model {self._model!r} has digest {digest[:12]}, expected {self._expected[:12]}; "
                "the tag now points at different weights"
            )
        return digest

    def count_tokens(self, messages: Sequence[Message]) -> int | None:
        return None  # no tokenize endpoint is relied on; see ContextOverflowError check

    def generate(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> Generation:
        if max_tokens <= 0 or max_tokens >= self._context:
            raise ConfigurationError("max_tokens must be positive and below the context window")
        _ = self.model_digest  # verifies pinning before spending time on inference
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {**self._options, "num_predict": max_tokens},
        }
        if response_schema is not None:
            payload["format"] = response_schema
        started = time.perf_counter()
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError("Ollama request timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise GenerationError(f"Ollama returned HTTP {exc.response.status_code}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderUnavailableError("Ollama request failed") from exc
        latency = (time.perf_counter() - started) * 1000
        prompt_tokens = body.get("prompt_eval_count")
        completion_tokens = body.get("eval_count")
        if isinstance(prompt_tokens, int) and prompt_tokens + max_tokens > self._context:
            raise ContextOverflowError(
                f"prompt used {prompt_tokens} tokens; with {max_tokens} reserved for the answer "
                f"it exceeds the {self._context}-token context and may have been truncated"
            )
        return Generation(
            text=str(body.get("message", {}).get("content", "")),
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            finish_reason=body.get("done_reason"),
            latency_ms=latency,
        )
