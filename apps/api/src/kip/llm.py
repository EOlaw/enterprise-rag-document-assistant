"""Builds the configured LLM provider and grounded-answer settings from Settings."""

from dataclasses import replace
from typing import Any

from kip.answering import AnswerSettings
from kip.settings import Settings
from rag.answering import AuthorityPolicy, SufficiencyThresholds
from rag.errors import ConfigurationError
from rag.generation import LLMProvider
from rag.generation.ollama import OllamaProvider


def make_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider != "ollama":
        raise ConfigurationError(f"unsupported llm_provider {settings.llm_provider!r}")
    return OllamaProvider(
        settings.llm_model,
        base_url=settings.ollama_url,
        expected_digest=settings.llm_model_digest,
        context_window=settings.llm_context_window,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def answer_settings(settings: Settings, **overrides: Any) -> AnswerSettings:
    base = AnswerSettings(
        rerank=settings.answer_rerank,
        top_k=settings.answer_top_k,
        budget_tokens=settings.answer_budget_tokens,
        max_sources=settings.answer_max_sources,
        max_per_document=settings.answer_max_per_document,
        max_answer_tokens=settings.llm_max_answer_tokens,
        policy=AuthorityPolicy.parse(settings.authority_policy),
        thresholds=SufficiencyThresholds(
            min_coverage=settings.sufficiency_min_coverage,
            min_rerank=settings.sufficiency_min_rerank,
        ),
        overfetch=settings.search_overfetch,
        max_candidates=settings.search_max_candidates,
        hybrid_candidates=settings.hybrid_candidates,
        rrf_k=settings.rrf_k,
        rerank_candidates=settings.rerank_candidates,
        store_question=settings.answer_store_question,
    )
    return replace(base, **{k: v for k, v in overrides.items() if v is not None})
