from __future__ import annotations

import pytest

from rag.chunking.tokens import HuggingFaceTokenCounter, RegexTokenCounter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MAX_SEQ_LENGTH = 256  # from the model's sentence_bert_config.json


@pytest.fixture(scope="session")
def regex_counter() -> RegexTokenCounter:
    return RegexTokenCounter()


@pytest.fixture(scope="session")
def minilm_counter() -> HuggingFaceTokenCounter:
    try:
        return HuggingFaceTokenCounter.from_pretrained(EMBEDDING_MODEL, EMBEDDING_MAX_SEQ_LENGTH)
    except Exception as exc:  # offline machine without a cached tokenizer
        pytest.skip(f"tokenizer unavailable: {exc}")
