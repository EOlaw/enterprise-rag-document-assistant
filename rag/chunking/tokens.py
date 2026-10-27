"""Token counting in the *embedding model's* tokenizer.

Chunk budgets must be measured with the tokenizer that will embed the chunk; a whitespace
word count underestimates word-piece counts by roughly 1.3x on technical text, which leads
to silent truncation at embedding time.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from tokenizers import Tokenizer

Span = tuple[int, int]


@runtime_checkable
class TokenCounter(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def max_input_tokens(self) -> int | None:
        """Content tokens the model accepts before truncation (excluding special tokens)."""
        ...

    def count(self, text: str) -> int: ...

    def spans(self, text: str) -> list[Span]:
        """Character span of every token, in order."""
        ...


class HuggingFaceTokenCounter:
    """Counts tokens with a Hugging Face ``tokenizers`` tokenizer.

    Important: the ``tokenizer.json`` shipped with sentence-transformers models enables
    padding *and* truncation to 128 tokens. Left as-is, every text would count as exactly
    128 tokens, so both are disabled here.
    """

    def __init__(self, tokenizer: Tokenizer, name: str, model_max_seq_length: int | None) -> None:
        tokenizer.no_padding()
        tokenizer.no_truncation()
        self._tokenizer = tokenizer
        self._name = name
        special = tokenizer.num_special_tokens_to_add(is_pair=False)
        self._max = None if model_max_seq_length is None else model_max_seq_length - special

    @classmethod
    def from_pretrained(
        cls, model_id: str, model_max_seq_length: int | None
    ) -> HuggingFaceTokenCounter:
        return cls(Tokenizer.from_pretrained(model_id), model_id, model_max_seq_length)

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_input_tokens(self) -> int | None:
        return self._max

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def spans(self, text: str) -> list[Span]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return [(start, end) for start, end in encoding.offsets]


class RegexTokenCounter:
    """Deterministic approximation (words and punctuation) for tests and offline previews.

    Not suitable for production chunking: it undercounts word pieces.
    """

    _TOKEN = re.compile(r"\w+|[^\w\s]")

    @property
    def name(self) -> str:
        return "regex-v1"

    @property
    def max_input_tokens(self) -> int | None:
        return None

    def count(self, text: str) -> int:
        return sum(1 for _ in self._TOKEN.finditer(text))

    def spans(self, text: str) -> list[Span]:
        return [m.span() for m in self._TOKEN.finditer(text)]
