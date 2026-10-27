from __future__ import annotations

import re
from itertools import pairwise

import pytest

from rag.chunking.chunker import Chunk, chunk_document
from rag.chunking.tokens import HuggingFaceTokenCounter, RegexTokenCounter, TokenCounter
from rag.config import ChunkingConfig
from rag.errors import ConfigurationError
from rag.parsing.models import Block, BlockKind, RawBlock
from rag.parsing.normalize import build_canonical_text

SENTENCES = [
    f"Control AC-{i} requires the organization to review privileged account usage every {i} days."
    for i in range(1, 40)
]


def corpus() -> tuple[str, list[Block]]:
    raw = [
        RawBlock(BlockKind.HEADING, "Access Control Standard", level=1),
        RawBlock(BlockKind.PARAGRAPH, "Short scope statement.", page=1),
        RawBlock(BlockKind.HEADING, "Account Management", level=2),
        RawBlock(BlockKind.PARAGRAPH, " ".join(SENTENCES[:25]), page=1),  # oversized block
        RawBlock(BlockKind.PARAGRAPH, " ".join(SENTENCES[25:30]), page=2),
        RawBlock(BlockKind.HEADING, "Exceptions", level=2),
        RawBlock(BlockKind.LIST, "\n".join(f"- exception {i}" for i in range(5)), page=2),
        RawBlock(BlockKind.PARAGRAPH, "x" * 3000, page=3),  # one giant "word"
        RawBlock(BlockKind.HEADING, "Appendix", level=1),
        RawBlock(BlockKind.PARAGRAPH, " ".join(SENTENCES[30:]), page=4),
    ]
    canonical, blocks = build_canonical_text(raw)
    return canonical, blocks


def assert_invariants(
    canonical: str,
    blocks: list[Block],
    chunks: list[Chunk],
    config: ChunkingConfig,
    counter: TokenCounter,
) -> None:
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.text == canonical[chunk.char_start : chunk.char_end]
        assert chunk.text.strip() == chunk.text and chunk.text
        assert chunk.token_count == counter.count(chunk.embedding_text)
        assert chunk.token_count <= config.max_tokens
        assert chunk.embedding_text.endswith(chunk.text)
    # Coverage: every non-whitespace character of every body block is in some chunk.
    covered = bytearray(len(canonical))
    for chunk in chunks:
        covered[chunk.char_start : chunk.char_end] = b"\x01" * (chunk.char_end - chunk.char_start)
    for block in blocks:
        if block.kind is BlockKind.HEADING and config.heading_aware:
            continue
        for offset in range(block.start, block.end):
            if not canonical[offset].isspace():
                assert covered[offset], (
                    f"uncovered text at {offset}: {canonical[offset : offset + 30]!r}"
                )


CONFIGS = [
    ChunkingConfig(),
    ChunkingConfig(max_tokens=64, overlap_tokens=16),
    ChunkingConfig(max_tokens=64, overlap_tokens=0, heading_aware=False),
    ChunkingConfig(max_tokens=96, overlap_tokens=24, page_boundaries=True),
    ChunkingConfig(strategy="fixed_tokens", max_tokens=80, overlap_tokens=20),
]


@pytest.mark.parametrize(
    "config", CONFIGS, ids=lambda c: f"{c.strategy}-{c.max_tokens}-{c.overlap_tokens}"
)
def test_invariants_with_regex_counter(
    config: ChunkingConfig, regex_counter: RegexTokenCounter
) -> None:
    canonical, blocks = corpus()
    chunks = chunk_document(canonical, blocks, config, regex_counter)
    assert len(chunks) > 3
    assert_invariants(canonical, blocks, chunks, config, regex_counter)


@pytest.mark.model
@pytest.mark.parametrize(
    "config", CONFIGS, ids=lambda c: f"{c.strategy}-{c.max_tokens}-{c.overlap_tokens}"
)
def test_invariants_with_minilm_tokenizer(
    config: ChunkingConfig, minilm_counter: HuggingFaceTokenCounter
) -> None:
    canonical, blocks = corpus()
    chunks = chunk_document(canonical, blocks, config, minilm_counter)
    assert_invariants(canonical, blocks, chunks, config, minilm_counter)


@pytest.mark.model
def test_minilm_counter_disables_shipped_padding_and_truncation(
    minilm_counter: HuggingFaceTokenCounter,
) -> None:
    assert minilm_counter.count("hello world") == 2  # would be 128 with the shipped config
    assert minilm_counter.count("word " * 600) == 600  # would be truncated to 128
    assert minilm_counter.max_input_tokens == 254  # 256 minus [CLS] and [SEP]


@pytest.mark.model
def test_config_exceeding_model_window_is_rejected(minilm_counter: HuggingFaceTokenCounter) -> None:
    canonical, blocks = corpus()
    with pytest.raises(ConfigurationError, match="silently truncated"):
        chunk_document(
            canonical, blocks, ChunkingConfig(max_tokens=512, overlap_tokens=0), minilm_counter
        )


def test_heading_path_and_prefix(regex_counter: RegexTokenCounter) -> None:
    canonical, blocks = corpus()
    chunks = chunk_document(
        canonical, blocks, ChunkingConfig(max_tokens=64, overlap_tokens=0), regex_counter
    )
    paths = {c.heading_path for c in chunks}
    assert ("Access Control Standard",) in paths
    assert ("Access Control Standard", "Account Management") in paths
    assert ("Appendix",) in paths  # level-1 heading resets the path
    account = next(c for c in chunks if c.heading_path[-1:] == ("Account Management",))
    assert account.embedding_text.startswith("Access Control Standard > Account Management\n\n")
    assert "Account Management" not in account.text  # headings live in the prefix only


def test_chunks_never_cross_sections(regex_counter: RegexTokenCounter) -> None:
    canonical, blocks = corpus()
    chunks = chunk_document(canonical, blocks, ChunkingConfig(), regex_counter)
    headings = [b for b in blocks if b.kind is BlockKind.HEADING]
    for chunk in chunks:
        assert not any(chunk.char_start < h.start < chunk.char_end for h in headings)


def test_page_boundaries_option_yields_single_page_chunks(regex_counter: RegexTokenCounter) -> None:
    canonical, blocks = corpus()
    chunks = chunk_document(canonical, blocks, ChunkingConfig(page_boundaries=True), regex_counter)
    assert all(c.page_start == c.page_end for c in chunks)
    spanning = chunk_document(canonical, blocks, ChunkingConfig(max_tokens=200), regex_counter)
    assert any(c.page_start != c.page_end for c in spanning)


def test_overlap_repeats_whole_sentences(regex_counter: RegexTokenCounter) -> None:
    canonical, blocks = corpus()
    chunks = chunk_document(
        canonical, blocks, ChunkingConfig(max_tokens=64, overlap_tokens=20), regex_counter
    )
    account = [c for c in chunks if c.heading_path[-1:] == ("Account Management",)]
    overlapping = [(a, b) for a, b in pairwise(account) if b.char_start < a.char_end]
    assert overlapping
    for _, nxt in overlapping:
        assert re.match(r"Control AC-\d+ requires", nxt.text)  # starts at a sentence boundary


def test_oversized_word_is_split_into_token_windows(regex_counter: RegexTokenCounter) -> None:
    raw = [RawBlock(BlockKind.PARAGRAPH, "intro " * 100 + "end.")]
    canonical, blocks = build_canonical_text(raw)
    config = ChunkingConfig(max_tokens=32, overlap_tokens=8, max_heading_prefix_tokens=8)
    chunks = chunk_document(canonical, blocks, config, regex_counter)
    assert all(c.token_count <= 32 for c in chunks)
    assert chunks[-1].text.endswith("end.")


def test_long_heading_prefix_is_bounded(regex_counter: RegexTokenCounter) -> None:
    long_heading = " ".join(f"word{i}" for i in range(100))
    raw = [
        RawBlock(BlockKind.HEADING, "Root", level=1),
        RawBlock(BlockKind.HEADING, long_heading, level=2),
        RawBlock(BlockKind.PARAGRAPH, "Body sentence."),
    ]
    canonical, blocks = build_canonical_text(raw)
    config = ChunkingConfig(max_tokens=64, overlap_tokens=0, max_heading_prefix_tokens=10)
    [chunk] = chunk_document(canonical, blocks, config, regex_counter)
    prefix = chunk.embedding_text.removesuffix(chunk.text)
    assert regex_counter.count(prefix) <= 10
    assert prefix.startswith("word0")  # outer ancestor dropped first, then truncated
    assert chunk.token_count <= 64


def test_chunking_is_deterministic(regex_counter: RegexTokenCounter) -> None:
    canonical, blocks = corpus()
    first = chunk_document(canonical, blocks, ChunkingConfig(), regex_counter)
    second = chunk_document(canonical, blocks, ChunkingConfig(), regex_counter)
    assert [c.content_hash for c in first] == [c.content_hash for c in second]


def test_config_validation_and_fingerprint() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkingConfig(max_tokens=100, overlap_tokens=60)
    assert ChunkingConfig().fingerprint() == ChunkingConfig().fingerprint()
    assert ChunkingConfig().fingerprint() != ChunkingConfig(max_tokens=128).fingerprint()
