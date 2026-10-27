import multiprocessing
import time
from multiprocessing.connection import Connection

import pytest
from kip.worker import processor as processor_module
from kip.worker.processor import ParseRequest, ProcessingError, SubprocessProcessor

from rag.config import ChunkingConfig, IngestionLimits


def _hang(connection: Connection, request: ParseRequest) -> None:
    time.sleep(10)


def _crash(connection: Connection, request: ParseRequest) -> None:
    connection.close()


def _processor(**kwargs) -> SubprocessProcessor:
    return SubprocessProcessor(
        tokenizer_json=None,
        tokenizer_name="regex-v1",
        max_seq_tokens=256,
        heartbeat_seconds=0.1,
        **kwargs,
    )


def test_subprocess_returns_chunks_and_heartbeats() -> None:
    pulses = []
    result = _processor().process(
        b"Evidence in a separate process.",
        "guide.txt",
        chunking=ChunkingConfig(),
        limits=IngestionLimits(),
        pulse=lambda: pulses.append(True),
    )
    assert result.chunks[0].text == result.canonical_text
    assert pulses


def test_subprocess_propagates_stable_error() -> None:
    with pytest.raises(ProcessingError) as error:
        _processor().process(
            b"   ",
            "empty.txt",
            chunking=ChunkingConfig(),
            limits=IngestionLimits(),
            pulse=lambda: None,
        )
    assert error.value.code == "empty_document" and not error.value.retryable


def test_deadline_kills_child(monkeypatch) -> None:
    monkeypatch.setattr(processor_module, "_parse", _hang)
    before = {process.pid for process in multiprocessing.active_children()}
    start = time.monotonic()
    with pytest.raises(ProcessingError) as error:
        _processor(timeout_seconds=1).process(
            b"a",
            "guide.txt",
            chunking=ChunkingConfig(),
            limits=IngestionLimits(),
            pulse=lambda: None,
        )
    assert error.value.code == "parser_timeout"
    assert time.monotonic() - start < 6
    assert {process.pid for process in multiprocessing.active_children()} == before


def test_crash_is_retryable(monkeypatch) -> None:
    monkeypatch.setattr(processor_module, "_parse", _crash)
    with pytest.raises(ProcessingError) as error:
        _processor().process(
            b"a",
            "guide.txt",
            chunking=ChunkingConfig(),
            limits=IngestionLimits(),
            pulse=lambda: None,
        )
    assert error.value.code == "parser_crashed" and error.value.retryable
