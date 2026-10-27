"""Killable parsing subprocess with deadline and periodic lease heartbeats.

Linux additionally enforces address-space and CPU limits. This is resource isolation,
not an OS security sandbox: network isolation/seccomp require deployment configuration.
Tokenizers are downloaded/loaded by the parent, never inside the parsing subprocess.
"""

import multiprocessing
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Protocol

from tokenizers import Tokenizer

from rag.chunking.tokens import HuggingFaceTokenCounter, RegexTokenCounter
from rag.config import ChunkingConfig, IngestionLimits
from rag.errors import IngestionError
from rag.ingestion.pipeline import ProcessedDocument, process_document

if sys.platform == "linux":
    import resource


class ProcessingError(IngestionError):
    def __init__(self, code: str, *, retryable: bool) -> None:
        super().__init__("document processing failed")
        self.code = code
        self.retryable = retryable


class Processor(Protocol):
    def process(
        self,
        data: bytes,
        filename: str,
        *,
        chunking: ChunkingConfig,
        limits: IngestionLimits,
        pulse: Callable[[], None],
    ) -> ProcessedDocument: ...


@dataclass(frozen=True)
class ParseRequest:
    data: bytes
    filename: str
    chunking: ChunkingConfig
    limits: IngestionLimits
    tokenizer_json: str | None
    tokenizer_name: str
    max_seq_tokens: int
    timeout_seconds: int
    memory_mb: int


def _parse(connection: Connection, request: ParseRequest) -> None:
    try:
        if sys.platform == "linux":
            ceiling = request.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
            resource.setrlimit(
                resource.RLIMIT_CPU, (request.timeout_seconds, request.timeout_seconds)
            )
        counter = (
            RegexTokenCounter()
            if request.tokenizer_json is None
            else HuggingFaceTokenCounter(
                Tokenizer.from_str(request.tokenizer_json),
                request.tokenizer_name,
                request.max_seq_tokens,
            )
        )
        result = process_document(
            request.data,
            request.filename,
            chunking=request.chunking,
            limits=request.limits,
            token_counter=counter,
        )
        connection.send(result)
    except IngestionError as exc:
        connection.send((exc.code, exc.retryable))
    except MemoryError:
        connection.send(("parser_memory_limit", False))
    except Exception:
        connection.send(("parser_error", False))
    finally:
        connection.close()


class SubprocessProcessor:
    def __init__(
        self,
        *,
        tokenizer_json: str | None,
        tokenizer_name: str,
        max_seq_tokens: int,
        timeout_seconds: int = 120,
        memory_mb: int = 1024,
        heartbeat_seconds: float = 5,
    ) -> None:
        if timeout_seconds <= 0 or heartbeat_seconds <= 0:
            raise ValueError("timeout and heartbeat interval must be positive")
        self.tokenizer_json = tokenizer_json
        self.tokenizer_name = tokenizer_name
        self.max_seq_tokens = max_seq_tokens
        self.timeout_seconds = timeout_seconds
        self.memory_mb = memory_mb
        self.heartbeat_seconds = heartbeat_seconds

    def process(
        self,
        data: bytes,
        filename: str,
        *,
        chunking: ChunkingConfig,
        limits: IngestionLimits,
        pulse: Callable[[], None],
    ) -> ProcessedDocument:
        context = multiprocessing.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        request = ParseRequest(
            data,
            filename,
            chunking,
            limits,
            self.tokenizer_json,
            self.tokenizer_name,
            self.max_seq_tokens,
            self.timeout_seconds,
            self.memory_mb,
        )
        process = context.Process(target=_parse, args=(writer, request), daemon=True)
        deadline = time.monotonic() + self.timeout_seconds
        try:
            process.start()
            writer.close()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProcessingError("parser_timeout", retryable=False)
                if reader.poll(min(self.heartbeat_seconds, remaining)):
                    try:
                        result = reader.recv()
                    except EOFError as exc:
                        raise ProcessingError("parser_crashed", retryable=True) from exc
                    if isinstance(result, ProcessedDocument):
                        pulse()
                        return result
                    if isinstance(result, tuple) and len(result) == 2:
                        raise ProcessingError(str(result[0]), retryable=bool(result[1]))
                    raise ProcessingError("parser_protocol", retryable=False)
                pulse()
        finally:
            reader.close()
            writer.close()
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
                process.close()
