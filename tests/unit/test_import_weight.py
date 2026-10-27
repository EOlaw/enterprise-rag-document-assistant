"""The parser subprocess re-imports the worker's module graph (multiprocessing ``spawn``).

Heavy libraries there are not just slow: numpy/grpc thread pools reserve hundreds of MB of
virtual address space, which exhausted the child's RLIMIT_AS in the Linux container and
failed every parse with ``parser_memory_limit``. Keep them lazy.
"""

from __future__ import annotations

import subprocess
import sys

HEAVY = ("numpy", "torch", "qdrant_client", "grpc", "sentence_transformers", "transformers")


def test_worker_module_graph_does_not_import_heavy_libraries() -> None:
    probe = (
        "import sys, kip.worker.__main__, kip.worker.runner, kip.worker.processor, kip.cli; "
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.strip() == ""
