"""Atomic filesystem object storage for trusted application-generated keys.

The root must be private to the application OS user. Path validation rejects symlinks
and traversal; it is not a defense against another process with write access to the root.
"""

import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Protocol
from uuid import UUID


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str, *, max_bytes: int) -> bytes: ...


def object_key(org_id: UUID, version_id: UUID, name: str) -> str:
    return f"{org_id}/{version_id}/{name}"


class LocalObjectStore:
    def __init__(self, root: Path) -> None:
        # UUID directories plus a content hash can exceed Windows' legacy 260-char
        # limit, even for ordinary pytest temporary directories. Use extended paths.
        root = root.absolute()
        if os.name == "nt" and not str(root).startswith("\\\\?\\"):
            value = str(root)
            root = Path(
                "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
            )
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()

    def _path(self, key: str) -> Path:
        parts = key.split("/")
        if (
            not parts
            or len(key) > 512
            or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) for part in parts)
            or PurePosixPath(key).is_absolute()
        ):
            raise ValueError("invalid object key")
        path = self.root.joinpath(*parts)
        for parent in (path, *path.parents):
            if parent == self.root:
                break
            if parent.is_symlink() or parent.is_junction():
                raise ValueError("object key traverses a filesystem link")
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("object key escapes storage root")
        return path

    def put(self, key: str, data: bytes) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".put-", dir=target.parent)
        scratch = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            scratch.replace(target)
        finally:
            scratch.unlink(missing_ok=True)

    def get(self, key: str, *, max_bytes: int) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        with self._path(key).open("rb") as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("stored object exceeds size limit")
        return data
