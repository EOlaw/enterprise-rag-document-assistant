from pathlib import Path
from unittest.mock import patch

import pytest
from kip.storage import LocalObjectStore


@pytest.mark.parametrize(
    "key",
    [
        "",
        "../outside",
        "/absolute",
        "a//b",
        "a/../b",
        "C:/x",
        "a\\b",
        "a/./b",
        "a\x00b",
        "a/",
        "a/" + "x" * 512,
    ],
)
def test_object_store_rejects_unsafe_keys(tmp_path: Path, key: str) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    with pytest.raises(ValueError):
        store.put(key, b"data")
    with pytest.raises(ValueError):
        store.get(key, max_bytes=100)


def test_atomic_replacement_preserves_previous_object_on_failure(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    store.put("org/version/original", b"original")
    with (
        patch.object(Path, "replace", side_effect=OSError("simulated filesystem failure")),
        pytest.raises(OSError),
    ):
        store.put("org/version/original", b"replacement")
    assert store.get("org/version/original", max_bytes=100) == b"original"
    assert not list(tmp_path.rglob(".put-*"))
    store.put("org/version/original", b"replacement")
    assert store.get("org/version/original", max_bytes=100) == b"replacement"


def test_bounded_read(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    store.put("key", b"12345")
    assert store.get("key", max_bytes=5) == b"12345"
    with pytest.raises(ValueError, match="size limit"):
        store.get("key", max_bytes=4)
    with pytest.raises(ValueError, match="positive"):
        store.get("key", max_bytes=0)
    with pytest.raises(FileNotFoundError):
        store.get("absent", max_bytes=10)


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    outside = tmp_path / "outside"
    outside.mkdir()
    store = LocalObjectStore(root)
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit symlink creation")
    with pytest.raises(ValueError):
        store.put("link/secret", b"should never escape")
    assert list(outside.iterdir()) == []
