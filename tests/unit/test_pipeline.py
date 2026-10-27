from __future__ import annotations

import uuid

import pytest

from rag.chunking.tokens import RegexTokenCounter
from rag.config import ChunkingConfig, IngestionLimits
from rag.errors import (
    ContentTypeMismatchError,
    DocumentLimitExceededError,
    EmptyDocumentError,
    EncryptedDocumentError,
    IngestionError,
)
from rag.ingestion.identity import chunk_point_id, index_fingerprint
from rag.ingestion.pipeline import process_document
from rag.ingestion.validation import FileType
from tests.builders import encrypt_pdf, make_docx, make_pdf

CONFIG = ChunkingConfig(max_tokens=64, overlap_tokens=8)
COUNTER = RegexTokenCounter()

POLICY_PAGES = [
    [
        "Password Policy",
        "",
        "Passwords must be at least 14 characters long.",
        "Administrators must use multi-factor authentication (MFA).",
    ],
    ["Exceptions require approval under policy SEC-POL-0042.", "Review exceptions every 90 days."],
]


def test_pdf_end_to_end_produces_page_cited_chunks() -> None:
    result = process_document(
        make_pdf(POLICY_PAGES, title="Password Policy"),
        "pw.pdf",
        chunking=CONFIG,
        token_counter=COUNTER,
    )
    assert result.file_type is FileType.PDF
    assert result.title == "Password Policy"
    assert result.page_count == 2
    assert result.chunking_fingerprint == CONFIG.fingerprint()
    joined = " ".join(c.text for c in result.chunks)
    assert "SEC-POL-0042" in joined  # identifiers survive normalization untouched
    sec = next(c for c in result.chunks if "SEC-POL-0042" in c.text)
    assert (sec.page_start, sec.page_end) == (1, 2)  # small doc: one chunk spans both pages
    for chunk in result.chunks:
        assert result.canonical_text[chunk.char_start : chunk.char_end] == chunk.text

    paged = process_document(
        make_pdf(POLICY_PAGES),
        "pw.pdf",
        token_counter=COUNTER,
        chunking=ChunkingConfig(max_tokens=64, overlap_tokens=8, page_boundaries=True),
    )
    sec = next(c for c in paged.chunks if "SEC-POL-0042" in c.text)
    assert (sec.page_start, sec.page_end) == (2, 2)


def test_docx_end_to_end_uses_heading_context() -> None:
    data = make_docx(
        [("heading", ("Leave Policy", 1)), ("paragraph", "Employees accrue 2 days per month.")]
    )
    result = process_document(data, "leave.docx", chunking=CONFIG, token_counter=COUNTER)
    [chunk] = result.chunks
    assert chunk.heading_path == ("Leave Policy",)
    assert chunk.page_start is None
    assert result.title == "leave"  # no core title -> filename stem


def test_identical_content_has_identical_text_hash_even_if_bytes_differ() -> None:
    a = process_document(b"Same policy text.\n", "a.txt", chunking=CONFIG, token_counter=COUNTER)
    b = process_document(
        b"Same   policy\ntext.\r\n\r\n", "b.txt", chunking=CONFIG, token_counter=COUNTER
    )
    assert a.raw_sha256 != b.raw_sha256
    assert a.text_sha256 == b.text_sha256
    assert [c.content_hash for c in a.chunks] == [c.content_hash for c in b.chunks]


@pytest.mark.parametrize(
    ("data", "filename", "error"),
    [
        (b"   \n\n  ", "blank.txt", EmptyDocumentError),
        (make_pdf([[], []]), "scanned.pdf", EmptyDocumentError),
        (b"# Only\n## Headings\n", "headings.md", EmptyDocumentError),
        (encrypt_pdf(make_pdf([["x"]]), "pw"), "locked.pdf", EncryptedDocumentError),
        (b"PK\x03\x04garbage", "fake.pdf", ContentTypeMismatchError),
    ],
)
def test_bad_inputs_raise_typed_errors_with_codes(
    data: bytes, filename: str, error: type[IngestionError]
) -> None:
    with pytest.raises(error) as info:
        process_document(data, filename, chunking=CONFIG, token_counter=COUNTER)
    assert info.value.code == error.code
    assert info.value.retryable is False


def test_extracted_text_limit() -> None:
    with pytest.raises(DocumentLimitExceededError):
        process_document(
            b"word " * 100,
            "big.txt",
            chunking=CONFIG,
            token_counter=COUNTER,
            limits=IngestionLimits(max_extracted_chars=100),
        )


def test_untrusted_filename_is_sanitized() -> None:
    result = process_document(
        b"text", "../../../etc/notes.txt", chunking=CONFIG, token_counter=COUNTER
    )
    assert result.filename == "notes.txt"


def test_point_ids_are_deterministic_and_scoped() -> None:
    version = uuid.UUID(int=1)
    fp = index_fingerprint(
        parser_version="1",
        chunking_fingerprint=CONFIG.fingerprint(),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2@rev",
    )
    assert chunk_point_id(version, fp, 0) == chunk_point_id(version, fp, 0)
    assert chunk_point_id(version, fp, 0) != chunk_point_id(version, fp, 1)
    assert chunk_point_id(version, fp, 0) != chunk_point_id(uuid.UUID(int=2), fp, 0)
    other = index_fingerprint(
        parser_version="2",
        chunking_fingerprint=CONFIG.fingerprint(),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2@rev",
    )
    assert chunk_point_id(version, fp, 0) != chunk_point_id(version, other, 0)
