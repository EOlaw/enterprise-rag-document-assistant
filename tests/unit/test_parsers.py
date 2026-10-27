from __future__ import annotations

import pytest

from rag.config import IngestionLimits
from rag.errors import (
    CorruptedDocumentError,
    DocumentLimitExceededError,
    EncryptedDocumentError,
    UnsupportedEncodingError,
)
from rag.parsing.docx import parse_docx
from rag.parsing.models import BlockKind
from rag.parsing.pdf import parse_pdf, strip_repeated_margins
from rag.parsing.text import parse_markdown, parse_plain_text
from tests.builders import encrypt_pdf, make_docx, make_pdf

LIMITS = IngestionLimits()


# ----------------------------------------------------------------------------------- PDF


def test_pdf_preserves_physical_page_numbers_and_title() -> None:
    pdf = make_pdf(
        [["Intro text."], [], ["Account management is AC-2."]], title="Security Controls"
    )
    doc = parse_pdf(pdf, LIMITS)
    assert doc.title == "Security Controls"
    assert doc.page_count == 3
    assert [(b.page, b.text.strip()) for b in doc.blocks] == [
        (1, "Intro text."),
        (3, "Account management is AC-2."),
    ]
    assert any("no extractable text on pages 2" in w for w in doc.warnings)


def test_pdf_running_headers_and_footers_are_removed() -> None:
    topics = ["account management", "incident handling", "media protection", "audit logging"]
    pages = [
        [
            "ACME Confidential",
            f"Section {n} covers {topic}.",
            "Shared sentence in the body.",
            f"Closing remark on {topic}.",
            f"Page {n} of 4",
        ]
        for n, topic in enumerate(topics, start=1)
    ]
    doc = parse_pdf(make_pdf(pages), LIMITS)
    text = "\n".join(b.text for b in doc.blocks)
    assert "ACME Confidential" not in text
    assert "Page 2 of 4" not in text
    assert text.count("Shared sentence in the body.") == 4  # mid-page repeats are kept
    for topic in topics:
        assert f"covers {topic}." in text
        assert f"Closing remark on {topic}." in text
    assert "removed 8 repeated header/footer lines" in doc.warnings


def test_margin_stripping_ignores_short_documents() -> None:
    lines = [["Header", "body"], ["Header", "body 2"]]
    assert strip_repeated_margins(lines) == (lines, 0)


def test_pdf_with_empty_user_password_is_readable() -> None:
    doc = parse_pdf(encrypt_pdf(make_pdf([["Readable restricted PDF."]]), user_password=""), LIMITS)
    assert doc.blocks[0].text.strip() == "Readable restricted PDF."


def test_password_protected_pdf_is_rejected() -> None:
    with pytest.raises(EncryptedDocumentError):
        parse_pdf(encrypt_pdf(make_pdf([["secret"]]), user_password="hunter2"), LIMITS)


def test_corrupted_pdf_is_rejected() -> None:
    with pytest.raises(CorruptedDocumentError):
        parse_pdf(b"%PDF-1.7\nthis is not a pdf body at all", LIMITS)


def test_pdf_page_limit_is_enforced() -> None:
    with pytest.raises(DocumentLimitExceededError):
        parse_pdf(make_pdf([["a"], ["b"], ["c"]]), IngestionLimits(max_pdf_pages=2))


# ---------------------------------------------------------------------------------- DOCX


def test_docx_reading_order_headings_lists_and_tables() -> None:
    data = make_docx(
        [
            ("heading", ("Remote Work Policy", 1)),
            ("paragraph", "Employees may work remotely two days per week."),
            ("bullet", "Request approval from your manager."),
            ("bullet", "Use the corporate VPN."),
            ("heading", ("Equipment", 2)),
            ("table", [["Item", "Allowance"], ["Monitor", "USD 200"]]),
        ],
        title="HR-POL-007",
    )
    doc = parse_docx(data)
    assert doc.title == "HR-POL-007"
    assert [(b.kind, b.level) for b in doc.blocks] == [
        (BlockKind.HEADING, 1),
        (BlockKind.PARAGRAPH, None),
        (BlockKind.LIST, None),
        (BlockKind.HEADING, 2),
        (BlockKind.TABLE, None),
    ]
    assert doc.blocks[2].text == "Request approval from your manager.\nUse the corporate VPN."
    assert doc.blocks[4].text == "Item | Allowance\nMonitor | USD 200"
    assert all(b.page is None for b in doc.blocks)


def test_corrupted_docx_is_rejected() -> None:
    data = make_docx([("paragraph", "x")])
    with pytest.raises(CorruptedDocumentError):
        parse_docx(data[: len(data) // 2])


# ------------------------------------------------------------------------- text/markdown


def test_plain_text_paragraphs_and_form_feed_pages() -> None:
    doc = parse_plain_text(b"First para\nwraps.\n\nSecond para.\fThird on page two.")
    assert [(b.page, b.text) for b in doc.blocks] == [
        (1, "First para\nwraps."),
        (1, "Second para."),
        (2, "Third on page two."),
    ]
    assert doc.page_count == 2


def test_plain_text_without_form_feeds_has_no_pages() -> None:
    doc = parse_plain_text("caf\u00e9".encode("utf-8-sig"))
    assert doc.blocks[0].text == "caf\u00e9"
    assert doc.blocks[0].page is None


def test_utf16_with_bom_is_decoded() -> None:
    assert parse_plain_text("Policy \u00a7 4".encode("utf-16")).blocks[0].text == "Policy \u00a7 4"


def test_legacy_encoding_is_rejected_not_guessed() -> None:
    with pytest.raises(UnsupportedEncodingError):
        parse_plain_text("caf\u00e9 r\u00e9sum\u00e9".encode("cp1252"))


MARKDOWN = b"""---
title: "Incident Response Runbook"
owner: sre
---
<!-- internal note: not indexed -->
# Runbook

Setext Section
--------------

Restart the **payment-api** service with `kubectl rollout restart` and see [the docs](https://x.y).

- Check error code `ERR_CONN_RESET`
- Escalate after
  15 minutes

| Severity | Response |
|----------|----------|
| SEV-1 | 15 min |

```bash
# not a heading
kubectl get pods
```
"""


def test_markdown_structure() -> None:
    doc = parse_markdown(MARKDOWN)
    assert doc.title == "Incident Response Runbook"
    kinds = [(b.kind, b.text) for b in doc.blocks]
    assert kinds == [
        (BlockKind.HEADING, "Runbook"),
        (BlockKind.HEADING, "Setext Section"),
        (
            BlockKind.PARAGRAPH,
            "Restart the payment-api service with kubectl rollout restart and see the docs.",
        ),
        (BlockKind.LIST, "- Check error code ERR_CONN_RESET\n- Escalate after 15 minutes"),
        (BlockKind.TABLE, "Severity | Response\nSEV-1 | 15 min"),
        (BlockKind.CODE, "# not a heading\nkubectl get pods"),
    ]
    assert [b.level for b in doc.blocks[:2]] == [1, 2]


def test_markdown_title_falls_back_to_first_h1() -> None:
    assert parse_markdown(b"Intro\n\n# Real Title\n\ntext").title == "Real Title"


def test_markdown_keeps_snake_case_identifiers() -> None:
    doc = parse_markdown(b"Set max_retry_count and __init__ hooks.")
    assert doc.blocks[0].text == "Set max_retry_count and __init__ hooks."
