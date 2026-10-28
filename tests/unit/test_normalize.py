from __future__ import annotations

from rag.parsing.models import BlockKind, RawBlock
from rag.parsing.normalize import build_canonical_text, clean_text, join_wrapped_lines


def test_clean_text_folds_pdf_artifacts() -> None:
    raw = "e\ufb03cient\u00a0work\u200b  flow\r\nnext\tline\x07"
    assert clean_text(raw) == "efficient work flow\nnext line"


def test_private_use_bullet_glyphs_become_bullets_and_stray_glyphs_are_dropped() -> None:
    # Observed in NIST SP 800-61r2: Symbol-font bullets extracted as U+F03C.
    raw = "services a team might offer:\n\uf03c Intrusion Detection. text\n\uf0b7 Advisories\ue000."
    cleaned = clean_text(raw)
    assert (
        cleaned
        == "services a team might offer:\n\u2022 Intrusion Detection. text\n\u2022 Advisories."
    )
    assert join_wrapped_lines(cleaned).count("\n") == 2  # each bullet keeps its own line


def test_join_wrapped_lines_keeps_list_items_and_hyphenated_identifiers() -> None:
    text = (
        "Controls such as AC-\n2 and self-\nservice portals\napply.\n"
        "- first item\n2) second item\nwraps"
    )
    assert join_wrapped_lines(text) == (
        "Controls such as AC-2 and self-service portals apply.\n- first item\n2) second item wraps"
    )


def test_canonical_offsets_are_exact_and_empty_blocks_dropped() -> None:
    raw = [
        RawBlock(BlockKind.HEADING, "  Leave   Policy ", level=1),
        RawBlock(BlockKind.PARAGRAPH, "\u200b \n"),
        RawBlock(BlockKind.PARAGRAPH, "Employees accrue\nleave monthly.", page=3),
        RawBlock(BlockKind.TABLE, "Type | Days\nAnnual | 25"),
        RawBlock(BlockKind.CODE, "    indented()\n"),
    ]
    canonical, blocks = build_canonical_text(raw)
    assert [b.kind for b in blocks] == [
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.TABLE,
        BlockKind.CODE,
    ]
    for block in blocks:
        assert canonical[block.start : block.end] == block.text
    assert blocks[0].text == "Leave Policy"
    assert blocks[1].text == "Employees accrue leave monthly."
    assert blocks[1].page == 3
    assert blocks[2].text == "Type | Days\nAnnual | 25"
    assert blocks[3].text == "    indented()"  # code indentation preserved
