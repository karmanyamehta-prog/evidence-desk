"""Unit tests for grounded_qa.ingestion.pdf — normalize, document_id, PDFError."""

import hashlib

import pytest

from grounded_qa.ingestion.pdf import PDFError, document_id, normalize, read_pdf

# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------


def test_normalize_collapses_whitespace():
    assert normalize("hello   world") == "hello world"


def test_normalize_strips_leading_trailing():
    assert normalize("  hello world  ") == "hello world"


def test_normalize_newlines_become_space():
    assert normalize("line1\nline2") == "line1 line2"


def test_normalize_tabs_become_space():
    assert normalize("col1\tcol2") == "col1 col2"


def test_normalize_removes_null_bytes():
    assert normalize("hel\x00lo") == "hello"


def test_normalize_removes_soft_hyphen():
    assert normalize("co­operate") == "cooperate"


def test_normalize_empty_string():
    assert normalize("") == ""


def test_normalize_only_whitespace():
    assert normalize("   ") == ""


# ---------------------------------------------------------------------------
# document_id()
# ---------------------------------------------------------------------------


def test_document_id_is_sha256(tmp_path):
    f = tmp_path / "sample.pdf"
    f.write_bytes(b"%PDF-1.4 content")
    expected = hashlib.sha256(b"%PDF-1.4 content").hexdigest()
    assert document_id(f) == expected


def test_document_id_changes_with_content(tmp_path):
    f1 = tmp_path / "a.pdf"
    f2 = tmp_path / "b.pdf"
    f1.write_bytes(b"%PDF-1.4 aaa")
    f2.write_bytes(b"%PDF-1.4 bbb")
    assert document_id(f1) != document_id(f2)


def test_document_id_stable(tmp_path):
    f = tmp_path / "stable.pdf"
    f.write_bytes(b"%PDF-1.4 stable content here")
    assert document_id(f) == document_id(f)


# ---------------------------------------------------------------------------
# PDFError
# ---------------------------------------------------------------------------


def test_pdf_error_is_value_error():
    """PDFError must be a ValueError so API layer catches it correctly."""
    err = PDFError("bad pdf")
    assert isinstance(err, ValueError)


# ---------------------------------------------------------------------------
# read_pdf() — rejection cases (no real PDF parsing needed)
# ---------------------------------------------------------------------------


def test_read_pdf_rejects_non_pdf_magic(tmp_path):
    p = tmp_path / "notapdf.pdf"
    p.write_bytes(b"this is not a PDF file at all")
    with pytest.raises(PDFError, match="not a PDF"):
        read_pdf(p)


def test_read_pdf_rejects_zip_magic(tmp_path):
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"PK\x03\x04" + b"\x00" * 20)  # ZIP magic
    with pytest.raises(PDFError, match="not a PDF"):
        read_pdf(p)


def test_read_pdf_rejects_empty_file(tmp_path):
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"")
    with pytest.raises(PDFError):
        read_pdf(p)
