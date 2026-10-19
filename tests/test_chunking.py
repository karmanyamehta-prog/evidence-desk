import re

import pytest

from grounded_qa.ingestion.chunking import chunk_pages
from grounded_qa.ingestion.pdf import PDFError, read_pdf


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return re.findall(r"\w+|[^\w\s]", text)


def test_budget_offsets_and_stable_ids():
    pages = [
        {
            "page": 1,
            "text": " ".join(f"token{i}" for i in range(150)),
            "section": "Overview",
        },
        {"page": 2, "text": "Second page has independent provenance."},
    ]
    chunks = chunk_pages(pages, "abc", "Report", Tokenizer(), 32, 8)
    assert chunks == chunk_pages(pages, "abc", "Report", Tokenizer(), 32, 8)
    assert len({c.id for c in chunks}) == len(chunks)
    for c in chunks:
        page = pages[c.metadata["page"] - 1]
        assert c.text == page["text"][c.metadata["start"] : c.metadata["end"]]
        assert len(Tokenizer().encode(c.text)) <= 32
    assert chunks[1].metadata["start"] < chunks[0].metadata["end"]
    assert chunks[-1].metadata["page"] == 2


@pytest.mark.parametrize("overlap", [-1, 32, 40])
def test_reject_invalid_overlap(overlap):
    with pytest.raises(ValueError):
        chunk_pages([], "a", "t", Tokenizer(), 32, overlap)


def test_malformed_pdf(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"this is not a pdf")
    with pytest.raises(PDFError):
        read_pdf(p)


def test_repeated_text_offsets():
    text = "Repeated sentence. " * 30
    chunks = chunk_pages([{"page": 1, "text": text}], "a", "t", Tokenizer(), 20, 4)
    starts = [c.metadata["start"] for c in chunks]
    assert starts == sorted(set(starts))


@pytest.mark.parametrize("bottom", [170.8, 662.7])
def test_table_crop_preserves_full_width_prose(tmp_path, monkeypatch, bottom):
    """Mixed table/prose pages must not lose the left half of following prose."""
    from types import SimpleNamespace

    import grounded_qa.ingestion.pdf as module

    class Page:
        width, height = 612, 792
        lines = [{"x0": 201.2, "x1": 522, "top": bottom, "bottom": bottom}]

        def extract_text(self, **kwargs):
            return "Subcategories\nOriginal full page text is retained for audit."

        def extract_words(self, **kwargs):
            return [{"text": "Subcategories", "x0": 207, "top": 92, "bottom": 104}]

        def crop(self, bounds):
            if bounds[1] == bottom + 1:
                assert bounds[0] == 0 and bounds[2] == self.width
                text = "The MEASURE function employs quantitative tools and methods."
            elif bounds[0] == 206:
                assert bounds[3] == bottom
                text = "MAP 5.2: Practices and personnel support engagement."
            else:
                text = "Table 2: Categories and subcategories"
            return SimpleNamespace(extract_text=lambda **kwargs: text)

    class PDF:
        pages = [Page()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(module.pdfplumber, "open", lambda _: PDF())
    monkeypatch.setattr(
        module,
        "document_id",
        lambda _: "cde9f5bab41c68718f3413ab2564c2285ac9195fa9160a61cc5058dd9c35d6e7",
    )
    pages = module.read_pdf(path)
    assert "The MEASURE function employs" in pages[0]["text"]
    assert "MAP 5.2:" in pages[0]["text"]
