import re
import pytest
from grounded_qa.ingestion.chunking import chunk_pages
from grounded_qa.ingestion.pdf import read_pdf, PDFError

class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return re.findall(r"\w+|[^\w\s]", text)

def test_budget_offsets_and_stable_ids():
    pages = [{"page": 1, "text": " ".join(f"token{i}" for i in range(150)), "section": "Overview"},
             {"page": 2, "text": "Second page has independent provenance."}]
    chunks = chunk_pages(pages, "abc", "Report", Tokenizer(), 32, 8)
    assert chunks == chunk_pages(pages, "abc", "Report", Tokenizer(), 32, 8)
    assert len({c.id for c in chunks}) == len(chunks)
    for c in chunks:
        page = pages[c.metadata["page"]-1]
        assert c.text == page["text"][c.metadata["start"]:c.metadata["end"]]
        assert len(Tokenizer().encode(c.text)) <= 32
    assert chunks[1].metadata["start"] < chunks[0].metadata["end"]
    assert chunks[-1].metadata["page"] == 2

@pytest.mark.parametrize("overlap", [-1, 32, 40])
def test_reject_invalid_overlap(overlap):
    with pytest.raises(ValueError): chunk_pages([], "a", "t", Tokenizer(), 32, overlap)

def test_malformed_pdf(tmp_path):
    p=tmp_path/"bad.pdf"; p.write_bytes(b"this is not a pdf")
    with pytest.raises(PDFError): read_pdf(p)

def test_repeated_text_offsets():
    text="Repeated sentence. "*30
    chunks=chunk_pages([{"page":1,"text":text}],"a","t",Tokenizer(),20,4)
    starts=[c.metadata["start"] for c in chunks]
    assert starts == sorted(set(starts))
