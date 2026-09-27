"""Token-budgeted chunks whose offsets refer to normalized page text."""
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from grounded_qa.models import Chunk

PIPELINE_VERSION = "pdfplumber-0.11.7-page-v1"

def chunk_pages(pages, document_id, title, tokenizer, size=350, overlap=50, source_url=""):
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size")
    count = lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, length_function=count,
        separators=["\n\n", "\n", ". ", "; ", " ", ""], keep_separator=True,
    )
    chunks = []
    for page in pages:
        text = page["text"]
        cursor = 0
        for part in splitter.split_text(text):
            # Search from the previous start, allowing overlapping chunks.
            start = text.find(part, cursor)
            if start < 0:
                raise ValueError("Chunk cannot be mapped to its source page")
            cursor = start + 1
            meta = {"document_id": document_id, "title": title, "page": page["page"],
                    "section": page.get("section", ""), "source_url": source_url,
                    "start": start, "end": start + len(part), "pipeline": PIPELINE_VERSION}
            identity = f"{document_id}:{page['page']}:{start}:{size}:{overlap}:{PIPELINE_VERSION}:{part}"
            cid = hashlib.sha256(identity.encode()).hexdigest()
            if count(part) > size:
                raise ValueError("Splitter exceeded token budget")
            chunks.append(Chunk(cid, part, meta))
    return chunks
