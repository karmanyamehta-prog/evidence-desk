"""Token-budgeted chunks whose offsets refer to normalized page text."""

import hashlib
import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from grounded_qa.models import Chunk

PIPELINE_VERSION = "pdfplumber-0.11.7-bounded-tables-v3"


def chunk_pages(
    pages: list[dict[str, Any]],
    document_id: str,
    title: str,
    tokenizer: Any,
    size: int = 350,
    overlap: int = 50,
    source_url: str = "",
) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size")

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=count,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        keep_separator=True,
    )
    chunks = []
    for page in pages:
        text = page["text"]
        cursor = 0
        # Keep action boundaries even when adjacent actions have similar wording.
        sections = re.split(
            r"(?=^(?:GOVERN|MAP|MEASURE|MANAGE)\s*\d+\.\d+:|^\d+\.\d+\s+[A-Z])",
            text,
            flags=re.M,
        )
        parts = []
        for section in sections:
            action = re.match(r"(?:GOVERN|MAP|MEASURE|MANAGE)\s*\d+\.\d+", section)
            parts.extend(
                (part, action.group(0) if action else page.get("section", ""))
                for part in splitter.split_text(section)
            )
        for part, section_name in parts:
            # Search from the previous start, allowing overlapping chunks.
            start = text.find(part, cursor)
            if start < 0:
                raise ValueError("Chunk cannot be mapped to its source page")
            cursor = start + 1
            meta = {
                "document_id": document_id,
                "title": title,
                "page": page["page"],
                "section": section_name,
                "source_url": source_url,
                "start": start,
                "end": start + len(part),
                "pipeline": PIPELINE_VERSION,
            }
            identity = f"{document_id}:{page['page']}:{start}:{size}:{overlap}:{PIPELINE_VERSION}:{part}"
            cid = hashlib.sha256(identity.encode()).hexdigest()
            if count(part) > size:
                raise ValueError("Splitter exceeded token budget")
            chunks.append(Chunk(cid, part, meta))
    return chunks
