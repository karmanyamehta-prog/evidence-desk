"""Token-budgeted chunks whose offsets refer to normalized page text."""

import hashlib
import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from grounded_qa.models import Chunk

PIPELINE_VERSION = "pdfplumber-0.11.7-bounded-tables-v4"
# v4: action-ID prefix injected into embedding text; neighbor IDs in metadata.
# Changing this version invalidates all existing indexes (new collection namespace).


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

    def count(text: str) -> int:
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
            # Prepend the action/section ID to the text used for the chunk ID and
            # embedding so that exact-ID queries ("What does GOVERN 1.6 say?") can
            # match even when the chunk body has little semantic overlap with the
            # query. The stored chunk text remains the verbatim source passage.
            embed_prefix = f"{section_name}: " if section_name else ""
            identity = (
                f"{document_id}:{page['page']}:{start}:{size}:{overlap}"
                f":{PIPELINE_VERSION}:{embed_prefix}{part}"
            )
            cid = hashlib.sha256(identity.encode()).hexdigest()
            if count(part) > size:
                raise ValueError("Splitter exceeded token budget")
            chunks.append(Chunk(cid, part, meta, embed_prefix=embed_prefix))

    # Second pass: stitch neighbor IDs now that all chunk IDs are known.
    for i, chunk in enumerate(chunks):
        prev_id = chunks[i - 1].id if i > 0 else ""
        next_id = chunks[i + 1].id if i < len(chunks) - 1 else ""
        # Rebuild with neighbor metadata — Chunk is frozen so we replace it.
        updated_meta = {
            **chunk.metadata,
            "prev_chunk_id": prev_id,
            "next_chunk_id": next_id,
        }
        chunks[i] = Chunk(
            chunk.id, chunk.text, updated_meta, embed_prefix=chunk.embed_prefix
        )

    return chunks
