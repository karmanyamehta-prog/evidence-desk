"""Page-bound PDF extraction with auditable source offsets.

No OCR: image-only documents are rejected. Raw page text is retained alongside
normalized text. A heading is a heuristic, not an authoritative classification.
"""
import hashlib
import re
from pathlib import Path
import pdfplumber

class PDFError(ValueError):
    pass

def normalize(text: str) -> str:
    text = text.replace("\x00", "").replace("\u00ad", "")
    return re.sub(r"\s+", " ", text).strip()

def read_pdf(path: Path, max_pages: int = 200) -> list[dict]:
    if path.read_bytes()[:5] != b"%PDF-":
        raise PDFError("Upload is not a PDF")
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            if len(pdf.pages) > max_pages:
                raise PDFError(f"PDF exceeds {max_pages} pages")
            for n, page in enumerate(pdf.pages, 1):
                raw = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                # Preserve paragraph/line separators for the recursive splitter.
                table_header = next((w for w in page.extract_words(x_tolerance=1)
                                     if w["text"] == "Subcategories"), None)
                extraction = raw
                region = "full-page"
                if table_header:
                    # A two-column RMF table is not prose: retain the action column
                    # separately so category descriptions do not interrupt action text.
                    bounds = (table_header["x0"] - 1, table_header["bottom"], page.width - 40, page.height - 65)
                    extraction = page.crop(bounds).extract_text(x_tolerance=1, y_tolerance=3) or ""
                    region = "subcategory-column"
                extraction = re.sub(r"-\n(?=[a-z])", "", extraction)
                lines = [x.strip() for x in extraction.splitlines()]
                heading = next((x for x in lines if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z]", x)
                                and len(x) < 100), "")
                printed = re.search(r"(?:Page\s+)?(\d+|[ivx]+)\s*$", raw)
                pages.append({"page": n, "raw": raw, "text": "\n".join(normalize(line) for line in extraction.splitlines()), "region": region,
                              "section": heading, "section_method": "page-heading-heuristic",
                              "printed_page_candidate": printed.group(1) if printed else ""})
    except PDFError:
        raise
    except Exception as exc:
        raise PDFError("Cannot parse PDF; it may be malformed or encrypted") from exc
    if not any(len(p["text"]) > 40 for p in pages):
        raise PDFError("PDF has no usable text; OCR is not included")
    return pages

def document_id(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
