"""Build the cross-document evaluation set with exact gold spans.

Every question is anchored to a quote that is verified to occur verbatim in the
named page of the named document, and the recorded offsets are the actual
character offsets of that quote in the extracted page text. Running this script
re-derives the spans, so a parser change that moves the text is caught rather
than silently invalidating the labels.

Usage:
    python tools/build_crossdoc_queries.py            # write evaluation/queries-crossdoc.json
    python tools/build_crossdoc_queries.py --check    # verify only, do not write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grounded_qa.config import corpus_documents  # noqa: E402
from grounded_qa.ingestion.pdf import read_pdf  # noqa: E402

RMF_SHA = "cde9f5bab41c68718f3413ab2564c2285ac9195fa9160a61cc5058dd9c35d6e7"
GAI_SHA = "6e73620ab6b64e90ef2c04bf0e0d6246185a2f4b1b13cab0df494496cff89b6a"

# Each question: id, split, category, question, expected answer, required facts,
# and the document + page + quote that carries the answer. The quote is checked
# against the extracted page text at build time; the offsets are derived from it,
# never typed by hand.
QUESTIONS: list[dict] = [
    # ---- Generative AI Profile: single-document facts -----------------------
    {
        "id": "g01",
        "split": "test",
        "category": "fact",
        "answerable": True,
        "question": "Is the NIST Generative AI Profile voluntary or mandatory?",
        "expected_answer": "Voluntary.",
        "required_facts": ["voluntary"],
        "document": GAI_SHA,
        "page": 5,
        "quote": "intended for voluntary use and to improve the ability of organizations to incorporate trustworthiness",
    },
    {
        "id": "g02",
        "split": "test",
        "category": "fact",
        "answerable": True,
        "question": "What is a profile in the AI RMF, and what does the Generative AI Profile implement?",
        "expected_answer": "A profile is an implementation of the AI RMF functions, categories and subcategories for a specific setting, application or technology; here, Generative AI.",
        "required_facts": [
            "implementation",
            "functions",
            "categories",
            "subcategories",
        ],
        "document": GAI_SHA,
        "page": 5,
        "quote": "A profile is an implementation of the AI RMF functions, categories, and subcategories for a specific",
    },
    {
        "id": "g03",
        "split": "test",
        "category": "fact",
        "answerable": True,
        "question": "What does the Content Provenance consideration say about distinguishing human-generated from AI-generated content?",
        "expected_answer": "Some GAI outputs, such as deepfake content, challenge our ability to distinguish human-generated content from AI-generated synthetic content.",
        "required_facts": ["deepfake", "distinguish", "synthetic content"],
        "document": GAI_SHA,
        "page": 55,
        "quote": "Some aspects of GAI outputs, such as the production of deepfake content, can challenge our ability to",
    },
    {
        "id": "g04",
        "split": "test",
        "category": "action",
        "answerable": True,
        "question": "In the Generative AI Profile, what data provenance information should be considered in GAI system inventory entries?",
        "expected_answer": "Data provenance information such as source, signatures, versioning and watermarks.",
        "required_facts": ["provenance", "signatures", "versioning", "watermarks"],
        "document": GAI_SHA,
        "page": 20,
        "quote": "consider the\nfollowing items in GAI system inventory entries: Data provenance information",
    },
    {
        "id": "g05",
        "split": "test",
        "category": "fact",
        "answerable": True,
        "question": "How are Action IDs in the Generative AI Profile tables structured, and what do the tags GV, MP, MS and MG mean?",
        "expected_answer": "Each Action ID corresponds to the relevant AI RMF function and subcategory; GV = Govern, MP = Map, MS = Measure, MG = Manage.",
        "required_facts": [
            "GV",
            "Govern",
            "MP",
            "Map",
            "MS",
            "Measure",
            "MG",
            "Manage",
        ],
        "document": GAI_SHA,
        "page": 17,
        "quote": "Action ID: Each Action ID corresponds to the relevant AI RMF function and subcategory",
    },
    # ---- Cross-document: both PDFs must be searched -----------------------
    {
        "id": "x01",
        "split": "test",
        "category": "synthesis",
        "answerable": True,
        "question": "The AI RMF Core defines four functions. Which of them does the Generative AI Profile add suggested actions under, and what are those action IDs?",
        "expected_answer": "The Generative AI Profile adds suggested actions under all four Core functions, using IDs of the form GV-1.1-001, MP-..., MS-... and MG-....",
        "required_facts": ["GV", "MP", "MS", "MG"],
        "document": GAI_SHA,
        "page": 17,
        "quote": "AI RMF functions are tagged as follows: GV = Govern;\nMP = Map; MS = Measure; MG = Manage.",
        "also_relevant": [
            {
                "document": RMF_SHA,
                "page": 8,
                "quote": "These functions – GOVERN,\nMAP, MEASURE, and MANAGE",
            }
        ],
    },
    {
        "id": "x02",
        "split": "test",
        "category": "synthesis",
        "answerable": True,
        "question": "Both the AI RMF and its Generative AI Profile say the framework is voluntary. What does the AI RMF say the Framework is intended to be?",
        "expected_answer": "Voluntary, rights-preserving, non-sector-specific and use-case agnostic.",
        "required_facts": [
            "voluntary",
            "rights-preserving",
            "non-sector-specific",
            "use-case agnostic",
        ],
        "document": RMF_SHA,
        "page": 7,
        "quote": "The Framework is intended to be\nvoluntary, rights-preserving, non-sector-specific, and use-case agnostic",
        "also_relevant": [
            {
                "document": GAI_SHA,
                "page": 5,
                "quote": "intended for voluntary use and to improve the ability of organizations to incorporate trustworthiness",
            }
        ],
    },
    {
        "id": "x03",
        "split": "test",
        "category": "synthesis",
        "answerable": True,
        "question": "The AI RMF says its four Core functions are used to organize the Framework. Where does the Generative AI Profile place its primary considerations?",
        "expected_answer": "In Appendix A, as primary GAI considerations: Governance, Pre-Deployment Testing, Content Provenance and Incident Disclosure.",
        "required_facts": [
            "Governance",
            "Pre-Deployment Testing",
            "Content Provenance",
            "Incident Disclosure",
        ],
        "document": GAI_SHA,
        "page": 51,
        "quote": "These considerations (Governance, Pre-Deployment Testing, Content Provenance,\nand Incident Disclosure) are relevant for voluntary use by any organization designing, developing, and",
        "also_relevant": [
            {
                "document": RMF_SHA,
                "page": 8,
                "quote": "These functions – GOVERN,\nMAP, MEASURE, and MANAGE",
            }
        ],
    },
    # ---- Unanswerable: the corpus does not contain these ------------------
    {
        "id": "u01",
        "split": "test",
        "category": "unanswerable",
        "answerable": False,
        "question": "What was the exact dollar cost of running the NIST AI RMF evaluation suite in fiscal year 2024?",
        "expected_answer": "Not stated in the corpus.",
        "required_facts": [],
    },
    {
        "id": "u02",
        "split": "test",
        "category": "unanswerable",
        "answerable": False,
        "question": "Which foundation model does the Generative AI Profile recommend organizations deploy as a default?",
        "expected_answer": "The Profile names no default model; it names no models at all.",
        "required_facts": [],
    },
    {
        "id": "u03",
        "split": "test",
        "category": "unanswerable",
        "answerable": False,
        "question": "What is the maximum number of AI systems the Framework allows an organization to inventory under GOVERN 1.6?",
        "expected_answer": "The Framework sets no maximum; the number is not in the corpus.",
        "required_facts": [],
    },
]


def build(sources_dir: Path) -> list[dict]:
    pages_by_doc: dict[str, list[dict]] = {}
    for doc in corpus_documents():
        path = sources_dir / doc["file"]
        if not path.is_file():
            raise SystemExit(
                f"missing source PDF: {path}\nRun: python -m grounded_qa.cli fetch"
            )
        pages_by_doc[doc["sha256"]] = read_pdf(path, 200)

    out = []
    for spec in QUESTIONS:
        if not spec.get("answerable", True):
            out.append(
                {
                    "id": spec["id"],
                    "split": spec["split"],
                    "category": spec["category"],
                    "answerable": False,
                    "question": spec["question"],
                    "expected_answer": spec["expected_answer"],
                    "required_facts": spec["required_facts"],
                    "gold_evidence": [],
                    "review_status": (
                        "assistant-authored and source-checked; "
                        "independent human review pending"
                    ),
                }
            )
            continue
        gold = []
        for entry in [spec] + spec.get("also_relevant", []):
            gold.append(_locate(pages_by_doc, entry))
        out.append(
            {
                "id": spec["id"],
                "split": spec["split"],
                "category": spec["category"],
                "answerable": True,
                "question": spec["question"],
                "expected_answer": spec["expected_answer"],
                "required_facts": spec["required_facts"],
                "gold_evidence": gold,
                "review_status": (
                    "assistant-authored and source-checked; "
                    "independent human review pending"
                ),
            }
        )
    return out


def _locate(pages_by_doc: dict[str, list[dict]], entry: dict) -> dict:
    """Find the quote in the named page and record its true offsets."""
    doc = entry["document"]
    page_number = entry["page"]
    quote = entry["quote"]
    for page in pages_by_doc[doc]:
        if page["page"] != page_number:
            continue
        start = page["text"].find(quote)
        if start < 0:
            raise SystemExit(
                f"quote not found on page {page_number} of {doc[:8]}:\n{quote!r}"
            )
        return {
            "document_id": doc,
            "page": page_number,
            "start": start,
            "end": start + len(quote),
            "quote": quote,
        }
    raise SystemExit(f"page {page_number} not found in document {doc[:8]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify only")
    parser.add_argument(
        "--sources", type=Path, default=ROOT / "data" / "sources", help="PDF folder"
    )
    args = parser.parse_args()
    queries = build(args.sources)
    target = ROOT / "evaluation" / "queries-crossdoc.json"
    if args.check:
        print(f"OK: {len(queries)} questions, all gold spans located verbatim")
        return
    target.write_text(
        json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {target.relative_to(ROOT)} ({len(queries)} questions)")


if __name__ == "__main__":
    main()
