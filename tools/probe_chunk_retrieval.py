"""Measure gold-span recall@5 at each chunk size, without running the model.

`tools/probe_chunk_size.py` measures how often the 3B model answers, and the
answer rate turned out to be flat across chunk sizes. That leaves two possible
causes that the answer rate alone cannot separate:

1. The model still abstains even when the gold span is retrieved.
2. Smaller chunks changed retrieval, so the gold span is no longer retrieved
   and the model has nothing to answer from.

This script measures the second directly. It runs the same chunking and the
same embedding model over the same questions as the answer-rate probe, but it
makes no model calls, so it is fast enough to re-run freely.

A gold span counts as retrieved when some chunk in the top 5 covers both its
start and its end offset on its page.

Writes a JSON report to tools/chunk_retrieval_probe.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from grounded_qa.ingestion.chunking import chunk_pages  # noqa: E402
from grounded_qa.ingestion.pdf import read_pdf  # noqa: E402
from grounded_qa.retrieval.embeddings import LocalEmbeddings  # noqa: E402

DOC = "cde9f5bab41c68718f3413ab2564c2285ac9195fa9160a61cc5058dd9c35d6e7"
SIZES = [350, 250, 200, 150]
TOP_K = 5


def covers(chunk, gold) -> bool:
    """Does this chunk contain the whole gold span?"""
    meta = chunk.metadata
    return (
        meta["page"] == gold["page"]
        and meta["start"] <= gold["start"]
        and meta["end"] >= gold["end"]
    )


def main() -> None:
    embedder = LocalEmbeddings(
        "intfloat/e5-small-v2", "ffb93f3bd4047442299a41ebb6fa998a38507c52", 4, None
    )
    pages = read_pdf(ROOT / "data" / "sources" / "nist-ai-100-1-rmf.pdf", 200)
    queries = json.loads(
        (ROOT / "evaluation" / "queries.json").read_text(encoding="utf-8")
    )
    answerable = [q for q in queries if q["answerable"]]

    report = []
    for size in SIZES:
        chunks = chunk_pages(pages, DOC, "RMF", embedder.tokenizer, size, 50)
        vectors = np.array(embedder.passages([c.embed_prefix + c.text for c in chunks]))
        hits = 0
        misses: list[str] = []
        top1_ranks: list[int] = []
        for q in answerable:
            query_vector = np.array(embedder.queries([q["question"]])[0])
            distances = 1 - vectors @ query_vector
            order = np.argsort(distances)[:TOP_K]
            top = [chunks[i] for i in order]
            covered = any(
                covers(chunk, gold) for chunk in top for gold in q["gold_evidence"]
            )
            if covered:
                hits += 1
            else:
                misses.append(q["id"])
                continue
            # Where does the first covering chunk rank?
            for rank, chunk in enumerate(top, start=1):
                if any(covers(chunk, gold) for gold in q["gold_evidence"]):
                    top1_ranks.append(rank)
                    break
        entry = {
            "chunk_tokens": size,
            "chunks": len(chunks),
            "gold_span_recall_at_5": hits,
            "answerable": len(answerable),
            "recall": round(hits / len(answerable), 4),
            "missed_ids": misses,
            "mean_covering_rank": (
                round(float(np.mean(top1_ranks)), 3) if top1_ranks else None
            ),
        }
        report.append(entry)
        print(json.dumps(entry), flush=True)
        (ROOT / "tools" / "chunk_retrieval_probe.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
