"""Was the gold chunk actually inside the "retrieved" prompt of the noise probe?

`tools/probe_context_noise_controlled.py` found the model answers more often
with the gold chunk alone than with the full retrieved context. Before that is
read as "distractors make the model give up", there is a confound to rule out:
the retrieved context is packed into a 4500-character budget, and the chunks
are long, so only 2-3 of the top 5 fit. If the gold chunk was **dropped by
the budget** in the retrieved arm, then the comparison is not "gold plus
distractors" versus "gold alone" — it is "gold that is present" versus "gold
that is missing", which is the retrieval/budget failure already ruled out
elsewhere, wearing a new hat.

This script reconstructs both prompts for every answerable question with no
model calls and reports, for each, whether the gold span reached the prompt.

Writes a JSON report to tools/context_noise_gold_presence.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from grounded_qa.citations import fold  # noqa: E402
from grounded_qa.ingestion.chunking import chunk_pages  # noqa: E402
from grounded_qa.ingestion.pdf import read_pdf  # noqa: E402
from grounded_qa.retrieval.embeddings import LocalEmbeddings  # noqa: E402

DOC = "cde9f5bab41c68718f3413ab2564c2285ac9195fa9160a61cc5058dd9c35d6e7"
CONTEXT_CHARS = 4500
CHUNK_TOKENS = 350
TOP_K = 5


def covers(chunk, gold) -> bool:
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

    chunks = chunk_pages(pages, DOC, "RMF", embedder.tokenizer, CHUNK_TOKENS, 50)
    vectors = np.array(embedder.passages([c.embed_prefix + c.text for c in chunks]))

    rows = []
    for q in answerable:
        query_vector = np.array(embedder.queries([q["question"]])[0])
        distances = 1 - vectors @ query_vector
        order = np.argsort(distances)[:TOP_K]

        # The probe's "retrieved" arm: pack top-5 in distance order until the
        # budget is spent.
        retrieved: dict[str, str] = {}
        used = 0
        for i in order:
            chunk = chunks[i]
            if used + len(chunk.text) > CONTEXT_CHARS:
                continue
            used += len(chunk.text)
            retrieved[f"S{len(retrieved) + 1}"] = chunk.text

        retrieved_text = fold(" ".join(retrieved.values()))
        gold_in_retrieved = any(
            fold(g["quote"]) in retrieved_text for g in q["gold_evidence"]
        )

        gold_chunk = next(
            (
                chunks[i]
                for i in order
                if any(covers(chunks[i], g) for g in q["gold_evidence"])
            ),
            None,
        )
        gold_rank = None
        if gold_chunk is not None:
            gold_rank = int(np.where(order == chunks.index(gold_chunk))[0][0]) + 1

        rows.append(
            {
                "id": q["id"],
                "category": q["category"],
                "n_chunks_packed": len(retrieved),
                "chars_packed": used,
                "chunks_in_top_k": TOP_K,
                "gold_chunk_in_top_k": gold_chunk is not None,
                "gold_chunk_rank": gold_rank,
                "gold_span_in_retrieved_prompt": gold_in_retrieved,
                "chunk_char_lengths": [len(chunks[i].text) for i in order],
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    packed = [r for r in rows if r["gold_chunk_in_top_k"]]
    summary = {
        "questions": len(rows),
        "questions_with_gold_chunk_in_top5": len(packed),
        "of_those_gold_span_actually_in_the_packed_prompt": sum(
            1 for r in packed if r["gold_span_in_retrieved_prompt"]
        ),
        "gold_span_missing_because_the_budget_dropped_it": [
            r["id"] for r in packed if not r["gold_span_in_retrieved_prompt"]
        ],
        "mean_chunks_packed": round(
            sum(r["n_chunks_packed"] for r in rows) / len(rows), 2
        ),
        "mean_chars_packed": round(sum(r["chars_packed"] for r in rows) / len(rows), 1),
    }
    (ROOT / "tools" / "context_noise_gold_presence.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
