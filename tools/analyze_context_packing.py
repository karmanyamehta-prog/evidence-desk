"""Should the context packer be changed? A measurement, not a guess.

The current packer takes hits in distance order and adds each whole chunk
while it fits. It is optimal for the *number* of chunks, but it commits the
budget greedily: for q06 it spent 4018 of 4500 characters on three chunks
and left 482, while the gold chunk needed 791.

The question is not "is the packer buggy" — it is "does a better policy
exist, and would it change the evidence the model actually sees?"

This script answers that with no model calls. It compares the shipped
greedy packer against a knapsack that maximises the total *retrieval
similarity* (1 - cosine distance) of the selected chunks subject to the
character budget, using the real measured distances rather than a rank
proxy. A rank proxy is not good enough: weighting by 1/(i+1) makes the
first chunk dominate so strongly that the knapsack always reproduces the
greedy answer, which would make the comparison vacuous.

It also enumerates, for the question that fails, every feasible subset of
the retrieved chunks and reports the best achievable value with and without
the gold chunk. That answers the question that actually matters: was the
gold chunk reachable at all, and what would it have cost to reach it?

Writes a JSON report to tools/context_packing_analysis.json.
"""

from __future__ import annotations

import itertools
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


def greedy_pack(lengths: list[int], budget: int) -> list[int]:
    """The shipped policy: take chunks in rank order while they fit."""
    chosen, used = [], 0
    for i, length in enumerate(lengths):
        if used + length <= budget:
            chosen.append(i)
            used += length
    return chosen


def knapsack_pack(lengths: list[int], values: list[float], budget: int) -> list[int]:
    """Maximise the total value of the selected chunks within the budget."""
    n = len(lengths)
    best = [[0.0] * (budget + 1) for _ in range(n + 1)]
    take = [[False] * (budget + 1) for _ in range(n + 1)]
    for i in range(n):
        for c in range(budget + 1):
            best[i + 1][c] = best[i][c]
            take[i + 1][c] = False
            if lengths[i] <= c:
                with_i = best[i][c - lengths[i]] + values[i]
                if with_i > best[i + 1][c]:
                    best[i + 1][c] = with_i
                    take[i + 1][c] = True
    chosen, c, i = [], budget, n
    while i > 0:
        if take[i][c]:
            chosen.append(i - 1)
            c -= lengths[i - 1]
        i -= 1
    return sorted(chosen)


def best_subset_with_gold(
    lengths: list[int], values: list[float], budget: int, gold_index: int
) -> tuple[float, list[int]] | None:
    """Highest total value achievable by a feasible subset that includes gold."""
    best: tuple[float, list[int]] | None = None
    n = len(lengths)
    for size in range(1, n + 1):
        for combo in itertools.combinations(range(n), size):
            if gold_index not in combo:
                continue
            if sum(lengths[i] for i in combo) > budget:
                continue
            total = sum(values[i] for i in combo)
            if best is None or total > best[0]:
                best = (total, list(combo))
    return best


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
        lengths = [len(chunks[i].text) for i in order]
        values = [float(1 - distances[i]) for i in order]

        greedy = greedy_pack(lengths, CONTEXT_CHARS)
        knap = knapsack_pack(lengths, values, CONTEXT_CHARS)

        gold_index = next(
            (
                r
                for r, i in enumerate(order)
                if any(covers(chunks[i], g) for g in q["gold_evidence"])
            ),
            None,
        )

        def gold_reaches(chosen, _order=order, _q=q) -> bool:
            text = fold(" ".join(chunks[_order[i]].text for i in chosen))
            return any(fold(g["quote"]) in text for g in _q["gold_evidence"])

        # How much similarity would be given up to include the gold chunk?
        with_gold = None
        if gold_index is not None:
            with_gold = best_subset_with_gold(
                lengths, values, CONTEXT_CHARS, gold_index
            )
        greedy_value = sum(values[i] for i in greedy)
        gold_cost = round(greedy_value - with_gold[0], 6) if with_gold else None

        rows.append(
            {
                "id": q["id"],
                "chunk_lengths": lengths,
                "distances": [round(1 - v, 6) for v in values],
                "gold_chunk_rank": None if gold_index is None else gold_index + 1,
                "greedy_chunks": greedy,
                "knapsack_chunks": knap,
                "greedy_chars": sum(lengths[i] for i in greedy),
                "knapsack_chars": sum(lengths[i] for i in knap),
                "packing_differs": greedy != knap,
                "greedy_gold_in_prompt": gold_reaches(greedy),
                "knapsack_gold_in_prompt": gold_reaches(knap),
                "greedy_total_similarity": round(greedy_value, 6),
                "best_with_gold_total_similarity": (
                    round(with_gold[0], 6) if with_gold else None
                ),
                "best_with_gold_chunks": with_gold[1] if with_gold else None,
                "similarity_given_up_to_include_gold": gold_cost,
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    differs = [r["id"] for r in rows if r["packing_differs"]]
    summary = {
        "questions": len(rows),
        "packings_that_differ": differs,
        "gold_in_prompt_under_greedy": sum(
            1 for r in rows if r["greedy_gold_in_prompt"]
        ),
        "gold_in_prompt_under_knapsack": sum(
            1 for r in rows if r["knapsack_gold_in_prompt"]
        ),
        "questions_where_the_knapsack_gains_the_gold_span": [
            r["id"]
            for r in rows
            if r["knapsack_gold_in_prompt"] and not r["greedy_gold_in_prompt"]
        ],
        "questions_where_the_knapsack_loses_the_gold_span": [
            r["id"]
            for r in rows
            if r["greedy_gold_in_prompt"] and not r["knapsack_gold_in_prompt"]
        ],
        "mean_chunks_greedy": round(
            sum(len(r["greedy_chunks"]) for r in rows) / len(rows), 2
        ),
        "mean_chunks_knapsack": round(
            sum(len(r["knapsack_chunks"]) for r in rows) / len(rows), 2
        ),
    }
    (ROOT / "tools" / "context_packing_analysis.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
