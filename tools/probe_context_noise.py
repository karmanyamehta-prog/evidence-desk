"""Measure whether distractor evidence causes the 3B model's abstentions.

The 3B model abstains on 7 answerable questions even though the gold text is
in the prompt. The evidence it was given was not just the gold chunk: the
retriever returns 5 chunks with 50-token overlap, so the prompt usually
contains near-duplicate copies of the gold text plus several irrelevant
passages.

Two readings fit that observation:

  A. The model is confused by the distractors and gives up.
     -> Giving it only the gold chunk should raise the answer rate.
  B. The model cannot use this particular text at all.
     -> Only the gold chunk will not help.

This script asks the same question twice per question, changing only the
evidence:

  - `retrieved`: the top-5 chunks, packed into the context budget, exactly as
    the service does it.
  - `gold_only`: just the chunk that contains the gold span.

Retrieval, model and prompt are otherwise identical.

Writes a JSON report to tools/context_noise_probe.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from grounded_qa.generation.client import SYSTEM  # noqa: E402
from grounded_qa.ingestion.chunking import chunk_pages  # noqa: E402
from grounded_qa.ingestion.pdf import read_pdf  # noqa: E402
from grounded_qa.models import Draft  # noqa: E402
from grounded_qa.retrieval.embeddings import LocalEmbeddings  # noqa: E402

DOC = "cde9f5bab41c68718f3413ab2564c2285ac9195fa9160a61cc5058dd9c35d6e7"
MODEL = "qwen2.5:3b-instruct-q4_K_M"
CONTEXT_CHARS = 4500
CHUNK_TOKENS = 350
TOP_K = 5


def ask(question: str, sources: dict[str, str]) -> dict:
    prompt = json.dumps(
        {
            "question": question,
            "evidence": {k: {"text": v} for k, v in sources.items()},
            "schema": Draft.model_json_schema(),
            "validation_error": "",
        }
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": Draft.model_json_schema(),
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 4096,
            "num_predict": 384,
            "num_thread": 4,
        },
    }
    with httpx.Client(timeout=180, trust_env=False) as client:
        response = client.post("http://localhost:11434/api/chat", json=payload)
    return json.loads(response.json()["message"]["content"])


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

        retrieved: dict[str, str] = {}
        used = 0
        for i in order:
            chunk = chunks[i]
            if used + len(chunk.text) > CONTEXT_CHARS:
                continue
            used += len(chunk.text)
            retrieved[f"S{len(retrieved) + 1}"] = chunk.text

        gold_chunk = next(
            (
                chunks[i]
                for i in order
                if any(covers(chunks[i], g) for g in q["gold_evidence"])
            ),
            None,
        )
        if gold_chunk is None:
            # Not retrieved at all: there is nothing to isolate.
            rows.append(
                {
                    "id": q["id"],
                    "retrieved_answered": None,
                    "gold_only_answered": None,
                    "gold_chunk_retrieved": False,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
            continue

        gold_only = {"S1": gold_chunk.text}

        retrieved_claims = ask(q["question"], retrieved).get("claims", [])
        gold_claims = ask(q["question"], gold_only).get("claims", [])
        row = {
            "id": q["id"],
            "retrieved_answered": bool(retrieved_claims),
            "gold_only_answered": bool(gold_claims),
            "gold_chunk_retrieved": True,
            "n_sources_retrieved": len(retrieved),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    comparable = [r for r in rows if r["gold_chunk_retrieved"]]
    summary = {
        "questions": len(rows),
        "comparable": len(comparable),
        "retrieved_answered": sum(r["retrieved_answered"] for r in comparable),
        "gold_only_answered": sum(r["gold_only_answered"] for r in comparable),
        "rescued_by_gold_only": [
            r["id"]
            for r in comparable
            if not r["retrieved_answered"] and r["gold_only_answered"]
        ],
        "lost_with_gold_only": [
            r["id"]
            for r in comparable
            if r["retrieved_answered"] and not r["gold_only_answered"]
        ],
    }
    (ROOT / "tools" / "context_noise_probe.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
