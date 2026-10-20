"""Measure whether the 3B model's abstentions are caused by evidence position.

The answer-rate probe (`tools/probe_chunk_size.py`) found the answer rate flat
across chunk sizes, even though the gold span's depth inside its chunk fell
three-fold. That refutes a simple "lost in the middle" explanation, but it does
not say what the cause is.

This script separates two competing explanations with a controlled experiment:

  A. The evidence is hard to use because the answer is buried in it.
     -> Moving the answer to the front of its chunk should raise the answer
        rate.
  B. The model abstains for some other reason (it does not judge the evidence
     to answer the question).
     -> The same move should change nothing.

For each question it asks the model twice with the same evidence chunks:

  - `buried`: the chunks exactly as retrieved.
  - `front`: the same chunks with the gold span moved to the front of its
    chunk, so the answer is the first thing the model reads.

Retrieval, model, prompt and evidence content are otherwise identical. The
text is only reordered, never edited.

Writes a JSON report to tools/prompt_abstention_probe.json.
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


def fronted(chunk, gold_evidence) -> str:
    """Return the chunk text with the gold span moved to the front.

    Chunk metadata holds `start` and `end` offsets into the normalized page
    text, and the chunk text is exactly that slice, so the gold span's local
    offset is `gold.start - chunk.start`.
    """
    meta = chunk.metadata
    for gold in gold_evidence:
        if meta["page"] != gold["page"]:
            continue
        if not (meta["start"] <= gold["start"] and meta["end"] >= gold["end"]):
            continue
        local = gold["start"] - meta["start"]
        span = chunk.text[local : local + (gold["end"] - gold["start"])]
        if not span.strip():
            continue
        head = chunk.text[:local]
        tail = chunk.text[local + (gold["end"] - gold["start"]) :]
        return f"{span} {head} {tail}".strip()
    return chunk.text


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

        buried: dict[str, str] = {}
        front: dict[str, str] = {}
        used = 0
        for i in order:
            chunk = chunks[i]
            if used + len(chunk.text) > CONTEXT_CHARS:
                continue
            used += len(chunk.text)
            key = f"S{len(buried) + 1}"
            buried[key] = chunk.text
            front[key] = fronted(chunk, q["gold_evidence"])

        buried_claims = ask(q["question"], buried).get("claims", [])
        front_claims = ask(q["question"], front).get("claims", [])
        row = {
            "id": q["id"],
            "buried_answered": bool(buried_claims),
            "front_answered": bool(front_claims),
            "reordered": buried != front,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    summary = {
        "questions": len(rows),
        "buried_answered": sum(r["buried_answered"] for r in rows),
        "front_answered": sum(r["front_answered"] for r in rows),
        "rescued_by_fronting": [
            r["id"] for r in rows if not r["buried_answered"] and r["front_answered"]
        ],
        "lost_by_fronting": [
            r["id"] for r in rows if r["buried_answered"] and not r["front_answered"]
        ],
    }
    (ROOT / "tools" / "prompt_abstention_probe.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
