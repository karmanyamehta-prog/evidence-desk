"""Measure the run-to-run noise floor of the 3B model on identical inputs.

The evaluation uses `temperature: 0` and a fixed `seed: 42`, yet q09 and q14
changed status between identical repeats. That means a single comparison
between two evidence conditions cannot be read as a result: the difference
between "11 answered" and "14 answered" may be noise.

This script establishes the noise floor by asking the **same question with the
same evidence twice** and recording how often the answer status differs. It
also repeats the gold-only condition, so the two conditions can be compared
against a measured baseline rather than an assumed one.

Every call uses the identical payload the service sends.

Writes a JSON report to tools/noise_floor_probe.json.
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
            continue

        # Three identical calls on the same evidence: the noise floor.
        same = [bool(ask(q["question"], retrieved).get("claims", [])) for _ in range(3)]
        # Two identical calls on gold-only evidence, for the comparison arm.
        gold_only = {"S1": gold_chunk.text}
        only = [bool(ask(q["question"], gold_only).get("claims", [])) for _ in range(2)]

        row = {
            "id": q["id"],
            "retrieved_repeat_answers": same,
            "retrieved_stable": len(set(same)) == 1,
            "gold_only_repeat_answers": only,
            "gold_only_stable": len(set(only)) == 1,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    unstable_same = [r["id"] for r in rows if not r["retrieved_stable"]]
    unstable_only = [r["id"] for r in rows if not r["gold_only_stable"]]
    summary = {
        "questions": len(rows),
        "identical_repeats_flipped": unstable_same,
        "flip_rate_identical_inputs": round(len(unstable_same) / len(rows), 3)
        if rows
        else None,
        "gold_only_repeats_flipped": unstable_only,
    }
    (ROOT / "tools" / "noise_floor_probe.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
