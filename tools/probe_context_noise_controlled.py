"""Controlled test: do distractor passages cause the 3B model's abstentions?

Earlier probes suggested that giving the model only the gold chunk raises the
answer rate (14 answered versus 11 with the full retrieved context), but
those two conditions were measured in separate runs, and the model turned out
to drift between runs. A difference between conditions measured in different
runs is not evidence.

This script removes that confound. For each question it issues the two
conditions **alternately, in the same session, with the same request options
the service uses** (`keep_alive`, `num_gpu: 0`):

  - `retrieved`: the top-5 chunks packed into the context budget.
  - `gold_only`: just the chunk containing the gold span.

Alternating the conditions means any slow drift in the model or the machine
affects both arms equally instead of being attributed to one of them. Each
condition is asked three times per question.

It also records the first call for each condition separately, because the
evaluation run showed a warm-up effect: its first repeat answered 9 of 24
while the later repeats answered 11 of 24.

Writes a JSON report to tools/context_noise_controlled.json.
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
ROUNDS = 3


def ask(question: str, sources: dict[str, str]) -> dict:
    """Ask exactly as the service does, including keep_alive and num_gpu."""
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
        "keep_alive": "5m",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 4096,
            "num_predict": 384,
            "num_thread": 4,
            "num_gpu": 0,
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
            rows.append(
                {
                    "id": q["id"],
                    "retrieved": [],
                    "gold_only": [],
                    "gold_chunk_retrieved": False,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
            continue

        gold_only = {"S1": gold_chunk.text}

        # Alternate the conditions so drift hits both arms equally.
        retrieved_answers: list[bool] = []
        gold_answers: list[bool] = []
        for _ in range(ROUNDS):
            retrieved_answers.append(
                bool(ask(q["question"], retrieved).get("claims", []))
            )
            gold_answers.append(bool(ask(q["question"], gold_only).get("claims", [])))

        row = {
            "id": q["id"],
            "retrieved": retrieved_answers,
            "gold_only": gold_answers,
            "gold_chunk_retrieved": True,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    comparable = [r for r in rows if r["gold_chunk_retrieved"]]

    def rate(key: str) -> float:
        answered = sum(1 for r in comparable if any(r[key]))
        return round(answered / len(comparable), 4) if comparable else 0.0

    def mean_rate(key: str) -> float:
        total = sum(sum(r[key]) for r in comparable)
        calls = sum(len(r[key]) for r in comparable)
        return round(total / calls, 4) if calls else 0.0

    flipped = [
        r["id"] for r in comparable if any(r["retrieved"]) != any(r["gold_only"])
    ]
    summary = {
        "questions": len(comparable),
        "rounds": ROUNDS,
        "retrieved_answered_any": sum(1 for r in comparable if any(r["retrieved"])),
        "gold_only_answered_any": sum(1 for r in comparable if any(r["gold_only"])),
        "retrieved_answer_rate_per_call": mean_rate("retrieved"),
        "gold_only_answer_rate_per_call": mean_rate("gold_only"),
        "retrieved_answer_rate_any": rate("retrieved"),
        "gold_only_answer_rate_any": rate("gold_only"),
        "questions_whose_status_differs_between_conditions": flipped,
        "questions_unstable_within_a_condition": [
            r["id"]
            for r in comparable
            if len(set(r["retrieved"])) > 1 or len(set(r["gold_only"])) > 1
        ],
    }
    (ROOT / "tools" / "context_noise_controlled.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
