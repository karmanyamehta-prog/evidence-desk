"""Measure the answer rate of the 3B model at different chunk sizes.

Retrieval quality and answer quality are different things. This script holds
the question set fixed and varies only the chunk size, so the effect of the
chunk size on the model's willingness to answer can be measured on its own.

The suspected mechanism: a 3B model often abstains when the answer sits deep
inside a long retrieved chunk ("lost in the middle"), and answers when the
same text leads the chunk.

Writes a JSON report to tools/chunk_size_probe.json.
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
SIZES = [350, 250, 200, 150]


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
        answered, abstained = [], []
        depths = []
        for q in answerable:
            query_vector = np.array(embedder.queries([q["question"]])[0])
            distances = 1 - vectors @ query_vector
            order = np.argsort(distances)[:5]
            # How deep is the gold text inside its own chunk?
            for gold in q["gold_evidence"]:
                for chunk in chunks:
                    meta = chunk.metadata
                    if (
                        meta["page"] == gold["page"]
                        and meta["start"] <= gold["start"]
                        and meta["end"] >= gold["end"]
                    ):
                        depths.append(gold["start"] - meta["start"])
                        break
            # Pack the context budget best-first, as the service does.
            sources: dict[str, str] = {}
            used = 0
            for i in order:
                text = chunks[i].text
                if used + len(text) > CONTEXT_CHARS:
                    continue
                used += len(text)
                sources[f"S{len(sources) + 1}"] = text
            claims = ask(q["question"], sources).get("claims", [])
            if claims:
                answered.append(q["id"])
            else:
                abstained.append(q["id"])
        entry = {
            "chunk_tokens": size,
            "chunks": len(chunks),
            "answered": len(answered),
            "answerable": len(answerable),
            "abstained_ids": abstained,
            "gold_depth_median": float(np.median(depths)) if depths else None,
            "gold_depth_p90": float(np.percentile(depths, 90)) if depths else None,
        }
        report.append(entry)
        print(json.dumps(entry), flush=True)
        (ROOT / "tools" / "chunk_size_probe.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
