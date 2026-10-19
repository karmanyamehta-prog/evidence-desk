"""Development-only retrieval ablation using exact cosine as a reference.

Span-hit rate is deliberately not called human-judged precision.
"""

import json
import time
from pathlib import Path

import numpy as np

from grounded_qa.config import Settings
from grounded_qa.ingestion.chunking import chunk_pages
from grounded_qa.ingestion.pdf import document_id, read_pdf
from grounded_qa.retrieval.embeddings import LocalEmbeddings


def main():
    settings = Settings()
    path = settings.data_dir / "sources" / "rmf.pdf"
    pages = read_pdf(path)
    did = document_id(path)
    queries = [
        q
        for q in json.loads(Path("evaluation/queries.json").read_text())
        if q["split"] == "dev" and q["answerable"]
    ]
    results = []
    configs = [
        ("intfloat/e5-small-v2", "ffb93f3bd4047442299a41ebb6fa998a38507c52"),
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        ),
        (
            "sentence-transformers/all-mpnet-base-v2",
            "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        ),
    ]
    common = None
    for name, revision in configs:
        try:
            start = time.perf_counter()
            e = LocalEmbeddings(name, revision, threads=1)
            if common is None:
                common = chunk_pages(pages, did, "RMF", e.tokenizer, 220, 30)
            # Same actual passages for all models; _encode refuses truncation.
            embeddings = np.asarray(e.passages([c.text for c in common]))
            index_seconds = time.perf_counter() - start
            t = time.perf_counter()
            qs = np.asarray(e.queries([q["question"] for q in queries]))
            query_seconds = time.perf_counter() - t
            hit = []
            recalls = []
            for q, scores in zip(queries, qs @ embeddings.T, strict=False):
                retrieved = [common[i] for i in np.argsort(-scores)[:5]]
                covered = [
                    any(
                        c.metadata["page"] == g["page"]
                        and c.metadata["start"] <= g["start"]
                        and c.metadata["end"] >= g["end"]
                        for c in retrieved
                    )
                    for g in q["gold_evidence"]
                ]
                hit.append(float(any(covered)))
                recalls.append(float(np.mean(covered)))
            results.append(
                {
                    "model": name,
                    "revision": revision,
                    "chunks": len(common),
                    "dev_answerable_queries": len(queries),
                    "gold_span_hit_rate_at_5": float(np.mean(hit)),
                    "gold_span_recall_at_5": float(np.mean(recalls)),
                    "cold_load_and_index_seconds": index_seconds,
                    "batch_query_seconds": query_seconds,
                }
            )
            del e, embeddings, qs
        except Exception as exc:
            results.append({"model": name, "error": str(exc)})
        Path("evaluation/ablation.json").write_text(json.dumps(results, indent=2))
        print(results[-1], flush=True)


if __name__ == "__main__":
    main()
