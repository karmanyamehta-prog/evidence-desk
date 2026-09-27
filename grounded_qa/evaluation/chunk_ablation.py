"""Compare page-bound token windows with recursive/action-aware chunks on dev only."""

import json
from pathlib import Path
import numpy as np
from grounded_qa.config import Settings
from grounded_qa.ingestion.pdf import read_pdf, document_id
from grounded_qa.ingestion.chunking import chunk_pages
from grounded_qa.retrieval.embeddings import LocalEmbeddings
from grounded_qa.models import Chunk


def main():
    s = Settings()
    path = s.data_dir / "sources" / "rmf.pdf"
    pages = read_pdf(path)
    did = document_id(path)
    e = LocalEmbeddings(s.embedding_model, s.embedding_revision, 1)
    fixed = []
    for page in pages:
        offsets = e.tokenizer(
            page["text"], add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        for i in range(0, len(offsets), 300):
            window = offsets[i : i + 350]
            if not window:
                continue
            a, b = window[0][0], window[-1][1]
            fixed.append(
                Chunk(
                    f"{page['page']}-{a}",
                    page["text"][a:b],
                    {"page": page["page"], "start": a, "end": b},
                )
            )
            if i + 350 >= len(offsets):
                break
    recursive = chunk_pages(pages, did, "RMF", e.tokenizer, 350, 50)
    queries = [
        q
        for q in json.loads(Path("evaluation/queries.json").read_text())
        if q["split"] == "dev" and q["answerable"]
    ]
    qv = np.asarray(e.queries([q["question"] for q in queries]))
    output = []
    for name, chunks in [("fixed", fixed), ("recursive-action-aware", recursive)]:
        v = np.asarray(e.passages([c.text for c in chunks]))
        recalls = []
        for q, scores in zip(queries, qv @ v.T):
            hits = [chunks[i] for i in np.argsort(-scores)[:5]]
            covered = [
                any(
                    h.metadata["page"] == g["page"]
                    and h.metadata["start"] <= g["start"]
                    and h.metadata["end"] >= g["end"]
                    for h in hits
                )
                for g in q["gold_evidence"]
            ]
            recalls.append(float(np.mean(covered)))
        output.append(
            {
                "strategy": name,
                "chunks": len(chunks),
                "dev_gold_span_recall_at_5": float(np.mean(recalls)),
            }
        )
    Path("evaluation/chunk-ablation.json").write_text(json.dumps(output, indent=2))
    print(output)


if __name__ == "__main__":
    main()
