"""Orchestration independent of the HTTP/UI layer."""

import json
import threading
import time
from pathlib import Path
from pydantic import ValidationError
from grounded_qa.ingestion.pdf import read_pdf, document_id
from grounded_qa.ingestion.chunking import chunk_pages, PIPELINE_VERSION
from grounded_qa.citations import sources_from_hits, validate
from grounded_qa.generation.client import Generator


class Service:
    def __init__(self, settings, embedder=None, store=None, generator=None):
        self.settings = settings
        self.lock = threading.RLock()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        if embedder is None:
            from grounded_qa.retrieval.embeddings import LocalEmbeddings

            embedder = LocalEmbeddings(
                settings.embedding_model,
                settings.embedding_revision,
                settings.cpu_threads,
            )
        self.embedder = embedder
        if store is None:
            from grounded_qa.retrieval.store import VectorStore

            fingerprint = json.dumps(
                [
                    settings.embedding_model,
                    settings.embedding_revision,
                    settings.chunk_tokens,
                    settings.chunk_overlap,
                    PIPELINE_VERSION,
                ]
            )
            store = VectorStore(settings.data_dir / "chroma", fingerprint, embedder)
        self.store = store
        self.generator = generator or Generator(settings)

    def ingest(self, path: Path, title=None, source_url=""):
        with self.lock:
            did = document_id(path)
            pages = read_pdf(path, self.settings.max_pages)
            chunks = chunk_pages(
                pages,
                did,
                title or path.name,
                self.embedder.tokenizer,
                self.settings.chunk_tokens,
                self.settings.chunk_overlap,
                source_url,
            )
            target = self.settings.data_dir / "documents" / did
            existing = self.store.collection.get(where={"document_id": did}, include=[])
            if (
                set(existing["ids"]) == {c.id for c in chunks}
                and (target / "metadata.json").exists()
            ):
                return json.loads((target / "metadata.json").read_text())
            target.mkdir(parents=True, exist_ok=True)
            (target / "source.pdf").write_bytes(path.read_bytes())
            (target / "pages.json").write_text(
                json.dumps(pages, ensure_ascii=False), encoding="utf-8"
            )
            try:
                self.store.add(chunks)
            except Exception:
                self.store.delete(
                    did
                )  # No partially indexed document after a failed batch.
                raise
            metadata = {
                "document_id": did,
                "title": title or path.name,
                "pages": len(pages),
                "chunks": len(chunks),
                "source_url": source_url,
            }
            (target / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            return metadata

    def query(self, question, document_ids=None):
        with self.lock:  # Bound CPU and memory: one local model request at a time.
            start = time.perf_counter()
            hits = self.store.search(question, self.settings.top_k, document_ids)
            retrieval_ms = (time.perf_counter() - start) * 1000
            sources = sources_from_hits(hits, self.settings.context_chars)
            draft, reason = None, "No usable evidence retrieved"
            if sources:
                for attempt in range(2):
                    try:
                        draft = self.generator.generate(
                            question, sources, reason if attempt else ""
                        )
                        validate(draft, sources)
                        reason = "Insufficient evidence" if draft.abstain else ""
                        break
                    except (ValidationError, ValueError) as exc:
                        reason, draft = str(exc), None
            abstain = draft is None or draft.abstain
            claims = [] if abstain else [c.model_dump() for c in draft.claims]
            total_ms = (time.perf_counter() - start) * 1000
            return {
                "question": question,
                "abstain": abstain,
                "reason": reason,
                "claims": claims,
                "mode": self.settings.llm_provider,
                "sources": sources,
                "retrieved": [
                    {
                        "id": h.chunk.id,
                        "text": h.chunk.text,
                        "metadata": h.chunk.metadata,
                        "distance": h.distance,
                    }
                    for h in hits
                ],
                "timing_ms": {
                    "retrieval": retrieval_ms,
                    "generation_and_validation": total_ms - retrieval_ms,
                    "total": total_ms,
                },
            }
