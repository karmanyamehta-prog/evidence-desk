"""Orchestration independent of the HTTP/UI layer."""

import json
import shutil
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
                settings.embedding_prefixes,
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
            # Index first, then write files: a failed embed leaves nothing behind.
            created = not target.exists()
            try:
                self.store.add(chunks)
                target.mkdir(parents=True, exist_ok=True)
                (target / "source.pdf").write_bytes(path.read_bytes())
                (target / "pages.json").write_text(
                    json.dumps(pages, ensure_ascii=False), encoding="utf-8"
                )
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
            except Exception:
                self.store.delete(did)  # No partially indexed document.
                if created:
                    shutil.rmtree(target, ignore_errors=True)
                raise
            return metadata

    def list_documents(self):
        """Documents present in the active index namespace, not stray metadata."""
        indexed = {
            m["document_id"]
            for m in self.store.collection.get(include=["metadatas"])["metadatas"]
        }
        docs = []
        for did in sorted(indexed):
            meta = self.settings.data_dir / "documents" / did / "metadata.json"
            if meta.exists():
                docs.append(json.loads(meta.read_text()))
        return docs

    def query(self, question, document_ids=None):
        with self.lock:  # Bound CPU and memory: one local model request at a time.
            start = time.perf_counter()
            hits = self.store.search(question, self.settings.top_k, document_ids)
            retrieval_ms = (time.perf_counter() - start) * 1000
            sources = sources_from_hits(hits, self.settings.context_chars)
            draft, reason, attempts = None, "No usable evidence retrieved", []
            if sources:
                for attempt in range(2):
                    t = time.perf_counter()
                    error = ""
                    try:
                        draft = self.generator.generate(
                            question, sources, reason if attempt else "", attempt
                        )
                        validate(draft, sources)
                        reason = "Insufficient evidence" if draft.abstain else ""
                    except (ValidationError, ValueError) as exc:
                        error = reason = str(exc)
                        draft = None
                    attempts.append(
                        {
                            "attempt": attempt,
                            "raw": getattr(self.generator, "last_raw", None),
                            "error": error,
                            "ms": (time.perf_counter() - t) * 1000,
                        }
                    )
                    if not error:
                        break
            if draft is None:
                status = "rejected" if sources else "abstained"
            else:
                status = "abstained" if draft.abstain else "answered"
            abstain = status != "answered"
            claims = [] if abstain else [c.model_dump() for c in draft.claims]
            total_ms = (time.perf_counter() - start) * 1000
            return {
                "question": question,
                "status": status,
                "abstain": abstain,
                "reason": reason,
                "claims": claims,
                "mode": self.settings.llm_provider,
                "sources": sources,
                "dropped_by_context_budget": len(hits) - len(sources),
                "attempts": attempts,
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
