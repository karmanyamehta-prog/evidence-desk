"""Orchestration independent of the HTTP/UI layer."""

import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from grounded_qa.citations import filter_claims, sources_from_hits
from grounded_qa.generation.client import Generator
from grounded_qa.ingestion.chunking import PIPELINE_VERSION, chunk_pages
from grounded_qa.ingestion.pdf import document_id, read_pdf

logger = logging.getLogger(__name__)


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

    def ingest(
        self, path: Path, title: str | None = None, source_url: str = ""
    ) -> dict[str, Any]:
        with self.lock:
            did = document_id(path)
            logger.debug("ingest start: path=%s document_id=%s", path, did)
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
                logger.debug(
                    "ingest skip: document already indexed document_id=%s", did
                )
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
                metadata: dict[str, Any] = {
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
            logger.debug(
                "ingest complete: document_id=%s pages=%d chunks=%d",
                did,
                len(pages),
                len(chunks),
            )
            return metadata

    def list_documents(self) -> list[dict[str, Any]]:
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

    def _retrieve(
        self, question: str, document_ids: list[str] | None = None
    ) -> list[Any]:
        """Cosine retrieval, optionally fused with BM25 keyword search.

        The distance threshold (MIN_RETRIEVAL_DISTANCE) drops hits that are too
        far from the query in embedding space; they are unlikely to carry
        supporting evidence and would waste the context budget. A threshold of
        0.0 (the default) disables the filter.
        """
        threshold = self.settings.min_retrieval_distance
        hits = self.store.search(
            question, self.settings.top_k, document_ids, max_distance=threshold
        )
        if not self.settings.hybrid_search:
            return hits
        try:
            from grounded_qa.retrieval.bm25 import BM25Index, hybrid_search

            chunks = self.store.all_chunks(document_ids)
            if not chunks:
                return hits
            index = BM25Index(chunks)
            return hybrid_search(question, hits, index, top_k=self.settings.top_k)
        except ImportError:
            logger.warning("hybrid search unavailable: install rank-bm25 to enable it")
            return hits
        except Exception:
            logger.exception("hybrid search failed; falling back to cosine only")
            return hits

    def query(
        self, question: str, document_ids: list[str] | None = None
    ) -> dict[str, Any]:
        with self.lock:  # Bound CPU and memory: one local model request at a time.
            start = time.perf_counter()
            hits = self._retrieve(question, document_ids)
            retrieval_ms = (time.perf_counter() - start) * 1000
            sources = sources_from_hits(hits, self.settings.context_chars)
            draft, reason, attempts = None, "No usable evidence retrieved", []
            if sources:
                for attempt in range(2):
                    t = time.perf_counter()
                    error = ""
                    try:
                        raw_draft = self.generator.generate(
                            question, sources, reason if attempt else "", attempt
                        )
                        if raw_draft.abstain:
                            # The model declined to answer: a legitimate abstention,
                            # not a validation failure. Accept and stop.
                            draft = raw_draft
                            reason = "Insufficient evidence"
                        else:
                            # Per-claim filtering: keep valid claims, surface errors
                            # for invalid ones. One bad citation no longer kills a
                            # good answer (CHANGES.md: q13/q14 each had a good claim).
                            valid_claims, invalid_msgs = filter_claims(
                                raw_draft, sources
                            )
                            if valid_claims:
                                draft = raw_draft.model_copy(
                                    update={"claims": valid_claims}
                                )
                                reason = ""
                                error = "; ".join(invalid_msgs)
                                if invalid_msgs:
                                    logger.debug(
                                        "query partial validation attempt=%d dropped=%d",
                                        attempt,
                                        len(invalid_msgs),
                                    )
                            else:
                                error = reason = "; ".join(invalid_msgs)
                                logger.debug(
                                    "query full reject attempt=%d error=%s",
                                    attempt,
                                    error,
                                )
                                draft = None
                    except (ValidationError, ValueError) as exc:
                        error = reason = str(exc)
                        logger.debug(
                            "query validation failure attempt=%d error=%s",
                            attempt,
                            error,
                        )
                        draft = None
                    attempts.append(
                        {
                            "attempt": attempt,
                            "raw": getattr(self.generator, "last_raw", None),
                            "error": error,
                            "ms": (time.perf_counter() - t) * 1000,
                        }
                    )
                    if draft is not None:
                        break  # Accepted (fully or partially); no retry needed.
            # Distinguish a model-initiated abstention from a validation rejection.
            if draft is None:
                status = "rejected" if sources else "abstained"
            elif draft.abstain:
                # Empty claims: either the model abstained, or every claim was
                # filtered out by validation. The attempt log tells them apart.
                status = (
                    "rejected" if attempts and attempts[-1]["error"] else "abstained"
                )
            else:
                status = "answered"
            abstain = status != "answered"
            claims = [] if abstain else [c.model_dump() for c in draft.claims]
            total_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "query complete: status=%s attempts=%d total_ms=%.0f",
                status,
                len(attempts),
                total_ms,
            )
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
