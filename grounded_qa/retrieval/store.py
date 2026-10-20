"""Persistent cosine search; API owns the only writer."""

import hashlib

from grounded_qa.models import Chunk, Hit


class VectorStore:
    def __init__(self, path, fingerprint, embedder):
        import chromadb
        from chromadb.config import Settings

        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            str(path), settings=Settings(anonymized_telemetry=False)
        )
        name = "evidence-" + hashlib.sha256(fingerprint.encode()).hexdigest()[:20]
        self.collection = self.client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks):
        for start in range(0, len(chunks), 32):
            batch = chunks[start : start + 32]
            # Prepend the section/action ID so exact-ID queries match. The stored
            # document text stays the verbatim source passage.
            vectors = self.embedder.passages([c.embed_prefix + c.text for c in batch])
            self.collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
                embeddings=vectors,
            )

    def search(self, question, k=5, document_ids=None, max_distance=None):
        count = self.collection.count()
        if not count:
            return []
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        result = self.collection.query(
            query_embeddings=self.embedder.queries([question]),
            n_results=min(k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = [
            Hit(Chunk(cid, text, meta), float(distance))
            for cid, text, meta, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=False,
            )
        ]
        # Drop hits that are too far from the query in embedding space. These are
        # unlikely to contain supporting evidence; filtering them before
        # generation reduces false abstentions and saves context budget.
        if max_distance is not None and max_distance > 0:
            hits = [h for h in hits if h.distance < max_distance]
        return hits

    def all_chunks(self, document_ids=None) -> list[Chunk]:
        """Every indexed chunk, optionally filtered by document.

        Used to build the BM25 index over the same corpus as the vector store.
        """
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        result = self.collection.get(where=where, include=["documents", "metadatas"])
        return [
            Chunk(cid, text, meta)
            for cid, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"], strict=False
            )
        ]

    def delete(self, document_id):
        self.collection.delete(where={"document_id": document_id})
