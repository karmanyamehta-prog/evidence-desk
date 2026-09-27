"""Persistent cosine search; API owns the only writer."""
import hashlib
import json
from grounded_qa.models import Chunk, Hit

class VectorStore:
    def __init__(self, path, fingerprint, embedder):
        import chromadb
        from chromadb.config import Settings
        self.embedder = embedder
        self.client = chromadb.PersistentClient(str(path), settings=Settings(anonymized_telemetry=False))
        name = "evidence-" + hashlib.sha256(fingerprint.encode()).hexdigest()[:20]
        self.collection = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks):
        for start in range(0, len(chunks), 32):
            batch = chunks[start:start+32]
            vectors = self.embedder.passages([c.text for c in batch])
            self.collection.upsert(ids=[c.id for c in batch], documents=[c.text for c in batch],
                                   metadatas=[c.metadata for c in batch], embeddings=vectors)

    def search(self, question, k=5, document_ids=None):
        count = self.collection.count()
        if not count:
            return []
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        result = self.collection.query(query_embeddings=self.embedder.queries([question]),
                                       n_results=min(k, count), where=where,
                                       include=["documents", "metadatas", "distances"])
        return [Hit(Chunk(cid, text, meta), float(distance)) for cid, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])]

    def delete(self, document_id):
        self.collection.delete(where={"document_id": document_id})
