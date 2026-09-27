from grounded_qa.retrieval.store import VectorStore
from grounded_qa.models import Chunk


class Vectors:
    def passages(self, texts):
        return [[1.0, 0.0] if "privacy" in x else [0.0, 1.0] for x in texts]

    def queries(self, texts):
        return self.passages(texts)


def test_rank_filter_persistence_and_upsert(tmp_path):
    store = VectorStore(tmp_path, "test", Vectors())
    assert store.search("privacy") == []
    chunks = [
        Chunk("a", "privacy controls", {"document_id": "doc-a", "page": 1}),
        Chunk("b", "safety controls", {"document_id": "doc-b", "page": 2}),
    ]
    store.add(chunks)
    store.add(chunks)
    assert store.collection.count() == 2
    assert store.search("privacy", 1)[0].chunk.id == "a"
    assert store.search("privacy", 5, ["doc-b"])[0].chunk.id == "b"
    assert store.search("privacy", 5, ["missing"]) == []
    reopened = VectorStore(tmp_path, "test", Vectors())
    assert reopened.search("safety", 1)[0].chunk.id == "b"
    store.delete("doc-a")
    assert store.collection.count() == 1


def test_namespace_isolates_embedding_spaces(tmp_path):
    a = VectorStore(tmp_path, "model-a", Vectors())
    b = VectorStore(tmp_path, "model-b", Vectors())
    a.add([Chunk("a", "privacy", {"document_id": "one"})])
    assert b.search("privacy") == []
