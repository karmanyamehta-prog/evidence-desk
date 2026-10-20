import pytest

from grounded_qa.models import Chunk
from grounded_qa.retrieval.store import VectorStore


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


def test_all_chunks_returns_every_chunk(tmp_path):
    store = VectorStore(tmp_path, "test", Vectors())
    store.add(
        [
            Chunk("a", "privacy controls", {"document_id": "doc-a", "page": 1}),
            Chunk("b", "safety controls", {"document_id": "doc-b", "page": 2}),
        ]
    )
    assert {c.id for c in store.all_chunks()} == {"a", "b"}


def test_all_chunks_filters_by_document(tmp_path):
    store = VectorStore(tmp_path, "test", Vectors())
    store.add(
        [
            Chunk("a", "privacy controls", {"document_id": "doc-a", "page": 1}),
            Chunk("b", "safety controls", {"document_id": "doc-b", "page": 2}),
        ]
    )
    assert {c.id for c in store.all_chunks(["doc-b"])} == {"b"}


def test_distances_returns_the_true_cosine_distance(tmp_path):
    """A keyword-only hit must get a real distance, not a synthetic one.

    The stored vectors are unit vectors, so the distance to an identical
    vector is 0 and to an orthogonal vector is 1.
    """
    store = VectorStore(tmp_path, "test", Vectors())
    store.add(
        [
            Chunk("a", "privacy controls", {"document_id": "doc-a", "page": 1}),
            Chunk("b", "safety controls", {"document_id": "doc-b", "page": 2}),
        ]
    )
    distances = store.distances("privacy", ["a", "b"])
    assert distances["a"] == pytest.approx(0.0, abs=1e-6)
    assert distances["b"] == pytest.approx(1.0, abs=1e-6)


def test_distances_with_no_ids_is_empty(tmp_path):
    store = VectorStore(tmp_path, "test", Vectors())
    assert store.distances("privacy", []) == {}


def test_distances_for_an_unknown_id_is_empty(tmp_path):
    store = VectorStore(tmp_path, "test", Vectors())
    assert store.distances("privacy", ["does-not-exist"]) == {}


def test_search_applies_the_distance_threshold(tmp_path):
    store = VectorStore(tmp_path, "test", Vectors())
    store.add(
        [
            Chunk("a", "privacy controls", {"document_id": "doc-a", "page": 1}),
            Chunk("b", "safety controls", {"document_id": "doc-b", "page": 2}),
        ]
    )
    # "privacy" is distance 0 and "safety" is distance 1 from that query.
    hits = store.search("privacy", 5, None, max_distance=0.5)
    assert [h.chunk.id for h in hits] == ["a"]


def test_search_threshold_of_zero_disables_the_filter(tmp_path):
    store = VectorStore(tmp_path, "test", Vectors())
    store.add([Chunk("b", "safety controls", {"document_id": "doc-b", "page": 2})])
    hits = store.search("privacy", 5, None, max_distance=0.0)
    assert len(hits) == 1
