"""Unit tests for grounded_qa.retrieval.bm25 — keyword search and RRF fusion."""

import pytest

from grounded_qa.models import Chunk, Hit
from grounded_qa.retrieval.bm25 import BM25Index, hybrid_search, rrf_fuse


def make_chunk(cid, text):
    return Chunk(cid, text, {"page": 1})


@pytest.fixture
def chunks():
    return [
        make_chunk("a", "GOVERN 1.6 requires organizations to inventory AI systems"),
        make_chunk("b", "MEASURE 2.5 requires the AI system to be demonstrated valid"),
        make_chunk("c", "MAP 5.2 covers engagement of practitioners and stakeholders"),
        make_chunk("d", "MANAGE 4.1 covers post deployment monitoring activities"),
    ]


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------


def test_bm25_finds_exact_keyword_match(chunks):
    index = BM25Index(chunks)
    hits = index.search("GOVERN 1.6 inventory")
    assert hits
    assert hits[0].chunk.id == "a"


def test_bm25_finds_action_id(chunks):
    """The action ID is the distinctive token; BM25 should surface its chunk."""
    index = BM25Index(chunks)
    hits = index.search("MANAGE 4.1")
    assert hits
    assert hits[0].chunk.id == "d"


def test_bm25_returns_nothing_for_unrelated_query(chunks):
    index = BM25Index(chunks)
    hits = index.search("quantum chromodynamics supersymmetry")
    assert hits == []


def test_bm25_respects_k_limit(chunks):
    index = BM25Index(chunks)
    hits = index.search("AI systems monitoring", k=2)
    assert len(hits) <= 2


def test_bm25_chunk_ids_property(chunks):
    index = BM25Index(chunks)
    assert index.chunk_ids == ["a", "b", "c", "d"]


def test_bm25_empty_corpus():
    index = BM25Index([])
    assert index.search("anything") == []


def test_bm25_is_case_insensitive(chunks):
    index = BM25Index(chunks)
    lower = index.search("govern inventory")
    upper = index.search("GOVERN INVENTORY")
    assert [h.chunk.id for h in lower] == [h.chunk.id for h in upper]


# ---------------------------------------------------------------------------
# rrf_fuse
# ---------------------------------------------------------------------------


def test_rrf_ranks_consensus_first():
    """A chunk ranked highly by both lists wins over one ranked by only one."""
    hits = {
        "x": Hit(make_chunk("x", "text"), 0.1),
        "y": Hit(make_chunk("y", "text"), 0.1),
    }
    fused = rrf_fuse([["y", "x"], ["x", "y"]], hits, top_n=2)
    # Both appear once in each list -> equal scores, order is stable by first sight
    assert {h.chunk.id for h in fused} == {"x", "y"}


def test_rrf_prefers_appearing_in_both_lists():
    hits = {
        "a": Hit(make_chunk("a", "text"), 0.1),
        "b": Hit(make_chunk("b", "text"), 0.1),
        "c": Hit(make_chunk("c", "text"), 0.1),
    }
    fused = rrf_fuse([["a", "b", "c"], ["c", "a"]], hits, top_n=3)
    # "a" is rank 1 + rank 2 = 1/61 + 1/62; "c" is rank 3 + rank 1 = 1/63 + 1/61
    # "b" only appears once. "a" has the highest combined score.
    assert fused[0].chunk.id == "a"


def test_rrf_respects_top_n():
    hits = {str(i): Hit(make_chunk(str(i), "text"), 0.1) for i in range(10)}
    fused = rrf_fuse([["0", "1", "2", "3", "4"]], hits, top_n=3)
    assert len(fused) == 3


def test_rrf_skips_ids_missing_from_hits():
    hits = {"a": Hit(make_chunk("a", "text"), 0.1)}
    fused = rrf_fuse([["a", "ghost"]], hits, top_n=5)
    assert [h.chunk.id for h in fused] == ["a"]


def test_rrf_empty_input():
    assert rrf_fuse([], {}, top_n=5) == []


# ---------------------------------------------------------------------------
# hybrid_search
# ---------------------------------------------------------------------------


def test_hybrid_search_finds_keyword_match_cosine_missed(chunks):
    """The point of hybrid search: a keyword-only match still surfaces."""
    # Cosine returns only chunk "b" (semantic match, no keyword overlap)
    cosine_hits = [Hit(chunks[1], 0.2)]
    index = BM25Index(chunks)
    fused = hybrid_search("GOVERN 1.6 inventory", cosine_hits, index, top_k=5)
    ids = [h.chunk.id for h in fused]
    assert "b" in ids
    # "a" is the keyword match that cosine missed
    assert "a" in ids


def test_hybrid_search_keeps_cosine_hit(chunks):
    cosine_hits = [Hit(chunks[0], 0.1)]
    index = BM25Index(chunks)
    fused = hybrid_search("GOVERN 1.6 inventory", cosine_hits, index, top_k=5)
    assert fused[0].chunk.id == "a"


def test_hybrid_search_respects_top_k(chunks):
    cosine_hits = [Hit(c, 0.1) for c in chunks]
    index = BM25Index(chunks)
    fused = hybrid_search("AI systems", cosine_hits, index, top_k=2)
    assert len(fused) <= 2


def test_hybrid_search_preserves_real_cosine_distance(chunks):
    """A chunk in both rankings must keep its true cosine distance."""
    cosine_hits = [Hit(chunks[0], 0.05)]
    index = BM25Index(chunks)
    fused = hybrid_search("GOVERN 1.6 inventory", cosine_hits, index, top_k=5)
    gov = next(h for h in fused if h.chunk.id == "a")
    assert gov.distance == pytest.approx(0.05)


def test_hybrid_search_with_no_cosine_hits(chunks):
    """Pure keyword search still works when cosine returns nothing."""
    index = BM25Index(chunks)
    fused = hybrid_search("MEASURE 2.5 demonstrated valid", [], index, top_k=5)
    assert fused
    assert fused[0].chunk.id == "b"
