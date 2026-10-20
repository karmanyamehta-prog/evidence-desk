"""Unit tests for Phase 2: per-claim filtering, action-ID embedding, thresholds."""

import pytest

from grounded_qa.citations import filter_claims, validate
from grounded_qa.config import Settings
from grounded_qa.ingestion.chunking import chunk_pages
from grounded_qa.models import Chunk, Claim, Draft, Evidence, Hit

# ---------------------------------------------------------------------------
# Per-claim filtering
# ---------------------------------------------------------------------------

SOURCES = {
    "S1": {"text": "MEASURE 2.5: The AI system to be deployed is demonstrated valid."},
    "S2": {"text": "Human oversight is important for managing AI risk effectively."},
}


def _claim(text, quote, source_id="S1"):
    return Claim(text=text, evidence=[Evidence(source_id=source_id, quote=quote)])


def test_filter_claims_keeps_all_valid():
    draft = Draft(
        claims=[
            _claim("MEASURE 2.5 holds.", "MEASURE 2.5: The AI system to be deployed"),
            _claim(
                "Oversight matters.",
                "Human oversight is important for managing",
                source_id="S2",
            ),
        ]
    )
    valid, errors = filter_claims(draft, SOURCES)
    assert len(valid) == 2
    assert errors == []


def test_filter_claims_keeps_valid_and_reports_invalid():
    """One bad claim must not remove the good one."""
    draft = Draft(
        claims=[
            _claim("MEASURE 2.5 holds.", "MEASURE 2.5: The AI system to be deployed"),
            _claim("Fabricated.", "A quote that is nowhere in the source text"),
        ]
    )
    valid, errors = filter_claims(draft, SOURCES)
    assert len(valid) == 1
    assert valid[0].text == "MEASURE 2.5 holds."
    assert len(errors) == 1
    assert "does not occur" in errors[0]


def test_filter_claims_all_invalid_returns_empty():
    draft = Draft(
        claims=[
            _claim("Fabricated one.", "A quote that is nowhere in the source text"),
            _claim("Fabricated two.", "Another quote absent from the source text"),
        ]
    )
    valid, errors = filter_claims(draft, SOURCES)
    assert valid == []
    assert len(errors) == 2


def test_filter_claims_empty_draft():
    valid, errors = filter_claims(Draft(), SOURCES)
    assert valid == [] and errors == []


def test_filter_claims_bad_source_id():
    draft = Draft(claims=[_claim("Bad.", "MEASURE 2.5: The AI system", source_id="S9")])
    valid, errors = filter_claims(draft, SOURCES)
    assert valid == []
    assert "unretrieved" in errors[0]


def test_filter_claims_quote_too_short():
    # Evidence requires >= 12 characters, but the word-count rule needs 5 words,
    # so a 3-word quote is long enough to construct yet too short to validate.
    draft = Draft(claims=[_claim("Bad.", "one two three")])
    valid, errors = filter_claims(draft, SOURCES)
    assert valid == []
    assert "words" in errors[0]


def test_filter_claims_identifier_mismatch():
    draft = Draft(claims=[_claim("MANAGE 2.5 holds.", "MEASURE 2.5: The AI system")])
    valid, errors = filter_claims(draft, SOURCES)
    assert valid == []
    assert "MANAGE 2.5" in errors[0]


def test_validate_still_raises_on_first_error():
    """The legacy all-or-nothing validator is retained for the retry path."""
    draft = Draft(claims=[_claim("Bad.", "A quote that is nowhere in the source text")])
    with pytest.raises(ValueError):
        validate(draft, SOURCES)


def test_filter_and_validate_agree_on_valid_claims():
    draft = Draft(
        claims=[
            _claim("MEASURE 2.5 holds.", "MEASURE 2.5: The AI system to be deployed")
        ]
    )
    valid, errors = filter_claims(draft, SOURCES)
    assert len(valid) == 1 and not errors
    validate(draft, SOURCES)  # must not raise


# ---------------------------------------------------------------------------
# Action-ID embedding prefix
# ---------------------------------------------------------------------------


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_action_id_is_prepended_to_embed_prefix():
    pages = [
        {
            "page": 1,
            "text": "GOVERN 1.6: The organization should inventory its AI systems.",
        }
    ]
    chunks = chunk_pages(pages, "doc", "Title", Tokenizer(), 32, 4)
    assert chunks[0].embed_prefix == "GOVERN 1.6: "
    # The stored chunk text stays the verbatim source passage
    assert (
        chunks[0].text
        == "GOVERN 1.6: The organization should inventory its AI systems."
    )


def test_embed_prefix_is_empty_without_a_section():
    pages = [{"page": 1, "text": "Plain prose with no action identifier at all."}]
    chunks = chunk_pages(pages, "doc", "Title", Tokenizer(), 32, 4)
    assert chunks[0].embed_prefix == ""


def test_neighbor_chunk_ids_are_recorded():
    pages = [
        {
            "page": 1,
            "text": "First sentence here for chunking purposes. Second sentence follows.",
        }
    ]
    chunks = chunk_pages(pages, "doc", "Title", Tokenizer(), 8, 2)
    assert len(chunks) > 1
    assert chunks[0].metadata["prev_chunk_id"] == ""
    assert chunks[0].metadata["next_chunk_id"] == chunks[1].id
    assert chunks[-1].metadata["next_chunk_id"] == ""
    assert chunks[1].metadata["prev_chunk_id"] == chunks[0].id


def test_neighbor_ids_are_consistent_both_ways():
    pages = [{"page": 1, "text": " ".join(f"word{i}" for i in range(60))}]
    chunks = chunk_pages(pages, "doc", "Title", Tokenizer(), 16, 4)
    for i, chunk in enumerate(chunks):
        expected_prev = chunks[i - 1].id if i > 0 else ""
        expected_next = chunks[i + 1].id if i < len(chunks) - 1 else ""
        assert chunk.metadata["prev_chunk_id"] == expected_prev
        assert chunk.metadata["next_chunk_id"] == expected_next


def test_chunk_ids_remain_unique_with_prefixes():
    pages = [
        {
            "page": 1,
            "text": "GOVERN 1.1: First action text. GOVERN 1.2: Second action text here.",
        }
    ]
    chunks = chunk_pages(pages, "doc", "Title", Tokenizer(), 16, 2)
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunk_ids_change_when_pipeline_version_changes():
    """A pipeline bump must invalidate old chunk IDs so vectors are not mixed."""
    import grounded_qa.ingestion.chunking as chunking_module

    pages = [{"page": 1, "text": "GOVERN 1.6: Inventory all AI systems in use."}]
    original = chunking_module.PIPELINE_VERSION
    try:
        first = chunk_pages(pages, "doc", "Title", Tokenizer(), 32, 4)
        chunking_module.PIPELINE_VERSION = "test-bumped-version"
        second = chunk_pages(pages, "doc", "Title", Tokenizer(), 32, 4)
        assert first[0].id != second[0].id
    finally:
        chunking_module.PIPELINE_VERSION = original


# ---------------------------------------------------------------------------
# Distance threshold setting
# ---------------------------------------------------------------------------


def test_threshold_defaults_to_disabled(tmp_path):
    s = Settings(data_dir=tmp_path)
    assert s.min_retrieval_distance == 0.0


def test_threshold_accepts_valid_values(tmp_path):
    s = Settings(data_dir=tmp_path, min_retrieval_distance=0.16)
    assert s.min_retrieval_distance == pytest.approx(0.16)


def test_threshold_rejects_out_of_range(tmp_path):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, min_retrieval_distance=1.5)
    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, min_retrieval_distance=-0.1)


def test_hybrid_search_flag_defaults_true(tmp_path):
    s = Settings(data_dir=tmp_path)
    assert s.hybrid_search is True


def test_hybrid_search_can_be_disabled(tmp_path):
    s = Settings(data_dir=tmp_path, hybrid_search=False)
    assert s.hybrid_search is False


# ---------------------------------------------------------------------------
# Service-level: threshold filtering and partial claims
# ---------------------------------------------------------------------------


class Store:
    def __init__(self, hits):
        self.hits = hits
        self.search_kwargs = None

    def search(self, question, k=5, document_ids=None, max_distance=None):
        self.search_kwargs = {
            "k": k,
            "document_ids": document_ids,
            "max_distance": max_distance,
        }
        return self.hits

    def all_chunks(self, document_ids=None):
        return [h.chunk for h in self.hits]

    def distances(self, question, chunk_ids):
        return dict.fromkeys(chunk_ids, 0.5)


class Generator:
    def __init__(self, drafts):
        self.drafts = iter(drafts)
        self.calls = 0

    def generate(self, *args):
        self.calls += 1
        return next(self.drafts)


def test_threshold_drops_distant_hits(tmp_path):
    """A hit beyond the distance threshold is dropped before generation."""
    from grounded_qa.service import Service

    hits = [
        Hit(Chunk("near", "close text", {"page": 1}), 0.10),
        Hit(Chunk("far", "distant text", {"page": 1}), 0.40),
    ]
    store = Store(hits)
    service = Service(
        Settings(data_dir=tmp_path, min_retrieval_distance=0.16),
        object(),
        store,
        Generator([Draft()]),  # empty draft: model abstains, no retry
    )
    result = service.query("a question")
    retrieved = [r["id"] for r in result["retrieved"]]
    assert "near" in retrieved
    assert "far" not in retrieved


def test_threshold_disabled_keeps_every_hit(tmp_path):
    from grounded_qa.service import Service

    hits = [Hit(Chunk("far", "distant text", {"page": 1}), 0.90)]
    service = Service(
        Settings(data_dir=tmp_path), object(), Store(hits), Generator([Draft()])
    )
    result = service.query("a question")
    assert [r["id"] for r in result["retrieved"]] == ["far"]


def test_threshold_applies_to_keyword_only_hits(tmp_path):
    """A keyword-only hit gets a real distance and is thresholded like any other."""
    from grounded_qa.service import Service

    hits = [Hit(Chunk("near", "GOVERN 1.6 inventory", {"page": 1}), 0.10)]
    store = Store(hits)
    store.distances = lambda question, chunk_ids: dict.fromkeys(chunk_ids, 0.8)
    service = Service(
        Settings(data_dir=tmp_path, min_retrieval_distance=0.16),
        object(),
        store,
        Generator([Draft()]),
    )
    result = service.query("a question")
    # The only hit is within the threshold, so nothing is dropped.
    assert [r["id"] for r in result["retrieved"]] == ["near"]


def test_keyword_only_hit_beyond_the_threshold_is_dropped(tmp_path):
    """A keyword-only hit that is too far away must not survive the threshold.

    The distance is computed from the stored vector, so a hit that only BM25
    found is filtered on the same footing as a cosine hit.
    """
    from grounded_qa.service import Service

    hits = [Hit(Chunk("near", "GOVERN 1.6 inventory", {"page": 1}), 0.10)]
    store = Store(hits)
    store.distances = lambda question, chunk_ids: dict.fromkeys(chunk_ids, 0.80)
    service = Service(
        Settings(data_dir=tmp_path, min_retrieval_distance=0.16),
        object(),
        store,
        Generator([Draft()]),
    )
    # The cosine hit is inside the threshold; a keyword-only hit at 0.80 is not.
    result = service.query("a question")
    assert all(r["distance"] < 0.16 for r in result["retrieved"])


def test_partial_claim_answer_is_answered_not_rejected(tmp_path):
    """A draft with one good and one bad claim must still answer."""
    from grounded_qa.service import Service

    draft = Draft(
        claims=[
            _claim("MEASURE 2.5 holds.", "MEASURE 2.5: The AI system to be deployed"),
            _claim("Fabricated.", "A quote that is nowhere in the source text"),
        ]
    )
    hits = [
        Hit(
            Chunk(
                "a",
                "MEASURE 2.5: The AI system to be deployed is demonstrated valid.",
                {"page": 1},
            ),
            0.1,
        )
    ]
    generator = Generator([draft])
    service = Service(Settings(data_dir=tmp_path), object(), Store(hits), generator)
    result = service.query("a question")
    assert result["status"] == "answered"
    assert len(result["claims"]) == 1
    assert result["claims"][0]["text"] == "MEASURE 2.5 holds."
    # No retry: a partially-valid answer is accepted
    assert generator.calls == 1
    # The dropped claim is still recorded in the attempt log
    assert result["attempts"][0]["error"] != ""


def test_model_initiated_abstention_does_not_retry(tmp_path):
    """An empty draft is a legitimate abstention, not a validation failure."""
    from grounded_qa.service import Service

    hits = [Hit(Chunk("a", "Some retrieved source text", {"page": 1}), 0.1)]
    generator = Generator([Draft(), Draft()])
    service = Service(Settings(data_dir=tmp_path), object(), Store(hits), generator)
    result = service.query("a question")
    assert result["status"] == "abstained"
    assert generator.calls == 1
    assert result["attempts"][0]["error"] == ""


def test_all_claims_invalid_is_rejected_and_retried(tmp_path):
    from grounded_qa.service import Service

    bad = Draft(claims=[_claim("Bad.", "A quote nowhere in the source text")])
    hits = [Hit(Chunk("a", "Some retrieved source text", {"page": 1}), 0.1)]
    generator = Generator([bad, bad])
    service = Service(Settings(data_dir=tmp_path), object(), Store(hits), generator)
    result = service.query("a question")
    assert result["status"] == "rejected"
    assert generator.calls == 2
    assert not result["claims"]
