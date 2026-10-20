"""Unit tests for grounded_qa.models — domain data structures."""

import pytest
from pydantic import ValidationError

from grounded_qa.models import (
    STATUSES,
    Chunk,
    Claim,
    Draft,
    Evidence,
    Hit,
    Query,
)

# ---------------------------------------------------------------------------
# Chunk and Hit (dataclasses)
# ---------------------------------------------------------------------------


def test_chunk_is_frozen():
    c = Chunk("id1", "some text", {"page": 1})
    with pytest.raises(AttributeError):
        c.id = "other"  # frozen dataclass raises FrozenInstanceError


def test_hit_stores_distance():
    c = Chunk("id1", "text", {})
    h = Hit(c, 0.42)
    assert h.chunk is c
    assert h.distance == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_evidence_requires_minimum_quote_length():
    with pytest.raises(ValidationError):
        Evidence(source_id="S1", quote="too short")  # < 12 chars


def test_evidence_valid():
    e = Evidence(source_id="S1", quote="This is a valid quote text.")
    assert e.source_id == "S1"


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------


def test_claim_requires_non_empty_text():
    with pytest.raises(ValidationError):
        Claim(
            text="",
            evidence=[Evidence(source_id="S1", quote="A valid long enough quote.")],
        )


def test_claim_requires_at_least_one_evidence():
    with pytest.raises(ValidationError):
        Claim(text="Some claim.", evidence=[])


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------


def test_draft_abstain_with_no_claims():
    d = Draft()
    assert d.abstain is True
    assert d.claims == []


def test_draft_abstain_false_when_claims_present():
    d = Draft(
        claims=[
            Claim(
                text="A factual claim.",
                evidence=[
                    Evidence(source_id="S1", quote="A sufficiently long quote here.")
                ],
            )
        ]
    )
    assert d.abstain is False


def test_draft_max_two_claims():
    e = Evidence(source_id="S1", quote="A sufficiently long quote here.")
    claim = Claim(text="Claim.", evidence=[e])
    with pytest.raises(ValidationError):
        Draft(claims=[claim, claim, claim])  # max_length=2


def test_draft_allows_two_claims():
    e = Evidence(source_id="S1", quote="A sufficiently long quote here.")
    c1 = Claim(text="First claim.", evidence=[e])
    c2 = Claim(text="Second claim.", evidence=[e])
    d = Draft(claims=[c1, c2])
    assert len(d.claims) == 2
    assert d.abstain is False


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def test_query_trims_whitespace():
    q = Query(question="  What is AI risk?  ")
    assert q.question == "What is AI risk?"


def test_query_rejects_too_short():
    with pytest.raises(ValidationError):
        Query(question="Hi")  # min_length=3, but "Hi" after trim = 2 chars


def test_query_rejects_too_long():
    with pytest.raises(ValidationError):
        Query(question="x" * 2001)


def test_query_accepts_boundary_lengths():
    Query(question="abc")  # exactly 3 chars
    Query(question="x" * 2000)  # exactly 2000 chars


def test_query_default_document_ids():
    q = Query(question="What is risk management?")
    assert q.document_ids == []


def test_query_document_ids_max_twenty():
    with pytest.raises(ValidationError):
        Query(question="What is AI?", document_ids=["id"] * 21)


def test_query_document_ids_up_to_twenty():
    q = Query(question="What is AI?", document_ids=["id"] * 20)
    assert len(q.document_ids) == 20


def test_query_trims_whitespace_only_string():
    """A question that is all whitespace becomes empty and fails min_length."""
    with pytest.raises(ValidationError):
        Query(question="   ")


# ---------------------------------------------------------------------------
# STATUSES constant
# ---------------------------------------------------------------------------


def test_statuses_contains_expected_values():
    assert set(STATUSES) == {"answered", "abstained", "rejected", "error"}
