"""Additional unit tests for grounded_qa.evaluation.metrics — uncovered functions."""

import pytest

from grounded_qa.evaluation.metrics import (
    evidence_coverage,
    evidence_coverage_by_quote,
    faithfulness,
    final_status,
    hit_at_k,
    precision_at_k,
    reciprocal_rank,
)

# ---------------------------------------------------------------------------
# final_status — backward compatibility shim
# ---------------------------------------------------------------------------


def test_final_status_returns_existing_status_field():
    assert final_status({"status": "answered", "abstain": False}) == "answered"
    assert final_status({"status": "rejected", "abstain": True}) == "rejected"
    assert final_status({"status": "error", "abstain": True}) == "error"


def test_final_status_no_status_answered():
    result = {"abstain": False, "reason": ""}
    assert final_status(result) == "answered"


def test_final_status_no_status_abstained():
    result = {"abstain": True, "reason": "Insufficient evidence"}
    assert final_status(result) == "abstained"


def test_final_status_no_status_rejected():
    result = {
        "abstain": True,
        "reason": "Citation quote does not occur in retrieved text",
    }
    assert final_status(result) == "rejected"


# ---------------------------------------------------------------------------
# evidence_coverage — additional cases
# ---------------------------------------------------------------------------


def test_evidence_coverage_none_on_empty_gold():
    assert evidence_coverage([], []) is None
    assert (
        evidence_coverage(
            [], [{"metadata": {"document_id": "a", "page": 1, "start": 0, "end": 10}}]
        )
        is None
    )


def test_evidence_coverage_full_match():
    gold = [{"document_id": "a", "page": 1, "start": 5, "end": 15}]
    retrieved = [{"metadata": {"document_id": "a", "page": 1, "start": 0, "end": 20}}]
    assert evidence_coverage(gold, retrieved) == 1.0


def test_evidence_coverage_partial():
    gold = [
        {"document_id": "a", "page": 1, "start": 5, "end": 15},
        {"document_id": "a", "page": 2, "start": 0, "end": 10},
    ]
    # Only first span is covered
    retrieved = [{"metadata": {"document_id": "a", "page": 1, "start": 0, "end": 20}}]
    assert evidence_coverage(gold, retrieved) == pytest.approx(0.5)


def test_evidence_coverage_wrong_document():
    gold = [{"document_id": "doc-a", "page": 1, "start": 0, "end": 10}]
    retrieved = [
        {"metadata": {"document_id": "doc-b", "page": 1, "start": 0, "end": 10}}
    ]
    assert evidence_coverage(gold, retrieved) == 0.0


def test_evidence_coverage_span_boundary_exact():
    """start must be <= gold start and end must be >= gold end."""
    gold = [{"document_id": "a", "page": 1, "start": 10, "end": 20}]
    # Chunk that ends exactly at gold end but starts after gold start — miss
    too_late_start = [
        {"metadata": {"document_id": "a", "page": 1, "start": 11, "end": 20}}
    ]
    assert evidence_coverage(gold, too_late_start) == 0.0
    # Chunk that starts exactly at gold start but ends before gold end — miss
    too_early_end = [
        {"metadata": {"document_id": "a", "page": 1, "start": 10, "end": 19}}
    ]
    assert evidence_coverage(gold, too_early_end) == 0.0
    # Exact match — hit
    exact = [{"metadata": {"document_id": "a", "page": 1, "start": 10, "end": 20}}]
    assert evidence_coverage(gold, exact) == 1.0


# ---------------------------------------------------------------------------
# evidence_coverage_by_quote — additional cases
# ---------------------------------------------------------------------------


def test_evidence_coverage_by_quote_none_on_empty_gold():
    assert evidence_coverage_by_quote([], []) is None


def test_evidence_coverage_by_quote_case_insensitive():
    gold = [{"quote": "HUMAN OVERSIGHT IS IMPORTANT"}]
    retrieved = [{"text": "human oversight is important for AI systems"}]
    assert evidence_coverage_by_quote(gold, retrieved) == 1.0


def test_evidence_coverage_by_quote_typographic_variants():
    gold = [{"quote": "the “Core” framework"}]
    retrieved = [{"text": 'the "Core" framework covers many areas'}]
    assert evidence_coverage_by_quote(gold, retrieved) == 1.0


def test_evidence_coverage_by_quote_not_found():
    gold = [{"quote": "this phrase does not appear anywhere"}]
    retrieved = [{"text": "completely different content about other things"}]
    assert evidence_coverage_by_quote(gold, retrieved) == 0.0


# ---------------------------------------------------------------------------
# precision_at_k — additional cases
# ---------------------------------------------------------------------------


def test_precision_at_k_all_relevant():
    assert precision_at_k({"a", "b", "c"}, ["a", "b", "c"], 3) == pytest.approx(1.0)


def test_precision_at_k_none_relevant():
    assert precision_at_k({"z"}, ["a", "b", "c"], 3) == pytest.approx(0.0)


def test_precision_at_k_k_larger_than_retrieved():
    # k=10 but only 2 items retrieved; denominator is still k
    assert precision_at_k({"a"}, ["a", "b"], 10) == pytest.approx(0.1)


def test_precision_at_k_truncates_at_k():
    # "c" is relevant but rank 3; k=2 so it should not count
    assert precision_at_k({"c"}, ["a", "b", "c"], 2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# hit_at_k — additional cases
# ---------------------------------------------------------------------------


def test_hit_at_k_empty_retrieved():
    assert hit_at_k({"a"}, [], 5) == 0.0


def test_hit_at_k_at_exactly_k():
    # relevant item is at position k (1-indexed)
    assert hit_at_k({"c"}, ["a", "b", "c"], 3) == 1.0


def test_hit_at_k_deduplicates():
    # ["a", "a", "b"] deduplicates to ["a", "b"] — "b" is at unique position 2
    # so k=1 misses it but k=2 hits it
    assert hit_at_k({"b"}, ["a", "a", "b"], 1) == 0.0
    assert hit_at_k({"b"}, ["a", "a", "b"], 2) == 1.0


# ---------------------------------------------------------------------------
# reciprocal_rank — additional cases
# ---------------------------------------------------------------------------


def test_reciprocal_rank_first_position():
    assert reciprocal_rank({"a"}, ["a", "b", "c"]) == pytest.approx(1.0)


def test_reciprocal_rank_third_position():
    assert reciprocal_rank({"c"}, ["a", "b", "c"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_not_found():
    assert reciprocal_rank({"z"}, ["a", "b", "c"]) == 0.0


def test_reciprocal_rank_deduplicates():
    # "a" at position 1 twice; first unique occurrence is rank 1
    assert reciprocal_rank({"b"}, ["b", "b", "c"]) == pytest.approx(1.0)


def test_reciprocal_rank_empty():
    assert reciprocal_rank({"a"}, []) == 0.0


# ---------------------------------------------------------------------------
# faithfulness — additional edge cases
# ---------------------------------------------------------------------------


def test_faithfulness_all_supported():
    assert faithfulness(["supported", "supported"]) == pytest.approx(1.0)


def test_faithfulness_all_contradicted():
    assert faithfulness(["contradicted", "contradicted"]) == pytest.approx(0.0)


def test_faithfulness_all_non_factual_is_none():
    assert faithfulness(["non_factual", "non_factual"]) is None


def test_faithfulness_mixed_with_non_factual_excludes_them():
    # 2 supported, 1 unsupported, 1 non_factual → factual count=3, supported=2
    result = faithfulness(["supported", "supported", "unsupported", "non_factual"])
    assert result == pytest.approx(2 / 3)
