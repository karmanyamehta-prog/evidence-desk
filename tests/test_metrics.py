import pytest
from grounded_qa.evaluation.metrics import (
    precision_at_k,
    faithfulness,
    evidence_coverage,
)


def test_precision_denominator_and_duplicates():
    assert precision_at_k({"a"}, ["a", "a", "b"], 5) == 0.2


def test_empty_faithfulness_is_not_perfect():
    assert faithfulness([]) is None
    assert faithfulness(["non_factual"]) is None
    assert faithfulness(["supported", "non_factual", "unsupported"]) == 0.5
    assert faithfulness(["supported", "contradicted"]) == 0.5
    with pytest.raises(ValueError):
        faithfulness(["looks fine"])


def test_page_only_match_is_insufficient():
    g = [{"document_id": "a", "page": 1, "start": 10, "end": 20}]
    assert (
        evidence_coverage(
            g, [{"metadata": {"document_id": "a", "page": 1, "start": 0, "end": 9}}]
        )
        == 0
    )


def test_rank_metrics_and_quote_anchored_coverage():
    from grounded_qa.evaluation.metrics import (
        evidence_coverage_by_quote,
        hit_at_k,
        reciprocal_rank,
    )

    assert hit_at_k({"b"}, ["a", "b", "c"], 1) == 0.0
    assert hit_at_k({"b"}, ["a", "b", "c"], 2) == 1.0
    assert reciprocal_rank({"b"}, ["a", "b"]) == 0.5
    assert reciprocal_rank({"z"}, ["a", "b"]) == 0.0
    gold = [{"quote": "voluntary, rights-preserving"}]
    assert (
        evidence_coverage_by_quote(
            gold, [{"text": "It is\nvoluntary, rights-preserving."}]
        )
        == 1.0
    )
    assert evidence_coverage_by_quote(gold, [{"text": "mandatory"}]) == 0.0
