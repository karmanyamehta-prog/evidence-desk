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
