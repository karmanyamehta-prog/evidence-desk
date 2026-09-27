import pytest
from grounded_qa.models import Draft, Claim, Evidence, Chunk, Hit
from grounded_qa.citations import validate, sources_from_hits

S = {"S1": {"text": "Human oversight is important for managing risk."}}


def draft(sid="S1", quote="Human oversight is important"):
    return Draft(
        claims=[
            Claim(
                text="Oversight matters.",
                evidence=[Evidence(source_id=sid, quote=quote)],
            )
        ]
    )


def test_valid_quote():
    validate(draft(), S)


@pytest.mark.parametrize(
    "d", [draft("S9"), draft(quote="This sentence was fabricated.")]
)
def test_rejects_invalid_provenance(d):
    with pytest.raises(ValueError):
        validate(d, S)


def test_empty_claims_are_abstention():
    empty = Draft()
    validate(empty, S)
    assert empty.abstain
    assert not draft().abstain


def test_context_budget_keeps_whole_chunks():
    hits = [Hit(Chunk("1", "x" * 20, {}), 0), Hit(Chunk("2", "y" * 20, {}), 0)]
    s = sources_from_hits(hits, 25)
    assert list(s) == ["S1"] and s["S1"]["text"] == "x" * 20
