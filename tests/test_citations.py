import pytest

from grounded_qa.citations import sources_from_hits, validate
from grounded_qa.models import Chunk, Claim, Draft, Evidence, Hit

S = {"S1": {"text": "Human oversight is important for managing risk."}}


def draft(sid="S1", quote="Human oversight is important for managing"):
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


def test_typographic_variants_still_match():
    src = {
        "S1": {
            "text": "The \u201cCore\u201d covers human-\nAI interaction \u2013 and more."
        }
    }
    validate(draft(quote='The "Core" covers human-AI interaction - and more'), src)


def test_quote_length_bounds():
    for quote in ("Human oversight is", "Human " * 71):
        with pytest.raises(ValueError, match="words"):
            validate(draft(quote=quote), S)


def test_claim_identifier_must_appear_in_quote():
    src = {
        "S1": {
            "text": "MEASURE 2.5: The AI system to be deployed is demonstrated valid."
        }
    }
    quote = "MEASURE 2.5: The AI system to be deployed"

    def claim(text):
        return Draft(
            claims=[Claim(text=text, evidence=[Evidence(source_id="S1", quote=quote)])]
        )

    validate(claim("MEASURE 2.5 requires demonstrated validity."), src)
    with pytest.raises(ValueError, match="MANAGE 2.5"):
        validate(claim("MANAGE 2.5 requires demonstrated validity."), src)
