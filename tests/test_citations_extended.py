"""Additional unit tests for grounded_qa.citations — fold(), sources_from_hits edge cases."""

import pytest

from grounded_qa.citations import fold, sources_from_hits, validate
from grounded_qa.models import Chunk, Claim, Draft, Evidence, Hit

# ---------------------------------------------------------------------------
# fold()
# ---------------------------------------------------------------------------


def test_fold_lowercases():
    assert fold("HELLO WORLD") == "hello world"


def test_fold_curly_single_quotes():
    assert fold("‘hello’") == "'hello'"


def test_fold_curly_double_quotes():
    assert fold("“hello”") == '"hello"'


def test_fold_en_dash():
    assert fold("one–two") == "one-two"


def test_fold_em_dash():
    assert fold("one—two") == "one-two"


def test_fold_minus_sign():
    assert fold("one−two") == "one-two"


def test_fold_line_break_hyphen():
    # "infor-\nmation" → "infor-mation" (not "infor- mation")
    result = fold("infor-\nmation")
    assert "infor-mation" in result


def test_fold_collapses_whitespace():
    assert fold("hello   world") == "hello world"


def test_fold_nfkc_normalization():
    # NFKC turns fullwidth chars into ASCII equivalents
    result = fold("ＨＩ")  # fullwidth HI
    assert result == "hi"


# ---------------------------------------------------------------------------
# sources_from_hits — budget edge cases
# ---------------------------------------------------------------------------


def make_hit(cid, text, distance=0.1):
    return Hit(Chunk(cid, text, {"page": 1}), distance)


def test_sources_from_hits_empty():
    assert sources_from_hits([]) == {}


def test_sources_from_hits_exact_budget_fit():
    """A chunk whose length exactly equals max_chars should be included."""
    text = "a" * 100
    sources = sources_from_hits([make_hit("1", text)], max_chars=100)
    assert "S1" in sources
    assert sources["S1"]["text"] == text


def test_sources_from_hits_one_over_budget_skipped():
    """A chunk that would push total over budget is skipped (whole-chunk policy)."""
    text = "a" * 101
    sources = sources_from_hits([make_hit("1", text)], max_chars=100)
    assert sources == {}


def test_sources_from_hits_sequential_numbering():
    hits = [make_hit("a", "text_a"), make_hit("b", "text_b")]
    sources = sources_from_hits(hits, max_chars=9000)
    assert list(sources.keys()) == ["S1", "S2"]


def test_sources_from_hits_preserves_metadata():
    chunk = Chunk("c1", "some text content", {"page": 3, "section": "Intro"})
    hit = Hit(chunk, 0.05)
    sources = sources_from_hits([hit])
    assert sources["S1"]["page"] == 3
    assert sources["S1"]["section"] == "Intro"
    assert sources["S1"]["chunk_id"] == "c1"
    assert sources["S1"]["distance"] == pytest.approx(0.05)


def test_sources_from_hits_skips_oversized_keeps_rest():
    """A large chunk in the middle is skipped; later smaller ones are still included."""
    big = make_hit("big", "b" * 200)
    small1 = make_hit("s1", "first small chunk text")
    small2 = make_hit("s2", "second small chunk text")
    sources = sources_from_hits([small1, big, small2], max_chars=100)
    texts = [v["text"] for v in sources.values()]
    assert "first small chunk text" in texts
    assert "second small chunk text" in texts
    assert "b" * 200 not in texts


# ---------------------------------------------------------------------------
# validate — additional edge cases
# ---------------------------------------------------------------------------


def test_validate_min_quote_word_boundary():
    """Exactly MIN_QUOTE_WORDS (5) should pass; 4 should fail."""
    src = {"S1": {"text": "one two three four five six seven eight nine ten"}}
    ok_quote = "one two three four five"
    too_short = "one two three four"

    ok_draft = Draft(
        claims=[
            Claim(text="Claim.", evidence=[Evidence(source_id="S1", quote=ok_quote)])
        ]
    )
    validate(ok_draft, src)

    bad_draft = Draft(
        claims=[
            Claim(text="Claim.", evidence=[Evidence(source_id="S1", quote=too_short)])
        ]
    )
    with pytest.raises(ValueError, match="words"):
        validate(bad_draft, src)


def test_validate_max_quote_word_boundary():
    """Exactly MAX_QUOTE_WORDS (70) should pass; 71 words should fail."""
    words_70 = " ".join(f"word{i}" for i in range(70))
    words_71 = " ".join(f"word{i}" for i in range(71))
    # Source must contain the quote
    src = {"S1": {"text": words_71}}

    ok_draft = Draft(
        claims=[
            Claim(text="Claim.", evidence=[Evidence(source_id="S1", quote=words_70)])
        ]
    )
    validate(ok_draft, src)

    bad_draft = Draft(
        claims=[
            Claim(text="Claim.", evidence=[Evidence(source_id="S1", quote=words_71)])
        ]
    )
    with pytest.raises(ValueError, match="words"):
        validate(bad_draft, src)


def test_validate_error_message_bad_source():
    src = {"S1": {"text": "Some valid source text content."}}
    bad = Draft(
        claims=[
            Claim(
                text="Claim.",
                evidence=[
                    Evidence(source_id="S99", quote="Some valid source text content")
                ],
            )
        ]
    )
    with pytest.raises(ValueError, match="unretrieved"):
        validate(bad, src)


def test_validate_multiple_claims_all_checked():
    """Both claims must be valid; a bad second claim still raises."""
    src = {
        "S1": {"text": "Human oversight is important for managing risk effectively."},
        "S2": {"text": "AI governance frameworks are essential for safety."},
    }
    good_claim = Claim(
        text="Oversight matters.",
        evidence=[
            Evidence(source_id="S1", quote="Human oversight is important for managing")
        ],
    )
    bad_claim = Claim(
        text="Something.",
        evidence=[
            Evidence(source_id="S2", quote="This quote was completely fabricated here")
        ],
    )
    with pytest.raises(ValueError):
        validate(Draft(claims=[good_claim, bad_claim]), src)


def test_validate_identifier_with_numeric_only():
    """Numeric identifiers like '3.4' should also be checked against the quote."""
    src = {
        "S1": {"text": "3.4 The system must log all inference requests and responses."}
    }
    quote = "3.4 The system must log all inference requests"
    good = Draft(
        claims=[
            Claim(
                text="Section 3.4 requires logging.",
                evidence=[Evidence(source_id="S1", quote=quote)],
            )
        ]
    )
    validate(good, src)

    # A different numeric identifier that isn't in the quote
    bad = Draft(
        claims=[
            Claim(
                text="Section 4.1 requires logging.",
                evidence=[Evidence(source_id="S1", quote=quote)],
            )
        ]
    )
    with pytest.raises(ValueError, match="4.1"):
        validate(bad, src)
