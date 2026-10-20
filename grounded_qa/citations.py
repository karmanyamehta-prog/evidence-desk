"""Validate provenance; these checks do NOT prove semantic entailment.

What is checked: source IDs exist, quotes occur in the cited chunk (after
folding typographic variants), quote length is sane, and identifiers named in a
claim (e.g. "MANAGE 2.5") also occur in the quote. A real quote can still be
irrelevant or misread, so this stays a provenance check.
"""

import re
import unicodedata
from typing import Any

from grounded_qa.ingestion.pdf import normalize

MIN_QUOTE_WORDS = 5
MAX_QUOTE_WORDS = 70
_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)
# Section/action identifiers such as "GOVERN 1.6", "A.1.1" or "3.4", used to
# catch a claim that cites a real quote about a different section.
_IDENTIFIER = re.compile(r"\b(?:[A-Z]{3,}\s?)?[A-Z]?\d+\.\d+(?:\.\d+)?\b")


def fold(text: str) -> str:  # comparison-only; never stored
    """Comparison-only normalisation; never used for stored text or offsets."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "-", text)  # line-break hyphen
    text = re.sub(r"\s*-\s*", "-", normalize(text))
    return text.casefold()


def _check_claim(claim: Any, sources: dict[str, Any]) -> str:
    """Return an error string if the claim fails provenance checks, else ''."""
    claim_ids = {re.sub(r"\s+", " ", m) for m in _IDENTIFIER.findall(claim.text)}
    for evidence in claim.evidence:
        if evidence.source_id not in sources:
            return "Citation references an unretrieved source"
        words = len(evidence.quote.split())
        if not MIN_QUOTE_WORDS <= words <= MAX_QUOTE_WORDS:
            return f"Citation quote must be {MIN_QUOTE_WORDS}-{MAX_QUOTE_WORDS} words"
        quote = fold(evidence.quote)
        if quote not in fold(sources[evidence.source_id]["text"]):
            return "Citation quote does not occur in retrieved text"
        missing = [i for i in claim_ids if fold(i) not in quote]
        if missing:
            return (
                "Claim names " + ", ".join(sorted(missing)) + " but the quote does not"
            )
    return ""


def validate(draft: Any, sources: dict[str, Any]) -> None:
    """Raise ValueError on the first provenance violation found.

    Kept for backward compatibility and the retry path in service.py.
    For per-claim filtering use filter_claims() instead.
    """
    for claim in draft.claims:
        error = _check_claim(claim, sources)
        if error:
            raise ValueError(error)


def filter_claims(draft: Any, sources: dict[str, Any]) -> tuple[list[Any], list[str]]:
    """Return (valid_claims, error_messages) without raising.

    Keeps every claim that passes provenance checks; collects errors for the
    rest. Lets the service surface a partial answer instead of a full rejection
    when only some claims are invalid.
    """
    valid, errors = [], []
    for claim in draft.claims:
        error = _check_claim(claim, sources)
        if error:
            errors.append(f"[{claim.text[:40]!r}] {error}")
        else:
            valid.append(claim)
    return valid, errors


def sources_from_hits(hits: list[Any], max_chars: int = 9000) -> dict[str, Any]:
    """Pack retrieved chunks into a context budget, best match first.

    Whole chunks only: a quote is never truncated silently. A chunk that does
    not fit is skipped, but a lower-ranked chunk that does fit is still used.

    The hits are re-sorted by distance before packing. The caller's order is
    the fused retrieval order, which is not distance order, so packing in the
    incoming order can spend the whole budget on weaker matches and drop the
    best one. Sorting first keeps the strongest evidence in the prompt.
    """
    if not hits:
        return {}
    ranked = sorted(hits, key=lambda h: h.distance)
    used = 0
    sources: dict[str, Any] = {}
    for hit in ranked:
        text = hit.chunk.text
        if used + len(text) > max_chars:
            continue  # Whole chunks only: never truncate a quote silently.
        used += len(text)
        sources[f"S{len(sources) + 1}"] = {
            "text": text,
            "chunk_id": hit.chunk.id,
            **hit.chunk.metadata,
            "distance": hit.distance,
        }
    return sources
