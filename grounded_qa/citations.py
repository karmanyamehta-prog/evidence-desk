"""Validate provenance; these checks do NOT prove semantic entailment."""
from grounded_qa.ingestion.pdf import normalize

def validate(draft, sources):
    if draft.abstain:
        if draft.claims:
            raise ValueError("Abstention cannot include factual claims")
        return
    if not draft.claims:
        raise ValueError("Answer contains no supported claims")
    for claim in draft.claims:
        for evidence in claim.evidence:
            if evidence.source_id not in sources:
                raise ValueError("Citation references an unretrieved source")
            if normalize(evidence.quote) not in normalize(sources[evidence.source_id]["text"]):
                raise ValueError("Citation quote does not occur in retrieved text")

def sources_from_hits(hits, max_chars=9000):
    sources, used = {}, 0
    for hit in hits:
        text = hit.chunk.text
        if used + len(text) > max_chars:
            continue  # Whole chunks only: never truncate a quote silently.
        used += len(text)
        sources[f"S{len(sources)+1}"] = {"text": text, "chunk_id": hit.chunk.id,
                                        **hit.chunk.metadata, "distance": hit.distance}
    return sources
