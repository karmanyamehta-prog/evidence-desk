"""Deterministic metrics; semantic labels are supplied separately by reviewers."""


def precision_at_k(relevant_ids, retrieved_ids, k):
    ids = list(dict.fromkeys(retrieved_ids))[:k]
    return sum(i in relevant_ids for i in ids) / k


def faithfulness(claim_labels):
    allowed = {"supported", "unsupported", "contradicted", "non_factual"}
    if any(x not in allowed for x in claim_labels):
        raise ValueError("Unknown claim label")
    factual = [x for x in claim_labels if x != "non_factual"]
    if not factual:
        return None
    return sum(x == "supported" for x in factual) / len(factual)


def evidence_coverage(gold, retrieved):
    if not gold:
        return None
    covered = 0
    for span in gold:
        covered += any(
            h["metadata"]["document_id"] == span["document_id"]
            and h["metadata"]["page"] == span["page"]
            and h["metadata"]["start"] <= span["start"]
            and h["metadata"]["end"] >= span["end"]
            for h in retrieved
        )
    return covered / len(gold)


def hit_at_k(relevant_ids, retrieved_ids, k):
    """1.0 if any judged-relevant chunk is in the top k. Unlike P@k, not capped by the
    number of relevant chunks that exist (most questions here have exactly one)."""
    ids = list(dict.fromkeys(retrieved_ids))[:k]
    return float(any(i in relevant_ids for i in ids))


def reciprocal_rank(relevant_ids, retrieved_ids):
    for rank, i in enumerate(dict.fromkeys(retrieved_ids), 1):
        if i in relevant_ids:
            return 1 / rank
    return 0.0


def evidence_coverage_by_quote(gold, retrieved):
    """Like evidence_coverage, but anchored on the gold quote text rather than parser
    character offsets, so a parser change cannot silently move the labels."""
    from grounded_qa.citations import fold

    if not gold:
        return None
    texts = [fold(h["text"]) for h in retrieved]
    return sum(any(fold(g["quote"]) in t for t in texts) for g in gold) / len(gold)


def final_status(result):
    """Status for runs recorded before the `status` field existed."""
    if "status" in result:
        return result["status"]
    if not result["abstain"]:
        return "answered"
    return "abstained" if result["reason"] == "Insufficient evidence" else "rejected"
