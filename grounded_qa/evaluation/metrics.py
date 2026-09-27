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
