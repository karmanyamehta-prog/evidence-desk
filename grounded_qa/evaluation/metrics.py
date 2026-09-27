"""Deterministic metrics; semantic labels are supplied separately by reviewers."""


def precision_at_k(relevant_ids, retrieved_ids, k):
    ids = list(dict.fromkeys(retrieved_ids))[:k]
    return sum(i in relevant_ids for i in ids) / k


def faithfulness(claim_labels):
    if not claim_labels:
        return None
    if any(x not in {"supported", "unsupported", "contradicted"} for x in claim_labels):
        raise ValueError("Unknown claim label")
    return sum(x == "supported" for x in claim_labels) / len(claim_labels)


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
