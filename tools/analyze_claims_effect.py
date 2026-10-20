"""Do the answers rescued by removing the distractors actually hold up?

`tools/probe_context_noise_controlled.py` found that the model answers more
often with the gold chunk alone than with the full retrieved context.
`tools/probe_context_noise_claims.py` captured the actual claims for both
conditions, and `tools/analyze_noise_effect.py` showed that one of the
differing questions (q06) was confounded: its gold text was not in the
retrieved prompt at all, because the context budget dropped it.

This script joins the two datasets and answers the question that actually
decides whether removing distractors is worth doing:

  - of the questions the gold-only condition rescues, how many of the
    rescued answers are *grounded* (every quote present in the evidence
    that was supplied) and *complete* (every required fact recalled)?

An answer rate that goes up while the quality of the extra answers goes
down is not an improvement, it is a different kind of error. And q06 has
to be excluded from the rescued set, because it is rescued by the absence
of the answer, not by the absence of the distractors.

Writes a JSON report to tools/context_noise_claims_analysis.json.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def exact_binomial_two_sided(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def main() -> None:
    claims = json.loads(
        (ROOT / "tools" / "context_noise_claims.json").read_text(encoding="utf-8")
    )
    presence = json.loads(
        (ROOT / "tools" / "context_noise_gold_presence.json").read_text(
            encoding="utf-8"
        )
    )
    queries = {
        q["id"]: q
        for q in json.loads(
            (ROOT / "evaluation" / "queries.json").read_text(encoding="utf-8")
        )
    }

    gold_in_prompt = {
        r["id"]: r["gold_span_in_retrieved_prompt"] for r in presence["rows"]
    }

    per_question = []
    for row in claims["rows"]:
        if not row.get("gold_chunk_retrieved"):
            continue
        qid = row["id"]
        rounds = row["rounds"]
        ret_answers = [rd["retrieved"] for rd in rounds if rd["retrieved"]["answered"]]
        gold_answers = [rd["gold_only"] for rd in rounds if rd["gold_only"]["answered"]]
        per_question.append(
            {
                "id": qid,
                "category": queries[qid]["category"],
                "gold_in_retrieved_prompt": gold_in_prompt.get(qid),
                "retrieved_answered_any": bool(ret_answers),
                "gold_only_answered_any": bool(gold_answers),
                "rescued": bool(gold_answers) and not ret_answers,
                "lost": bool(ret_answers) and not gold_answers,
                "rescued_answers_grounded": (
                    all(c["quotes_grounded"] for c in gold_answers)
                    if gold_answers
                    else None
                ),
                "rescued_answers_complete": (
                    all(
                        c["n_facts_recalled"] == c["n_facts_required"]
                        for c in gold_answers
                    )
                    if gold_answers
                    else None
                ),
                "rescued_mean_facts_recalled": (
                    round(
                        sum(c["n_facts_recalled"] for c in gold_answers)
                        / len(gold_answers),
                        3,
                    )
                    if gold_answers
                    else None
                ),
                "rescued_required_facts": len(queries[qid]["required_facts"]),
            }
        )

    clean = [r for r in per_question if r["gold_in_retrieved_prompt"]]
    rescued = [r for r in clean if r["rescued"]]
    lost = [r for r in clean if r["lost"]]

    def quality(rows):
        return {
            "questions": len(rows),
            "grounded": sum(1 for r in rows if r["rescued_answers_grounded"]),
            "complete": sum(1 for r in rows if r["rescued_answers_complete"]),
            "grounded_and_complete": sum(
                1
                for r in rows
                if r["rescued_answers_grounded"] and r["rescued_answers_complete"]
            ),
        }

    summary = {
        "questions": len(clean),
        "excluded_because_gold_span_not_in_retrieved_prompt": [
            r["id"] for r in per_question if not r["gold_in_retrieved_prompt"]
        ],
        "rescued_by_gold_only": [r["id"] for r in rescued],
        "lost_by_gold_only": [r["id"] for r in lost],
        "rescued_answer_quality": quality(rescued),
        "question_level_exact_binomial_two_sided_p": round(
            exact_binomial_two_sided(len(lost), len(rescued)), 5
        ),
        "per_question": per_question,
    }
    (ROOT / "tools" / "context_noise_claims_analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
