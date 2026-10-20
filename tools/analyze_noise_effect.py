"""Analyse the context-noise probe, with the budget confound removed.

`tools/probe_context_noise_controlled.py` reported a paired difference
favouring the gold-only condition, with an exact binomial two-sided p of
0.019 on the discordant rounds. `tools/check_noise_gold_presence.py` then
showed that for one of the differing questions (q06) the gold span was *not*
in the retrieved prompt at all: its gold chunk ranked 5th and the context
budget was already spent, so it was dropped.

That question therefore does not test distractors. It tests "gold present"
against "gold absent", which is the budget failure already ruled out
elsewhere, and it is counted in the wrong direction here. It has to be
separated out before the headline number means anything.

This script recomputes the comparison in three ways, so the reader can see
which conclusions survive:

  1. round-level paired, all questions;
  2. round-level paired, excluding the questions where the gold span was
     not in the retrieved prompt;
  3. question-level, which is the unit the design actually samples
     independently - the 3 rounds of one question are not independent
     draws, so a round-level test overstates the evidence.

It also reports the direction per question, because a clean effect would
move every question the same way. It does not.

Writes a JSON report to tools/context_noise_analysis.json.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def exact_binomial_two_sided(a: int, b: int) -> float:
    """Exact two-sided binomial p for a vs b under p=0.5.

    The two-sided p is the total probability of outcomes at least as
    extreme as the observed one in either direction, which for the
    symmetric p=0.5 case is 2 * P(X <= min(a,b)).
    """
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def main() -> None:
    probe = json.loads(
        (ROOT / "tools" / "context_noise_controlled.json").read_text(encoding="utf-8")
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

    rows = [r for r in probe["rows"] if r.get("gold_chunk_retrieved")]
    per_question = []
    for r in rows:
        qid = r["id"]
        ret = r["retrieved"]
        gold = r["gold_only"]
        per_question.append(
            {
                "id": qid,
                "category": queries[qid]["category"],
                "gold_in_retrieved_prompt": gold_in_prompt.get(qid),
                "retrieved": ret,
                "gold_only": gold,
                "retrieved_answered_any": any(ret),
                "gold_only_answered_any": any(gold),
                "retrieved_rounds_answered": sum(ret),
                "gold_only_rounds_answered": sum(gold),
                "unstable_within_a_condition": len(set(ret)) > 1 or len(set(gold)) > 1,
            }
        )

    def round_counts(question_rows):
        a = b = 0
        for r in question_rows:
            for ret, gold in zip(r["retrieved"], r["gold_only"], strict=True):
                if gold and not ret:
                    b += 1
                elif ret and not gold:
                    a += 1
        return a, b

    all_q = per_question
    clean_q = [r for r in per_question if r["gold_in_retrieved_prompt"]]
    excluded = [r["id"] for r in per_question if not r["gold_in_retrieved_prompt"]]

    a_all, b_all = round_counts(all_q)
    a_clean, b_clean = round_counts(clean_q)

    def question_counts(question_rows):
        rescued = [
            r["id"]
            for r in question_rows
            if r["gold_only_answered_any"] and not r["retrieved_answered_any"]
        ]
        lost = [
            r["id"]
            for r in question_rows
            if r["retrieved_answered_any"] and not r["gold_only_answered_any"]
        ]
        same = [
            r["id"]
            for r in question_rows
            if r["retrieved_answered_any"] == r["gold_only_answered_any"]
        ]
        return rescued, lost, same

    res_all, lost_all, same_all = question_counts(all_q)
    res_clean, lost_clean, same_clean = question_counts(clean_q)

    summary = {
        "questions_in_probe": len(all_q),
        "excluded_because_gold_span_not_in_retrieved_prompt": excluded,
        "round_level_all_questions": {
            "discordant_retrieved_only": a_all,
            "discordant_gold_only_only": b_all,
            "exact_binomial_two_sided_p": round(
                exact_binomial_two_sided(a_all, b_all), 5
            ),
        },
        "round_level_excluding_budget_confound": {
            "discordant_retrieved_only": a_clean,
            "discordant_gold_only_only": b_clean,
            "exact_binomial_two_sided_p": round(
                exact_binomial_two_sided(a_clean, b_clean), 5
            ),
        },
        "question_level_all_questions": {
            "rescued_by_gold_only": res_all,
            "lost_by_gold_only": lost_all,
            "same_status": same_all,
            "exact_binomial_two_sided_p": round(
                exact_binomial_two_sided(len(lost_all), len(res_all)), 5
            ),
        },
        "question_level_excluding_budget_confound": {
            "rescued_by_gold_only": res_clean,
            "lost_by_gold_only": lost_clean,
            "same_status": same_clean,
            "exact_binomial_two_sided_p": round(
                exact_binomial_two_sided(len(lost_clean), len(res_clean)), 5
            ),
        },
        "per_question": per_question,
    }
    (ROOT / "tools" / "context_noise_analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
