"""Check whether the gold answer text actually reached the model.

A retrieval metric and an answer metric answer different questions. A chunk
can be retrieved and still never reach the prompt, because the context budget
packs only what fits. And a model can be handed the answer and still decline
to use it.

This script separates those cases. For every row of an evaluation run it
checks, in the text that was actually sent to the model:

  - whether the gold quote is present (after the same typographic folding the
    citation validator applies), and
  - whether the row ended in an answer or an abstention.

The output is the table that shows which failures are retrieval failures and
which are the model's own choice.

Writes a JSON report to evaluation/gold-in-prompt.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grounded_qa.citations import fold  # noqa: E402


def gold_present(result: dict, gold_evidence: list[dict]) -> bool:
    """Is any gold quote present in the text actually sent to the model?"""
    prompt_text = " || ".join(
        fold(source.get("text", "")) for source in result.get("sources", {}).values()
    )
    if not prompt_text:
        return False
    return any(fold(gold["quote"]) in prompt_text for gold in gold_evidence)


def main() -> None:
    run_path = Path(sys.argv[1] if len(sys.argv) > 1 else "evaluation/runs/cpu-3b.json")
    queries = json.loads(
        (ROOT / "evaluation" / "queries.json").read_text(encoding="utf-8")
    )
    gold = {q["id"]: q for q in queries}
    run = json.loads(run_path.read_text(encoding="utf-8"))

    per_question: dict[str, list[dict]] = defaultdict(list)
    for row in run["rows"]:
        result = row.get("result")
        if not result:
            continue
        q = gold[row["query_id"]]
        per_question[row["query_id"]].append(
            {
                "repetition": row["repetition"],
                "status": result["status"],
                "gold_in_prompt": gold_present(result, q["gold_evidence"]),
                "sources": len(result.get("sources", {})),
                "retrieved": len(result.get("retrieved", [])),
            }
        )

    rows = []
    for qid in sorted(per_question):
        q = gold[qid]
        entries = per_question[qid]
        statuses = {e["status"] for e in entries}
        rows.append(
            {
                "id": qid,
                "answerable": q["answerable"],
                "category": q["category"],
                "required_facts": len(q["required_facts"]),
                "repeats": len(entries),
                "gold_in_prompt_repeats": sum(e["gold_in_prompt"] for e in entries),
                "statuses": sorted(statuses),
                "answered_in_any_repeat": "answered" in statuses,
            }
        )

    answerable = [r for r in rows if r["answerable"]]
    in_prompt_but_abstained = [
        r["id"]
        for r in answerable
        if r["gold_in_prompt_repeats"] == r["repeats"]
        and not r["answered_in_any_repeat"]
    ]
    missing_gold = [r["id"] for r in answerable if r["gold_in_prompt_repeats"] == 0]
    unanswerable_answered = [
        r["id"] for r in rows if not r["answerable"] and r["answered_in_any_repeat"]
    ]

    report = {
        "run": str(run_path),
        "questions": len(rows),
        "answerable": len(answerable),
        "gold_in_prompt_but_abstained": in_prompt_but_abstained,
        "gold_missing_from_prompt": missing_gold,
        "unanswerable_answered": unanswerable_answered,
        "rows": rows,
    }
    (ROOT / "evaluation" / "gold-in-prompt.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"run: {run_path}")
    print(f"answerable questions: {len(answerable)}")
    print(
        "gold text in the prompt in every repeat, yet abstained: "
        f"{len(in_prompt_but_abstained)} -> {in_prompt_but_abstained}"
    )
    print(f"gold text never reached the prompt: {len(missing_gold)} -> {missing_gold}")
    print(
        "unanswerable questions that were answered: "
        f"{len(unanswerable_answered)} -> {unanswerable_answered}"
    )
    print(f"\nwrote {ROOT / 'evaluation' / 'gold-in-prompt.json'}")


if __name__ == "__main__":
    main()
