"""Are the extra answers the model gives on gold-only evidence actually right?

`tools/probe_context_noise_controlled.py` measured that the model answers more
often when given only the gold chunk than when given the full retrieved
context: per-call answer rate 0.77 versus 0.58, exact binomial two-sided
p = 0.019 on the discordant rounds. That measures *whether* the model answers,
not whether the answer is usable. A higher answer rate is only worth having
if the extra answers are grounded and on-topic.

This script runs the same alternating, same-session, three-round design and
captures the actual claims, so each answer is checked mechanically:

  - `quotes_grounded`: every quote the model cited occurs verbatim (after the
    same typographic folding the citation validator applies) in the evidence
    that was supplied to the model in that condition.
  - `facts_recalled`: how many of the question's required fact strings appear
    in the answer text. This is a lexical recall check, not a semantic
    judgement, and is reported as such.

The interesting comparison is not the aggregate but the *discordant* set: the
questions the model answers under gold-only but abstains on under the full
retrieved context. If those extra answers are grounded and recall their
required facts, then dropping the distractors is a real improvement rather
than a trade of one error for another.

Writes a JSON report to tools/context_noise_claims.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from grounded_qa.citations import fold  # noqa: E402
from grounded_qa.generation.client import SYSTEM, extract_json  # noqa: E402
from grounded_qa.ingestion.chunking import chunk_pages  # noqa: E402
from grounded_qa.ingestion.pdf import read_pdf  # noqa: E402
from grounded_qa.models import Draft  # noqa: E402
from grounded_qa.retrieval.embeddings import LocalEmbeddings  # noqa: E402

DOC = "cde9f5bab41c68718f3413ab2564c2285ac9195fa9160a61cc5058dd9c35d6e7"
MODEL = "qwen2.5:3b-instruct-q4_K_M"
CONTEXT_CHARS = 4500
CHUNK_TOKENS = 350
TOP_K = 5
ROUNDS = 3


def ask(question: str, sources: dict[str, str]) -> dict:
    """Ask exactly as the service does, including keep_alive and num_gpu."""
    prompt = json.dumps(
        {
            "question": question,
            "evidence": {k: {"text": v} for k, v in sources.items()},
            "schema": Draft.model_json_schema(),
            "validation_error": "",
        }
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": Draft.model_json_schema(),
        "keep_alive": "5m",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 4096,
            "num_predict": 384,
            "num_thread": 4,
            "num_gpu": 0,
        },
    }
    with httpx.Client(timeout=180, trust_env=False) as client:
        response = client.post("http://localhost:11434/api/chat", json=payload)
    text = response.json()["message"]["content"]
    try:
        return json.loads(extract_json(text))
    except json.JSONDecodeError:
        return {"claims": [], "parse_error": True}


def covers(chunk, gold) -> bool:
    meta = chunk.metadata
    return (
        meta["page"] == gold["page"]
        and meta["start"] <= gold["start"]
        and meta["end"] >= gold["end"]
    )


def quote_grounded(claims: list[dict], evidence: str) -> bool:
    """Is every cited quote present in the evidence that was supplied?"""
    if not claims:
        return False
    for claim in claims:
        for ev in claim.get("evidence", []):
            quote = fold(ev.get("quote", ""))
            if not quote or quote not in evidence:
                return False
    return True


def facts_recalled(claims: list[dict], required: list[str]) -> list[bool]:
    """Which required fact strings appear in the answer text?

    Lexical recall, not semantic truth. It answers "did the answer mention the
    things the question asked for", which is not the same as "is the answer
    correct", and is reported as such.

    The check is known to under-report: 16 of the 50 required fact strings
    across the 20 answerable questions are paraphrases rather than literal
    substrings of their own gold quote ("time interval and conditions" against
    a gold quote reading "for a given time interval, under given
    conditions"). A fact marked False here may well have been conveyed in
    different words. The measurement is therefore a lower bound on recall,
    and the numbers are read as a floor, not as a verdict on correctness.
    """
    text = fold(" ".join(c.get("text", "") for c in claims))
    return [fold(f) in text for f in required]


def main() -> None:
    embedder = LocalEmbeddings(
        "intfloat/e5-small-v2", "ffb93f3bd4047442299a41ebb6fa998a38507c52", 4, None
    )
    pages = read_pdf(ROOT / "data" / "sources" / "nist-ai-100-1-rmf.pdf", 200)
    queries = json.loads(
        (ROOT / "evaluation" / "queries.json").read_text(encoding="utf-8")
    )
    answerable = [q for q in queries if q["answerable"]]

    chunks = chunk_pages(pages, DOC, "RMF", embedder.tokenizer, CHUNK_TOKENS, 50)
    vectors = np.array(embedder.passages([c.embed_prefix + c.text for c in chunks]))

    rows = []
    for q in answerable:
        query_vector = np.array(embedder.queries([q["question"]])[0])
        distances = 1 - vectors @ query_vector
        order = np.argsort(distances)[:TOP_K]

        retrieved: dict[str, str] = {}
        used = 0
        for i in order:
            chunk = chunks[i]
            if used + len(chunk.text) > CONTEXT_CHARS:
                continue
            used += len(chunk.text)
            retrieved[f"S{len(retrieved) + 1}"] = chunk.text

        gold_chunk = next(
            (
                chunks[i]
                for i in order
                if any(covers(chunks[i], g) for g in q["gold_evidence"])
            ),
            None,
        )
        if gold_chunk is None:
            rows.append(
                {
                    "id": q["id"],
                    "gold_chunk_retrieved": False,
                    "rounds": [],
                }
            )
            print(json.dumps(rows[-1]), flush=True)
            continue

        gold_only = {"S1": gold_chunk.text}

        # Alternate the conditions so drift hits both arms equally.
        rounds = []
        for _ in range(ROUNDS):
            round_row = {}
            for name, sources in (("retrieved", retrieved), ("gold_only", gold_only)):
                evidence = fold(" ".join(sources.values()))
                claims = ask(q["question"], sources).get("claims", [])
                recalled = facts_recalled(claims, q["required_facts"])
                round_row[name] = {
                    "answered": bool(claims),
                    "n_claims": len(claims),
                    "quotes_grounded": quote_grounded(claims, evidence),
                    "facts_recalled": recalled,
                    "n_facts_recalled": sum(recalled),
                    "n_facts_required": len(q["required_facts"]),
                    "claim_texts": [c.get("text", "") for c in claims],
                }
            rounds.append(round_row)
            print(
                json.dumps(
                    {
                        "id": q["id"],
                        "retrieved": round_row["retrieved"],
                        "gold_only": round_row["gold_only"],
                    }
                ),
                flush=True,
            )

        rows.append(
            {
                "id": q["id"],
                "gold_chunk_retrieved": True,
                "rounds": rounds,
            }
        )

    comparable = [r for r in rows if r["gold_chunk_retrieved"]]

    def calls(condition: str) -> list[dict]:
        return [rd[condition] for r in comparable for rd in r["rounds"]]

    def summarise(condition: str) -> dict:
        all_calls = calls(condition)
        answered = [c for c in all_calls if c["answered"]]
        complete = [
            c for c in answered if c["n_facts_recalled"] == c["n_facts_required"]
        ]
        return {
            "calls": len(all_calls),
            "answers": len(answered),
            "answers_with_grounded_quotes": sum(
                1 for c in answered if c["quotes_grounded"]
            ),
            "answers_recalling_every_required_fact": len(complete),
            "answers_grounded_and_complete": sum(
                1 for c in answered if c["quotes_grounded"] and c in complete
            ),
            "mean_facts_recalled": (
                round(sum(c["n_facts_recalled"] for c in answered) / len(answered), 3)
                if answered
                else None
            ),
        }

    # The discordant set: answers that exist only because the distractors were
    # removed. This is the set that has to hold up for the hypothesis to be
    # worth acting on.
    rescued = []
    for r in comparable:
        r_ans = any(rd["retrieved"]["answered"] for rd in r["rounds"])
        g_ans = any(rd["gold_only"]["answered"] for rd in r["rounds"])
        if g_ans and not r_ans:
            gold_calls = [
                rd["gold_only"] for rd in r["rounds"] if rd["gold_only"]["answered"]
            ]
            rescued.append(
                {
                    "id": r["id"],
                    "gold_only_answers": len(gold_calls),
                    "all_quotes_grounded": all(
                        c["quotes_grounded"] for c in gold_calls
                    ),
                    "all_recall_every_fact": all(
                        c["n_facts_recalled"] == c["n_facts_required"]
                        for c in gold_calls
                    ),
                    "claim_texts": gold_calls[0]["claim_texts"],
                }
            )

    lost = []
    for r in comparable:
        r_ans = any(rd["retrieved"]["answered"] for rd in r["rounds"])
        g_ans = any(rd["gold_only"]["answered"] for rd in r["rounds"])
        if r_ans and not g_ans:
            lost.append(r["id"])

    summary = {
        "questions": len(comparable),
        "rounds": ROUNDS,
        "retrieved": summarise("retrieved"),
        "gold_only": summarise("gold_only"),
        "rescued_by_gold_only": rescued,
        "lost_by_gold_only": lost,
        "note": (
            "facts_recalled is a lexical recall check over the required fact "
            "strings, not a semantic faithfulness judgement. "
            "quotes_grounded applies the service's own folding rule to the "
            "evidence supplied in that condition."
        ),
    }
    (ROOT / "tools" / "context_noise_claims.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
