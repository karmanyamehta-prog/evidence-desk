"""Create a README-ready report from a real run and explicit review labels."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grounded_qa.evaluation.metrics import (
    evidence_coverage,
    evidence_coverage_by_quote,
    faithfulness,
    final_status,
    hit_at_k,
    precision_at_k,
    reciprocal_rank,
)


def mean(values):
    return statistics.mean(values) if values else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=Path("evaluation/runs/cpu-1.5b.json"))
    p.add_argument("--labels", type=Path, default=Path("evaluation/review.json"))
    p.add_argument("--out", type=Path, default=Path("evaluation/results"))
    args = p.parse_args()
    run = json.loads(args.run.read_text())
    labels = json.loads(args.labels.read_text())
    gold = {q["id"]: q for q in json.loads(Path("evaluation/queries.json").read_text())}
    rows = []
    for r in run["rows"]:
        if r["repetition"] != 0:
            continue
        q = gold[r["query_id"]]
        review = labels[r["query_id"]]
        if not 0 <= review["correct_facts"] <= len(q["required_facts"]):
            raise ValueError("Reviewed fact count outside gold bounds")
        result = r.get("result")
        if result:
            ids = [h["id"] for h in result["retrieved"]]
            if len(review["claim_labels"]) != len(result["claims"]):
                raise ValueError("Review claim count mismatch")
            if any(x not in ids for x in review["relevant_chunk_ids"]):
                raise ValueError("Review IDs outside retrieval")
            f = faithfulness(review["claim_labels"])
            row = {
                "id": q["id"],
                "split": q["split"],
                "answerable": q["answerable"],
                "p1": precision_at_k(set(review["relevant_chunk_ids"]), ids, 1)
                if q["answerable"]
                else None,
                "p3": precision_at_k(set(review["relevant_chunk_ids"]), ids, 3)
                if q["answerable"]
                else None,
                "p5": precision_at_k(set(review["relevant_chunk_ids"]), ids, 5)
                if q["answerable"]
                else None,
                "hit5": hit_at_k(set(review["relevant_chunk_ids"]), ids, 5)
                if q["answerable"]
                else None,
                "hit3": hit_at_k(set(review["relevant_chunk_ids"]), ids, 3)
                if q["answerable"]
                else None,
                "rr": reciprocal_rank(set(review["relevant_chunk_ids"]), ids)
                if q["answerable"]
                else None,
                "recall_by_quote": evidence_coverage_by_quote(
                    q["gold_evidence"], result["retrieved"]
                ),
                "status": final_status(result),
                "faithfulness": f,
                "completeness": review["correct_facts"] / len(q["required_facts"])
                if q["required_facts"]
                else None,
                "evidence_recall": evidence_coverage(
                    q["gold_evidence"], result["retrieved"]
                ),
                "abstain": result["abstain"],
                "latency_s": result["timing_ms"]["total"] / 1000,
                "error": None,
            }
        else:
            row = {
                "id": q["id"],
                "split": q["split"],
                "answerable": q["answerable"],
                "p1": None,
                "p3": None,
                "p5": None,
                "hit5": None,
                "hit3": None,
                "rr": None,
                "recall_by_quote": None,
                "status": "error",
                "faithfulness": None,
                "completeness": 0 if q["answerable"] else None,
                "evidence_recall": None,
                "abstain": None,
                "latency_s": r["elapsed_seconds"],
                "error": r["error"],
            }
        rows.append(row)
    if len(rows) != 24:
        raise ValueError("Report requires all 24 query outcomes, including failures")
    ps = [r["p5"] for r in rows if r["p5"] is not None]
    fs = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    complete = [r["completeness"] for r in rows if r["completeness"] is not None]
    unanswerable = [r for r in rows if not r["answerable"]]
    answerable = [r for r in rows if r["answerable"]]
    statements = [
        label
        for row in rows
        for label in labels[row["id"]]["claim_labels"]
        if label != "non_factual"
    ]
    summary = {
        "queries": len(rows),
        "precision_at_1": mean([r["p1"] for r in rows if r["p1"] is not None]),
        "precision_at_3": mean([r["p3"] for r in rows if r["p3"] is not None]),
        "precision_at_5": mean(ps),
        "retrieval_scored_queries": len(ps),
        "hit_at_3": mean([r["hit3"] for r in rows if r["hit3"] is not None]),
        "hit_at_5": mean([r["hit5"] for r in rows if r["hit5"] is not None]),
        "mrr": mean([r["rr"] for r in rows if r["rr"] is not None]),
        "gold_span_recall_at_5_by_quote": mean(
            [r["recall_by_quote"] for r in rows if r["recall_by_quote"] is not None]
        ),
        "answered": sum(r["status"] == "answered" for r in rows),
        "model_abstentions": sum(r["status"] == "abstained" for r in rows),
        "validation_rejections": sum(r["status"] == "rejected" for r in rows),
        "model_abstentions_on_unanswerable": sum(
            r["status"] == "abstained" for r in unanswerable
        ),
        "gold_span_recall_at_5": mean(
            [r["evidence_recall"] for r in rows if r["evidence_recall"] is not None]
        ),
        "faithfulness_macro": mean(fs),
        "faithfulness_scored_answers": len(fs),
        "factual_statements": len(statements),
        "supported_statements": sum(x == "supported" for x in statements),
        "faithfulness_micro": faithfulness(statements),
        "completeness": mean(complete),
        "correct_abstentions": sum(r["abstain"] is True for r in unanswerable),
        "unanswerable_queries": len(unanswerable),
        "false_abstentions": sum(r["abstain"] is True for r in answerable),
        "answerable_queries": len(answerable),
        "errors": sum(bool(r["error"]) for r in rows),
        "latency_p50_s": statistics.median(r["latency_s"] for r in rows),
        "latency_p95_s": float(
            __import__("numpy").percentile([r["latency_s"] for r in rows], 95)
        ),
        "reviewer": "assistant; independent human validation pending",
        "quality_protocol": "exploratory, single pass; dev/test assignments retained but not a sealed holdout",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "per-query.json").write_text(json.dumps(rows, indent=2))

    def fmt(x):
        return "N/A" if x is None else f"{x:.3f}"

    text = [
        "# Evaluation results",
        "",
        f"Model: `{run['settings']['llm_model']}`. 24 real local queries. One pass; not laptop benchmarks.",
        "",
        "Labels are assistant-reviewed, not independent human judgments. Faithfulness is scored only on factual statements; abstentions and heading-only outputs have N/A, and completeness penalizes missed answers. Latency includes cold first-query effects and any retries. Separate p50/p95 are descriptive, not stable warm-load estimates.",
        "",
        "| Query | Split | P@5 | Faithfulness | Completeness | Abstained | Seconds |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for r in rows:
        text.append(
            f"| {r['id']} | {r['split']} | {fmt(r['p5'])} | {fmt(r['faithfulness'])} | {fmt(r['completeness'])} | {r['abstain']} | {r['latency_s']:.2f} |"
        )
    text.extend(["", "## Aggregate", "", "| Metric | Value |", "|---|---:|"])
    for key, value in summary.items():
        text.append(f"| {key} | {fmt(value) if isinstance(value, float) else value} |")
    text.extend(
        [
            "",
            "![Measured scores](metrics.png)",
            "",
            "All raw answers, citations, and timings are retained in the run JSON; review.json supplies semantic labels and rationales. P@5 uses judged relevance, not same-page matching or citation validity. Alternate relevant passages count even when not in the selected gold spans. The errors count covers requests that raised exceptions; validation-rejection abstentions still count as failures in answer metrics.",
        ]
    )
    (args.out / "RESULTS.md").write_text("\n".join(text) + "\n")
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    values = [
        summary["hit_at_5"],
        summary["mrr"],
        summary["faithfulness_macro"],
        summary["completeness"],
    ]
    bars = axes[0].bar(
        ["Hit@5", "MRR", "Faithfulness*", "Completeness"],
        [v or 0 for v in values],
        color=["#276b8e", "#4f8fb0", "#399b8f", "#dc9960"],
    )
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Score (0–1)")
    axes[0].set_title("Quality: assistant-reviewed")
    for bar, value in zip(bars, values, strict=False):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            fmt(value),
            ha="center",
        )
    bars = axes[1].bar(
        ["p50", "p95"],
        [summary["latency_p50_s"], summary["latency_p95_s"]],
        color="#5d6d87",
    )
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("End-to-end latency (single pass)")
    for bar in bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
        )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Evidence Desk · measured CPU evaluation", fontweight="bold")
    fig.text(
        0.5,
        0.01,
        f"*Faithfulness excludes abstentions and headings ({len(fs)} scored answers). Independent human review pending.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(args.out / "metrics.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
