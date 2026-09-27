"""Build a static, explicitly recorded showcase from the measured run."""

import html
import json
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "deploy/static"
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / "evaluation/results/summary.json").read_text())
    run = json.loads((root / "evaluation/runs/cpu-1.5b.json").read_text())
    records = {r["query_id"]: r["result"] for r in run["rows"] if "result" in r}
    cards = []
    examples = {
        "q09": "A supported action lookup",
        "q06": "A grounded answer that misses the question",
        "q13": "An attribution error despite a valid quote",
        "q01": "A false abstention",
    }
    for query_id, label in examples.items():
        result = records[query_id]
        body = ""
        for claim in result["claims"]:
            body += f"<p>{html.escape(claim['text'])}</p>"
            for evidence in claim["evidence"]:
                source = result["sources"][evidence["source_id"]]
                url = source["source_url"] + f"#page={source['page']}"
                body += (
                    f"<details><summary>Evidence · PDF page {source['page']}</summary>"
                    f"<blockquote>{html.escape(evidence['quote'])}</blockquote>"
                    f'<a href="{html.escape(url, quote=True)}">Open source PDF</a></details>'
                )
        if result["abstain"]:
            body = f"<p>Abstained: {html.escape(result['reason'])}</p>"
        cards.append(
            f"<article><p class='eyebrow'>{label}</p>"
            f"<h3>{html.escape(result['question'])}</h3>{body}</article>"
        )
    page = """<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Evidence Desk · recorded evaluation</title>
<style>
body{margin:0;background:#f5f3ed;color:#192d3b;font:17px/1.65 system-ui,sans-serif}
main{max-width:1000px;margin:auto;padding:48px 24px}h1{font-size:clamp(36px,6vw,60px);line-height:1.1}
h2{margin-top:48px}h3{line-height:1.4}a{color:#165d87}
.eyebrow{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#526a77}
.notice{background:#e6edf0;padding:18px 24px;border-left:4px solid #286d8a}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}
article{background:white;border:1px solid #d8dfdf;border-radius:8px;padding:24px}
summary{cursor:pointer;color:#165d87}blockquote{margin:12px 0;padding:12px;border-left:3px solid #8babb8}
img{width:100%;height:auto;background:white}footer{margin-top:48px;color:#526a77}
</style><main><p class="eyebrow">Local RAG · source audit · CPU evaluation</p>
<h1>Evidence Desk</h1><p>Trace AI governance answers back to their source passages.</p>
<p class="notice"><strong>Recorded showcase.</strong> These are saved outputs from a real
24-question run. This static page does not run a model or accept new questions.
The downloadable Python project provides the live local application.</p>
<h2>Measured results, with the failures visible</h2>
<p>Corpus: NIST AI RMF 1.0 (2023), one 48-page PDF. Model: Qwen2.5 1.5B Q4_K_M,
CPU-only, one exploratory pass on the build machine. These are not laptop timings.
Labels are assistant-reviewed; independent human review is pending.</p>
<img src="metrics.png" alt="Measured retrieval precision, statement faithfulness, answer completeness and latency">
<p>SUMMARY</p><h2>Inspect the evidence</h2><div class="grid">CARDS</div>
<footer>The system verifies quote membership, not entailment. A valid citation can still accompany
an irrelevant or misattributed answer. Local-first, single-user portfolio prototype;
not a production compliance product. Nothing has been deployed by this build.</footer></main></html>
"""
    text = (
        f"Correct abstentions: {summary['correct_abstentions']}/{summary['unanswerable_queries']}. "
        f"False abstentions: {summary['false_abstentions']}/{summary['answerable_queries']}. "
        "Heading-only responses and abstentions do not inflate faithfulness."
    )
    page = page.replace("SUMMARY", text).replace("CARDS", "".join(cards))
    (output / "index.html").write_text(page, encoding="utf-8")
    shutil.copyfile(root / "evaluation/results/metrics.png", output / "metrics.png")


if __name__ == "__main__":
    main()
