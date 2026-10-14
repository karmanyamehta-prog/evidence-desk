# Changes in the hardening pass

Scope: fixes from a code and evaluation review of commit `9f1c210`. Verified by the
test suite (37 passing, Linux, Python 3.12) and by replaying the *recorded* 1.5B
answers through the new code. **No model was run: Ollama, real E5 weights, Docker,
Windows and CI were not available, so none of those are verified here.**

## Fixed

| Area | Change | Evidence |
|---|---|---|
| Citation check | Quote comparison folds curly quotes, dashes, Unicode forms and line-break hyphens (`citations.fold`; stored text and offsets untouched). | `tests/test_citations.py` |
| Citation check | Quote must be 5-70 words (the old limit was 12 *characters*). A section ID named in a claim must appear in its quote. | Replay: rejects q13, q14, q16 (wrong IDs); the other 15 recorded answers still pass, including q20 and q22, which the reviewer judged unsupported or off-topic. |
| Status | Responses carry `status` (answered, abstained, rejected, error), `attempts` (raw output, error, ms per try) and `dropped_by_context_budget`. UI no longer says "cannot answer from evidence" for a JSON failure. | `tests/test_service.py` |
| Retry | Attempt number is passed to the generator (seed offset). The retry prompt already included the validation error, so it was never an identical repeat. | |
| JSON parsing | Tolerates markdown fences and chatter around the object (needed for the unverified Anthropic/OpenAI adapters). | `test_extract_json_...` |
| Ingest | Index first, then write files; a failed embed removes the directory. `GET /documents` lists only documents in the active index. | `tests/test_integration.py` |
| UI | Source PDF is fetched once per document (cached 10 min), not once per citation per rerun. Download keys no longer use `id()`. | |
| Packaging | `pytest` and `matplotlib` moved to `requirements-dev.txt`; `python-dotenv` pinned (the UI imports it); CI installs dev deps. | |
| CLI | Manifest found via `CORPUS_MANIFEST`, then `./corpus`, then next to the package. `fetch` sends a User-Agent. | `test_manifest_path_...` |
| Embeddings | E5 prefix detection uses a name pattern instead of the substring `"e5"`; `EMBEDDING_PREFIXES` overrides it. | `test_e5_prefix_...` |
| PDF | Hash computed once, not per table page; a "Subcategories" page with no ruled lines falls back to normal extraction instead of failing the upload. | |
| API | Rejects unexpected `Host` (DNS rebinding) and cross-origin browser POSTs; `/health` reports whether the Ollama model is present without failing when it is not. | `tests/test_api.py` |
| Evaluation | Added Hit@k, MRR, status split and a quote-anchored evidence recall (identical to the offset recall, 0.85). Original summary values are unchanged. `evaluation/results/` regenerated from the checked-in run and labels. | `metrics.py`, `report.py` |
| Docs | README, DESIGN and RESEARCH no longer carry assistant-to-user wording or a personal workplace reference; DESIGN lists what changed since the proposal. | |

## Findings from the recorded run (no new model run)

- All 6 final abstentions were validation rejections. The model never returned an
  empty answer by itself, so the "2/4 correct abstentions" says nothing about
  recognising missing evidence.
- Top-1 retrieval distance: answerable 0.074-0.159; unanswerable 0.117, 0.134,
  0.168, 0.210. A cutoff near 0.16 would catch 2 of 4 unanswerable and 0 of 20
  answerable, but n=4, so it is not implemented.
- Precision@5 is capped by the labels (13 of 20 questions have one relevant chunk).
- One rejected claim rejects the whole answer (q13, q14 each had a correct claim).
  Per-claim filtering is a design choice not made here.

## Not done (needs your machine, data or judgment)

1. Re-run the 24 queries with the new validator, and the 3B default on your laptop
   (3 warm repeats, peak RAM). The published numbers are the 1.5B pre-hardening baseline.
2. Independent labels: re-judge all outputs yourself, keep the originals, add 20+
   fresh questions (at least 10 unanswerable), measure agreement.
3. Strip repeated page headers (they start 17 of 76 retrieved chunks) and merge
   tiny chunks (median 300 chars). This moves text offsets and the frozen gold spans,
   and the NIST PDF is not in the repo, so it needs re-extraction and re-labelling.
4. Run CI once, Docker, native Windows, and the Spaces image.
5. A second, different PDF through the pipeline (the table rule is tied to one hash).
6. Heading/action ID in the embedded text, bounding boxes, printed page labels.
7. `requirements-linux.lock.txt` was not regenerated and no install uses it.
