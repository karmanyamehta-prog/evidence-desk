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

---

# Phase 2 — retrieval and validation

Scope: the four Phase 2 items that can be verified on the i5-1240P laptop
(Windows 11, 15.7 GB RAM), with Ollama 0.40.5 and `qwen2.5:3b-instruct-q4_K_M`
actually running. **Every number below is measured on that machine unless it says
otherwise.** The model was downloaded and run; nothing here is a projection.

## Fixed

| Area | Change | Evidence |
|---|---|---|
| Per-claim validation | `citations.filter_claims` keeps every claim that passes provenance and reports the failures, instead of rejecting the whole draft for one bad citation. `service` accepts a partially-valid draft. | `tests/test_phase2.py` (10 tests) |
| Status | A partially-valid answer is `answered`; a model-initiated abstention is `abstained`; a draft whose every claim failed validation is `rejected`. These were previously conflated. | `tests/test_phase2.py` |
| Retry | A partially-valid draft is accepted without a retry. A model-initiated abstention is accepted without a retry (it used to be treated as a failure and triggered a second call, then `StopIteration`). | `tests/test_phase2.py` |
| Action-ID embedding | The action/section ID is prepended to the **embedding text only**; the stored chunk text stays the verbatim source passage, so quotes and offsets are unaffected. RMF `GOVERN 1.6` and Generative AI Profile `A.1.1` are both recognised. | `tests/test_phase2.py`, `tests/test_corpus.py` |
| Chunk neighbours | Each chunk's metadata records the previous and next chunk ID, so a cited passage can be followed to its neighbours. | `tests/test_phase2.py` |
| Distance threshold | `MIN_RETRIEVAL_DISTANCE` drops hits whose cosine distance is at or beyond a cutoff. Applied **once, after fusion**, so it means the same thing for every hit. | `tests/test_phase2.py` |
| Hybrid search | BM25 keyword search fused with cosine by reciprocal rank fusion (k=60). Falls back to cosine-only if `rank-bm25` is missing or the index fails. | `tests/test_bm25.py` (17 tests) |
| Keyword-hit distance | A hit found only by BM25 had carried a **synthetic distance** derived from its BM25 score. `store.distances` now computes the true cosine distance from the stored vectors, so the threshold and the reported per-hit distances are comparable across hits. | `tests/test_corpus.py`, `tests/test_phase2.py` |
| Corpus | `corpus/manifest.json` is now a document list. `cli fetch`/`cli ingest` handle every document, each verified against its own SHA-256. | `tests/test_corpus.py` (18 tests) |
| Context budget | `sources_from_hits` re-sorts hits by distance before packing. The incoming order is the fused retrieval order, not the distance order, so packing in the incoming order could spend the whole budget on weaker matches and drop the best one. | `tests/test_citations.py` |

## Measured on this machine

- **Corpus:** two documents, 320 chunks. NIST AI RMF 1.0 (48 pages, 168 chunks)
  and NIST AI 600-1 Generative AI Profile (64 pages, 152 chunks), both fetched
  and checksum-verified, then indexed into one vector store.
- **Gold-span recall@5 on the 24 original questions:** 19 of 20 answerable
  questions have their gold span inside a retrieved chunk. Unrestricted and
  RMF-only retrieval score identically, so the second document does not
  displace the first document's evidence.
- **Cross-document set:** 11 new questions, gold spans derived from the
  extracted text by `tools/build_crossdoc_queries.py` rather than typed by hand.
  15 of 16 gold spans are retrieved in the top 5. The one miss (x03) is a real
  retrieval miss and is left in the set as a measured failure.
- **Model:** `qwen2.5:3b-instruct-q4_K_M`, 3.1B parameters, 1.93 GB on disk,
  CPU-only, four threads.

## Findings

- **The 3B model abstains when the answer is buried inside a retrieved chunk.**
  This is the dominant failure mode, and it is not a retrieval failure. For q01
  ("Is AI RMF 1.0 voluntary or mandatory?") the gold text is retrieved and is
  inside the context, but it sits 1296 characters into a 1628-character chunk.
  The same model, given that same chunk with the answer at the start, answers
  correctly. Given the chunk with a 2-character header prepended, it answers;
  given the chunk unchanged, it abstains. Retrieval Hit@5 was 1.0 for this
  question — the evidence was there and the model would not use it.
  This is a "lost in the middle" failure at 3B scale, and it is the reason the
  abstention rate is high. It is not a fixable-by-prompt problem and it is not
  what the earlier 1.5B run measured.
- **The context-budget fix mattered.** Before the re-sort, q01's gold chunk was
  dropped from the context by the greedy packer while the budget was spent on
  weaker matches. After the re-sort the gold chunk is always in the prompt.
  This is a genuine bug fix; it does not by itself fix the abstention above.
- **Per-claim filtering changes what a rejected answer means.** q13 and q14
  each had one good claim and one bad one; under the old all-or-nothing rule
  both were rejected outright. Under the new rule they are answered with the
  bad claim dropped. The 1.5B baseline numbers cannot be compared directly
  against a 3B run on this basis.

## Still not done

1. **The 3B evaluation numbers.** The 24-query run with three warm repeats was
   started but has not completed. The table in `README.md` is still the 1.5B
   pre-hardening baseline and must not be presented as 3B results.
2. **Independent review of the new outputs.** The new 3B and cross-document runs
   need semantic labels before any faithfulness or completeness number is
   published. The existing `evaluation/review.json` labels the 1.5B answers.
3. **The chunk-size question is measured but not yet decided.** `tools/probe_chunk_size.py`
   measures the answer rate at chunk sizes 350/250/200/150 on the same 20
   answerable questions. The gold depth p90 falls from 1296 characters at 350
   tokens to 436 at 150, which is the mechanism the abstention depends on.
   The result has not been read, and no chunk size has been changed on the
   strength of it.
4. **Docker, CI, the Spaces image and `requirements-linux.lock.txt`** are
   still unverified on this machine.
5. **Independent human review and inter-rater agreement** remain outstanding for
   the whole evaluation, and the questions are still assistant-authored.
