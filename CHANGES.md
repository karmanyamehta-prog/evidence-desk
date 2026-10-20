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

- **The 3B model's abstentions are real and mostly not a retrieval problem.**
  The completed 24-question run (3 warm repeats, 72 rows) answered 31 and
  abstained 41. Of the 20 answerable questions, 12 of 20 were answered in at
  least one repeat; 3 (q11, q12, q14) were never answered at any repeat. All
  4 unanswerable questions abstained in all 3 repeats (12/12), so the model is
  not answering questions it cannot answer. The remaining abstentions are
  false abstentions on answerable questions.
- **The decisive measurement: the answer is in the prompt and the model still
  abstains.** `tools/check_gold_in_prompt.py` searches the gold quote for in
  the text actually sent to the model, using the same typographic folding the
  citation validator applies. For **7 of the 20 answerable questions — q03,
  q04, q07, q11, q12, q13 and q16 — the gold text was present in the prompt
  in all 3 repeats and the model still abstained.** The raw output is
  `{"claims": []}`: a model-initiated abstention, not a validation rejection.
  Only 2 of the 20 (q06, q23) had their gold text missing from the prompt,
  and both are misleading-premise questions, where refusing the false premise
  is the correct behaviour.
  So the false abstentions are not a retrieval failure and not a context-budget
  failure. The evidence is in the prompt, and the model chooses to return
  `{"claims": []}`. This is the finding that matters, and it is what the
  earlier "lost in the middle" story got wrong.
- **"Lost in the middle" is NOT the explanation, and the earlier claim of it
  was wrong.** The first investigation found one question (q01) where the gold
  text sat 1296 characters into a 1628-character chunk and the model abstained
  while answering correctly when given the same text at the front. That is a
  real observation about q01, but it does not generalise, and generalising it
  was a mistake. Three separate experiments refute it:
  1. **Answer rate at four chunk sizes.** `tools/probe_chunk_size.py` holds the
     20 answerable questions fixed and varies only the chunk size. The gold
     span's depth p90 falls from 1172 characters at 350 tokens to 429 at 150,
     and the answer rate does not move: 12, 11, 12, 11 of 20.
  2. **Retrieval at the same four sizes.** `tools/probe_chunk_retrieval.py`
     makes no model calls and measures gold-span recall@5 directly: 0.95,
     0.90, 0.90, 0.95. Shallower chunks did not cost retrieval quality, so the
     flat answer rate is not a retrieval/answer trade-off.
  3. **The controlled front-versus-buried test.** `tools/probe_prompt_abstention.py`
     asks the model twice about the same question with the same evidence,
     changing only whether the gold span leads its chunk. In 11 of the 20
     questions the gold span was genuinely moved; in those 11 the answer
     status was **identical both ways, every time**. Overall: 11 answered
     buried, 11 answered fronted, **0 rescued, 0 lost**.
  The counter-example that settles it: q11's gold span is a whole chunk
  starting at offset 0 of its page, so the answer is already the first thing
  in the evidence, and the model abstained on it at all four chunk sizes and
  in all three evaluation repeats. A position effect cannot explain that.
  **No chunk size has been changed on the strength of this**, and the earlier
  "the dominant failure mode" wording has been removed from the docs.
- **The 3B model is not deterministic even at temperature 0 with a fixed
  seed, and the drift is a warm-up effect.** The evaluation uses
  `temperature: 0, seed: 42` for every repeat. Repeat 0 answered 9 of 24
  questions; repeats 1 and 2 each answered 11. Both status flips (q09 and
  q14) are between repeat 0 and the later repeats, in the same direction:
  cold start abstains, warm answers.
  `tools/probe_noise_floor.py` makes this precise: asking the same question
  with the same evidence three times in a row **never** flips (0 of 19
  questions), so the model is stable within a session. But two *independent*
  runs of the same probe disagree on 3 of 19 questions, and on 2 of 19 when
  only the retrieval method differs.
  **Consequence: a difference measured in one run against a number measured
  in another run is not a result.** Every comparison in this file is
  therefore reported as a same-session, alternating, repeated measurement, and
  the chunk-size probe's 11-vs-12 differences are explicitly not evidence.
  A single-run 3B number is not a measurement of this model.
- **The context budget is doing real work, and it drops chunks.** `top_k` is 5
  but only 2-5 chunks reach the prompt (`context_chars` is 4500, and chunks
  with 50-token overlap are long). 33 of 72 rows dropped at least one
  retrieved chunk. The packer sorts by distance first, so the closest match is
  kept; but the overlap means near-duplicate text can consume the budget, and
  `retrieved` and `sources` do not contain the same number of chunks.
- **Per-claim filtering changes what a rejected answer means.** q13 and q14
  each had one good claim and one bad one; under the old all-or-nothing rule
  both were rejected outright. Under the new rule they are answered with the
  bad claim dropped. The 1.5B baseline numbers cannot be compared directly
  against a 3B run on this basis.
- **Several other candidate explanations were measured and ruled out.**
  - *Number of required facts.* The prompt allows "at most 2 short claims",
    and several abstaining questions want 3 or 7 facts. But q02 (4 facts) was
    answered and q07 (3 facts) abstained, so the fact count does not separate
    the two groups.
  - *Prompt size.* Answered rows carry 2272-4147 characters of evidence;
    abstained rows carry 2950-4459. The ranges overlap almost completely.
  - *Quote length.* The prompt asks for 8-20 word quotes. The three
    never-answered questions (q11, q12, q14) have gold quotes of 20, 15 and
    20 words, all inside that range, so the quote rule is not what blocks
    them.
  - *Retrieval method.* The evaluation uses hybrid BM25+RRF; the probes use
    pure cosine. The two disagree on 2 of 19 questions, which is within the
    run-to-run drift measured above, so hybrid vs cosine is not yet shown to
    matter either way.
- **The open question is now narrow and stated precisely: why does the model
  return `{"claims": []}` when the gold text is in the prompt?** The strongest
  remaining hypothesis is that the *distractor* passages in the context are
  what makes it give up, rather than the gold passage itself.
  `tools/probe_context_noise_controlled.py` tests that directly: for each
  question it alternates the two conditions (full retrieved context versus
  the gold chunk alone) within one session, with the service's exact request
  options, three rounds each.

## Still not done

1. **Independent review of the new outputs.** The 3B run has no semantic
   labels, so no faithfulness or completeness number can be published for it.
   The existing `evaluation/review.json` labels the 1.5B answers and cannot be
   reused for the 3B outputs. The `README.md` table must not be overwritten
   with the 3B run until it is labelled.
2. **The cause of the false abstentions.** Position and retrieval are both
   refuted, and the gold text is demonstrably in the prompt. The distractor
   hypothesis is being tested by `tools/probe_context_noise_controlled.py`.
3. **A reproducible 3B measurement protocol.** Because the model drifts
   between runs, any published 3B number needs multiple warm repeats reported
   together with the spread, not a single pass. The evaluation runner already
   supports `--repeats`; the reporting does not yet summarise the spread.
4. **Docker, the Spaces image and a CI run** are still unverified on this
   machine. `requirements-linux.lock.txt` has been regenerated, now resolves
   torch from the CPU index, and is checked by CI.
5. **Independent human review and inter-rater agreement** remain outstanding for
   the whole evaluation, and the questions are still assistant-authored.
