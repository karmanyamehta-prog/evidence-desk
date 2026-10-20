# Research and implementation record

Checked 27 September 2026. External documentation informs interfaces and limits; measured project results are kept separate from vendor claims.

| Question | Evidence | Decision / uncertainty |
|---|---|---|
| Can local setup avoid a paid API? | [Ollama Windows](https://docs.ollama.com/windows), [chat API](https://docs.ollama.com/api/chat) | Native local inference with explicit model downloads. A paid provider is opt-in only. |
| Can Iris Xe be assumed to accelerate this workload? | [Ollama hardware support](https://docs.ollama.com/gpu) | No. CPU is the tested baseline. Integrated GPU driver/compatibility was not verified on the user's computer. |
| How much disk do models need? | [Ollama Qwen tags](https://ollama.com/library/qwen2.5/tags) | About 1.9 GB for 3B and 4.7 GB for 7B downloads; runtime memory differs. Full-install budget is an explicitly labeled estimate. |
| Are all Qwen sizes identically licensed? | [3B model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE) | No. The 3B model has a non-commercial research/evaluation license; commercial use requires a separate license. This portfolio is an evaluation project. Project MIT licensing does not relicense weights, and no weights are redistributed. |
| What are E5's input rules? | [E5-small-v2 model card](https://huggingface.co/intfloat/e5-small-v2) | Prefix queries/passages, normalize vectors, enforce 512-token model input limit. Pin revision. |
| Is MPNet necessarily better for this corpus? | [Sentence Transformers comparison](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) | General comparisons are not corpus evidence. The development-only comparison here favors MiniLM on span recall; E5 is a frozen baseline rather than a winner claim. |
| Why Chroma? | [Persistent client](https://cookbook.chromadb.dev/core/clients/), [filters](https://docs.trychroma.com/docs/querying-collections/metadata-filtering), [FAISS](https://github.com/facebookresearch/faiss/wiki) | Chroma reduces metadata/filter/persistence plumbing. Only the API owns its client. |
| Does recursive mean semantic splitting? | [LangChain splitter reference](https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter) | No. Recursive splitting uses separators and a budget; embedding-based semantic boundaries are a different method. |
| What is grounding? | [RAGAS faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) | Assess whether response claims follow from retrieved context. Quote membership is only provenance validation. Custom statement-level labels here are disclosed, not presented as official RAGAS results. |
| Is P@5 the same as RAGAS context precision? | [RAGAS context precision](https://github.com/vibrantlabsai/ragas/blob/main/docs/concepts/metrics/available_metrics/context_precision.md) | No. Simple P@5 divides relevant results by five; RAGAS context precision is rank-sensitive. |
| Can this exact app be hosted free on HF? | [Spaces overview](https://huggingface.co/docs/hub/spaces-overview) | Current Docker compute creation requires a paid plan. Static showcases are free but do not run the Python backend. No deployment performed. |

## Corpus evidence

Original NIST [publication page](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10) identifies the report and voluntary scope. Downloaded bytes come from the [GPO archive](https://www.govinfo.gov/app/details/GOVPUB-C13-ef6882c6bd970bc733de394f469ea3f3). The file's SHA-256 and page count are frozen in `corpus/manifest.json`.

The Generative AI Profile is now also corpus. The earlier blocker — the direct NIST endpoint returning HTTP 403 in the build environment — did not reproduce on the target laptop: `nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf` was retrieved successfully, and its SHA-256 (`6e73620a…`, 64 pages) is frozen in `corpus/manifest.json` alongside the RMF. Both documents are indexed into one vector store. The 403 was an environment-specific failure, not a property of the document, and the corpus was not left one document short.

Public browser-indexed text was useful for researching the candidate, but was never reconstructed into a fake PDF or silently used as the ingestion benchmark.

## Observations, not assumptions

- Real dependencies were installed in an isolated Python 3.12 environment. CPU PyTorch, E5 embedding inference, Chroma persistence, PDF parsing, API response validation and PDF serving executed successfully.
- A rendered table on PDF page 28 confirmed that row-wise prose extraction merges left category labels into right action text. The specialized parser extracts only the right column, scoped to the verified file hash. Raw page text is still retained.
- Original 3B sizing responses took 71.4 and 127.0 seconds. Their prompt allowed longer quotations. A preliminary even larger prompt was interrupted before producing an answer; it is not included as a completed query.
- The speed-profile run uses a shorter prompt and 1.5B quantized model. Models and prompts differ, so this is not a controlled size-only speed comparison. Invalid citation/abstention behavior is preserved as failures, not repaired into success.
- The benchmark environment exposes an Intel Xeon Platinum 8573C, an 8-CPU quota, and an 8 GiB memory limit. Generation uses four threads. Other build tasks ran during parts of evaluation; timings are noisy and cannot predict i5 performance.
- All three embedding comparisons used the same 200 passages (220 E5-token target); every model rejected oversize inputs rather than silently truncating. Seven answerable development queries were scored for exact annotated-span coverage. E5/MPNet span recall@5: 0.929; MiniLM: 1.000. All had 1.000 query-level span-hit@5. This is not semantic P@5 and is too small for a broad quality claim.
- Fixed windows (83 chunks) and recursive/action chunks (168 chunks), both targeting 350 tokens with 50 overlap, tied at 0.786 development span recall@5. Action chunks were retained for inspectable action boundaries. This does not prove better retrieval.

## Completed quality run

The final 1.5B run completed all 24 queries: P@1 0.900, P@5 0.310, macro statement faithfulness 0.8125 over 16 factual answers, and mean answer completeness 0.525 over 20 answerable questions. Four answerable questions ended in abstention. Two of four out-of-scope questions ended in structured abstention, both following validation failures. Median request latency was 35.4 seconds and descriptive p95 was 91.5 seconds. The full reviewed record and chart are in `evaluation/results/`. No output was repaired into a success. All six abstentions were validation rejections; the model never abstained on its own.

## Design deviations made explicit

The first DESIGN.md was a proposal. The implementation now uses **two** verified PDFs; generic page headings remain heuristic; printed page labels are not displayed as verified. Table category-column coverage is explicitly excluded. The first full generative benchmark is the experimental 1.5B profile, not the default 3B model — a 3B run with three warm repetitions is in progress and is **not** yet published. Annotation is assistant-reviewed, not human-verified. Windows and the corpus have been exercised on the target laptop; Docker execution, hosted runtime and paid-provider calls have not been tested here.

These limitations are research findings and open validation work. They must not be replaced with guessed results in a portfolio presentation.

## Measured on the target laptop (i5-1240P, 15.7 GB, Windows 11)

- The corpus, the 3B model, the retrieval layer and the API were all run for real. `qwen2.5:3b-instruct-q4_K_M` is 3.1B parameters, 1.93 GB on disk.
- **The 3B model's dominant failure mode is abstention caused by position, not by missing evidence.** For q01 the gold text was retrieved and present in the prompt but sat 1296 characters into a 1628-character chunk; the model returned `{"claims": []}`. Given the same chunk with the answer at the start, it answered correctly. Retrieval Hit@5 for that question was 1.0.
- A genuine bug fell out of the same investigation: the context budget was packed in fused-retrieval order rather than distance order, so a long weak chunk could displace the closest match from the prompt entirely. Fixed by sorting by distance before packing.
- The 3B model *did* return a partially-valid answer on q02: one claim with a bad quote and one with a good quote. Under the old all-or-nothing rule that answer was rejected; under per-claim filtering the valid claim is kept and the failure is recorded. This is the intended behaviour, observed live rather than only unit-tested.
- `tools/probe_chunk_size.py` measures the answer rate against chunk size on the same 20 answerable questions, holding everything else fixed. See `CHANGES.md` for the numbers.
