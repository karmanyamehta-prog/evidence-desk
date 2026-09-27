# Document Q&A with Source-Grounded Citations

**Status:** Implementation authorized on 27 September 2026. This design records the original proposal; the hardware amendment below supersedes conflicting defaults. See README for actual validation status.

## 1. Corpus and business framing

| Candidate | Business use | Strength | Main cost/risk |
|---|---|---|---|
| Microsoft FY2023–FY2024 annual reports / 10-Ks | Analyst evidence assistant for business risks and year-specific disclosures | Recognizable business context; tests version confusion | Financial tables and changing reporting categories complicate extraction and ground truth |
| **NIST AI RMF 1.0 (2023) + Generative AI Profile, AI 600-1 (2024)** | **AI governance evidence assistant for product/risk teams** | Defined concepts, numbered actions, complementary documents, traceable evidence | Dense lists/tables; recommendations must not be presented as legal requirements |
| RAG, DPR, and RocketQA papers | Research evidence assistant for an ML team | Tests synthesis and attribution of experimental claims | Two-column PDFs, equations, and incompatible experimental setups make comparison harder |

**Recommendation: the two fixed NIST PDFs**, subject to your approval. The user asks which risks and suggested actions the reports describe and gets inspectable evidence. Your prescription-auditing experience translates directly: a plausible answer is insufficient unless the source supports each claim. These are a deliberately versioned historical corpus, not a promise of current regulatory guidance.

Start with definitions, risks, and numbered actions; support cross-document synthesis. Exclude legal compliance determinations, exhaustive checklist guarantees, scanned-PDF OCR, and visual/chart interpretation in v1. Example question types: “How does the profile describe confabulation?” and “Which reported actions concern evaluating generated output?” These are proposed questions, not yet verified evaluation items.

After approval, inspect extraction against rendered pages, then freeze exact source URLs, publication identifiers, download dates, SHA-256 hashes, and parser versions in a manifest. Keep document downloads out of Git; include a reproducible fetch command and attribution. A parser change creates a new index version.

## 2. Chunking

| Method | Benefit | Tradeoff for these reports |
|---|---|---|
| Fixed token windows | Simple, deterministic baseline | Can separate an action identifier from its explanation |
| **Structure-aware recursive splitting** | Respects headings, paragraphs, and list items before enforcing length | Requires reliable extraction; page boundaries can interrupt a thought |
| Semantic splitting using embedding similarity | Can follow topic shifts without explicit headings | Adds embedding work and threshold tuning; similar neighboring actions can merge incorrectly |

**Propose recursive splitting, not semantic splitting.** They are distinct methods. Parse with pdfplumber, retain page/block provenance, split within page and section boundaries, then prefer paragraph → list item → sentence → token boundaries. Target **350 embedding-token bodies with 50-token overlap**, allowing shorter chunks. Include the heading/action ID in the embedded representation; ensure the entire input, including prefix, heading, and special tokens, fits the model limit. Never silently truncate.

Preserve action IDs and table row headers with their text. Validate table extraction manually; reject or flag unreliable pages instead of silently indexing scrambled columns. Do not join text across pages in v1: exact page attribution is simpler, at the cost of losing some cross-page context. Record neighboring chunk IDs so this limitation can be investigated later.

Benchmark fixed-size versus recursive chunks on development queries at the same token budget. Defer semantic splitting unless this comparison identifies a concrete retrieval problem. LangChain supplies splitters and adapters; application modules retain explicit control of provenance and validation.

## 3. Vector store

| Option | Advantages | Responsibilities/costs |
|---|---|---|
| FAISS | Efficient local vector search; exact flat search is an excellent small-corpus baseline | Application must manage document text, metadata, filtering, IDs, and persistence alongside the index |
| **ChromaDB, local persistent client** | Stores vectors, text, metadata, and IDs; built-in metadata filters and upserts suit file uploads | More dependencies and storage lifecycle complexity; approximate search needs measurement |

**Recommend ChromaDB** because upload, document filtering, and citation lookup are core features. FastAPI alone owns the persistent client; Streamlit calls the API. Serialize ingestion for this single-user app. Use normalized embeddings and cosine distance, initially **top-k = 5**. Filter only by explicit selected documents/versions. Similarity is a ranking signal, not a probability that an answer is correct.

Namespace collections by corpus version, embedding model/revision, and chunk configuration. Stable chunk IDs make repeated ingestion idempotent. Changing embeddings requires reindexing; never mix vector spaces. Hybrid search and reranking remain possible later improvements, not hidden scope additions.

## 4. Embeddings and generation

| Open model, through sentence-transformers | Quality/speed rationale | Caveat |
|---|---|---|
| all-MiniLM-L6-v2 | Fast, small general-purpose baseline; 384-dimensional vectors | Default 256-token sequence limit; needs smaller chunks |
| all-mpnet-base-v2 | Strong general-purpose quality; 768-dimensional vectors | Slower and larger; default 384-token limit |
| **intfloat/e5-small-v2** | Retrieval-oriented, 384-dimensional vectors; 512-token input supports longer evidence | Requires correct `query: ` / `passage: ` prefixes; not proven best on this corpus |

**Recommend E5-small-v2.** It balances CPU use and passage length; this is a design hypothesis, not a benchmark result. Sentence Transformers reports MiniLM as roughly five times faster than MPNet in its general comparison; actual hardware and corpus performance must be measured. Compare all three on identical ≤220-token development chunks to avoid truncation confounds, then tune the selected model's chunk size separately. Pin model revisions as well as package versions.

Propose **Ollama qwen2.5:7b**, quantized, as the default generator; llama3.1:8b is an explicit alternative. Confirm hardware before model setup: roughly 16 GB system RAM is a planning target, not a guarantee, and CPU latency must be measured. Use bounded context/output tokens, low temperature, and a configurable timeout. Smaller models require approval and separate evaluation.

Required path has no paid API dependency. Optional provider adapters are activated explicitly by `LLM_PROVIDER` and environment keys; an Ollama timeout must never silently incur API charges. Source text is untrusted data, never an instruction. Chat history may help interpret a follow-up, but cannot count as supporting evidence.

## 5. Verifiable citations

Persist `document_id`, source URL, PDF hash/version, title, publication ID, **1-based PDF page index**, printed page label when verified, section/action ID, source block IDs, normalized text offsets, bounding boxes when available, and `chunk_id`. Preserve original extracted blocks separately from cleaned text. Unknown headings remain unknown; do not fabricate them.

For each query, assign temporary evidence labels `[S1]` … `[S5]` to retrieved chunks. Request structured claims with evidence IDs and supporting verbatim excerpts. The backend checks schema, requires citations on factual claims, rejects IDs outside retrieved context, and checks quote substrings against stored evidence. It resolves page/section labels itself; the model never supplies authoritative page numbers.

Render claims inline with `[S1]`; expandable references show title/version, section/action, PDF page, printed label if known, and supporting excerpt. A local document endpoint opens the correct PDF with `#page=N`; printed pagination and PDF index are not interchangeable. Multiple-source claims must reference each needed source.

**These checks establish provenance, not entailment.** A real quote can still be misinterpreted. Grounding is audited separately below. Retry invalid structured output once, then abstain with a clear reason. Empty/insufficient evidence produces an abstention plus any useful retrieved excerpts, clearly labeled as retrieved references rather than support for an answer. Do not advertise guaranteed hallucination prevention.

## 6. Evaluation protocol

Build **24 human-verified queries** after freezing the PDFs: 8 definitions/facts, 6 action lookups, 4 cross-section/document synthesis, 4 unanswerable/out-of-scope questions, and 2 misleading-premise questions. Label answerability individually. Each record contains an expected answer, required atomic facts, exact evidence passages with pages/sections, and acceptable alternatives. Gold evidence uses source spans, not unstable chunk IDs.

Stratify into **8 development and 16 held-out test queries** before tuning. No tuning on test results; publish the split and small-sample limitation. For retrieval labeling, inspect the full small corpus for alternative support and manually judge pooled returned chunks. Mere topical similarity is not relevance: a chunk must support an expected fact. Inspect disputed span-to-chunk mappings rather than treating all same-page chunks as relevant.

| Metric | Definition and reporting |
|---|---|
| Precision@5 | Relevant returned chunks / 5 for answerable questions; missing slots count as zero. Also report P@1 and P@3. Remove duplicate chunk IDs before ranking. |
| Evidence recall@5 | Required gold evidence units covered by top five / all required units; overlapping chunks do not earn repeated credit. |
| Faithfulness | Supported atomic generated claims / all factual generated claims, judged against retrieved context. Report macro average and claim counts. |
| Citation support/coverage | Fraction of cited claims actually supported by their cited chunks; fraction of factual claims carrying valid citations. ID validity alone is insufficient. |
| Answer completeness | Required gold facts correctly conveyed / required gold facts. Prevents an always-abstaining or very short system from looking successful. |
| Abstention/errors | Correct abstentions, false abstentions, answered unanswerables, invalid citations, and timeouts, with denominators. No-claim answers have faithfulness N/A, not 1. |
| Latency | Per-query retrieval, generation, and total request-to-final-response wall time; warm p50/p95 and cold startup separately. Exclude offline judging time. |

Use a **custom RAGAS-style claim-support rubric**, with a local model drafting labels and a human reviewing all 24 outputs and their evidence. Store claim text, supported/unsupported/contradicted label, and justification. A second reviewer independently labels at least six outputs if available; otherwise disclose single-reviewer bias. Automated local judging is supplementary and can share the generator's blind spots. Do not call custom scores official RAGAS results. RAGAS context precision is rank-sensitive and is not identical to simple P@5.

Compare fixed versus recursive splitting and embeddings on development only, then lock configuration. Run held-out queries with at least three warm repetitions for latency; preserve each answer and mark the designated quality run. Record hardware, seed where supported, model digest, prompt, corpus/index hashes, versions, and run timestamp. Include failures in reporting, never silently drop them. Phase 3 will produce actual per-query/aggregate Markdown tables and PNG bars for quality scores, with latency on a separate seconds axis/panel. No numerical results exist yet.

## 7. Build boundary and deployment constraints

Proposed flow: Streamlit → FastAPI `/query` → retrieval → Ollama → citation validation → response. An upload endpoint runs PDF validation → extraction → chunking → embeddings → Chroma. Proposed modules: `ingestion/`, `retrieval/`, `generation/`, `citations/`, `api/`, `ui/`, `evaluation/`, and `tests/`.

All paths, model names, provider selection, limits, top-k, and timeouts use validated environment settings with `.env.example`. Phase 2 pins tested package versions and a reproducible dependency lock. Unit tests cover token limits, overlap, page attribution, deterministic IDs, relevance ranking/filtering using controlled vectors, and empty retrieval. Integration tests cover real local embeddings, persistence, citation rejection, malformed/encrypted/empty PDFs, and timeouts. CI unit tests need neither a downloaded LLM nor API keys.

Local Docker Compose will run API/UI with persistent data and an Ollama service or documented host connection. Runtime model download is explicit; model weights and secrets stay outside Git. A live hosted 7B model on a small CPU allocation is likely too slow; even a 1–3B fallback needs a measured quality/latency check. A clearly labeled extractive evidence mode is the safer hosted fallback proposal.

**Hosting conflict:** current Hugging Face documentation says creating Docker/Gradio compute Spaces requires a paid plan, although CPU Basic has no hourly charge. Static Spaces remain free; limited free ZeroGPU Gradio eligibility is a different architecture. Therefore this exact Streamlit/FastAPI Docker app cannot be promised as a universally free hosted Space. Propose preparing the Docker Space configuration as requested, without deploying, plus a $0 static showcase of real saved examples/results and local setup instructions. The showcase is not a live RAG backend. Any different hosting architecture requires approval. Recheck eligibility in Phase 4.

After design approval, commit this document first (`docs: approve grounded QA design`), then separate ingestion/tests, retrieval/tests, generation/citations, API/UI, evaluation, containers, and README commits. Commit evaluation artifacts only after actual runs. No public push or deployment without explicit confirmation. Current limitations include small historical corpus, English text PDFs, single-user operation, imperfect parsing/entailment, and no production monitoring.

## Original approval checkpoint (superseded)

Confirm (or amend): **NIST corpus; recursive 350/50 page-bound chunks; Chroma; E5-small-v2; Qwen 7B subject to hardware; human-reviewed custom grounding evaluation; local full app plus prepared hosting config/static showcase.** Share available RAM, OS, and GPU/VRAM before we finalize model settings. This checkpoint was superseded by the user’s subsequent authorization for autonomous technical decisions and the implementation amendments below.

## Primary references

- [Microsoft 2024 report downloads](https://www.microsoft.com/investor/reports/ar24/download-center/)
- [NIST AI RMF 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10) and [Generative AI Profile PDF](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)
- [RAG paper](https://arxiv.org/abs/2005.11401), [DPR paper](https://arxiv.org/abs/2004.04906), [RocketQA paper](https://arxiv.org/abs/2010.08191)
- [LangChain recursive splitter](https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter)
- [FAISS](https://github.com/facebookresearch/faiss/wiki), [Chroma persistent client](https://cookbook.chromadb.dev/core/clients/), [Chroma filters](https://docs.trychroma.com/docs/querying-collections/metadata-filtering)
- [Sentence Transformers comparison](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html), [E5 model card](https://huggingface.co/intfloat/e5-small-v2)
- [RAGAS faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/), [RAGAS context precision](https://github.com/vibrantlabsai/ragas/blob/main/docs/concepts/metrics/available_metrics/context_precision.md)
- [Hugging Face Spaces requirements](https://huggingface.co/docs/hub/spaces-overview)

## Hardware amendment (27 September 2026)

User authorized autonomous technical decisions. CPU: i5-1240P; usable RAM: 15.7 GB; approximately 101 GB disk free. Iris Xe acceleration is not a dependency. Native Windows is the primary setup (Windows version must pass Ollama requirements). Use qwen2.5:3b-instruct-q4_K_M for everyday use and qwen2.5:7b-instruct-q4_K_M as an explicit quality profile. Limit concurrency to one query, use 4096 context tokens, bounded evidence, and 384 output tokens. These are conservative defaults, not measured laptop performance. Reserve 15–20 GB for a native installation as a planning allowance; Docker may require substantially more. Never publish supplied device/product identifiers.

Ollama lists 3B at 1.9 GB and 7B at 4.7 GB; downloads are not runtime memory measurements. Native Windows requires Windows 10 22H2 or later. Primary sources: https://docs.ollama.com/windows, https://docs.ollama.com/gpu, https://ollama.com/library/qwen2.5/tags, https://docs.ollama.com/api/chat.

Evaluation annotations created by the assistant must be labeled assistant-reviewed; human validation is not implied. RAGAS-style labels require recorded review rather than treating quote membership as semantic faithfulness. If model execution cannot run in the build environment, publish the blocker and runnable evaluator, never fabricated metrics.

## Implementation record superseding the proposal

See README.md and RESEARCH.md for the executed configuration and validation evidence. The benchmark is the one verified GPO-hosted RMF PDF, with GenAI Profile ingestion deferred after download failures. CPU 3B remains the default, 1.5B is an experimental evaluated speed profile, and 7B is optional and unbenchmarked. Citation offsets reference processed page text, raw extraction is retained, and section headings are heuristics. Table subcategory columns are handled explicitly; category descriptions are not included. The evaluator uses assistant-reviewed statement-level judgments and a single exploratory timing pass. No independent human review, sealed holdout claim, Windows execution, Docker execution or public deployment has occurred.
