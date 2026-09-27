# Evaluation protocol and review guide

This is a small exploratory study. All question authoring and semantic review in
this repository was performed by the building assistant. No independent human
review or inter-rater agreement is claimed. Dev/test assignments are recorded,
but inspection and debugging during development mean this is not a sealed test.

## Files and provenance

- `queries.json`: 24 questions, expected facts and exact gold evidence spans for
  the current parser. `queries-v2.json` preserves the earlier parser's spans.
- `retrieval-pool.json`: the current top-five chunks used for relevance review.
- `qrels.json`: relevance labels with rank lists and rationales. A chunk is
  relevant when it supports at least one expected answer fact. Partial coverage
  counts; topical similarity or being on the same page does not.
- `runs/cpu-1.5b.json`: completed current-model evaluation, including source text,
  model settings, hashes, outcomes, failures and timings.
- `runs/baseline-v2-1.5b.json`: interrupted, superseded exploratory run. Do not mix
  its rows into the current report.
- `runs/sizing-3b.json`: two early sizing requests, with a different prompt. This
  does not establish 3B quality or a controlled model-size latency comparison.
- `review.json`: per-output semantic judgments and reasons. Labels apply only to
  the matching recorded answers and retrieval pool.

## Review procedure

1. Check the source PDF and gold excerpt. Correct any authoring mistake in a
   separately versioned set; do not quietly change labels to fit an answer.
2. Judge each retrieved chunk against the required facts. Record alternate valid
   evidence even if it differs from the selected gold span.
3. Judge each emitted factual statement against its cited source chunk(s).
   Every part, including action identifiers and qualifiers, must be entailed.
   Use `supported`, `unsupported`, or `contradicted`. A heading without a factual
   proposition is `non_factual` and excluded from the faithfulness denominator.
4. Count required facts correctly conveyed by answer text. Citation quotations
   do not substitute for an absent answer. A required fact containing several
   components earns credit only if all those components are conveyed. Record the
   reason for partial credit. Irrelevant but supported answers can score high on
   faithfulness and zero on completeness.
5. Record abstentions on answerable and unanswerable questions separately.
   Faithfulness is N/A for abstentions and heading-only outputs, not 1.0.

These judgments are subjective and should be reviewed independently. For a
stronger portfolio claim, have two people label outputs without seeing each
other's decisions, retain original labels, measure agreement and adjudicate
disagreements. Create a fresh untouched test set before further prompt tuning.

## Reproduction

From the repository root, with Ollama running and the corpus indexed:

```sh
python -m grounded_qa.evaluation.run --out evaluation/runs/new-run.json
# Review new-run.json and save matching labels before generating its report.
python -m grounded_qa.evaluation.report --run evaluation/runs/new-run.json --labels evaluation/new-review.json
python scripts/build_showcase.py
```

The checked-in static showcase deliberately reads the checked-in 1.5B run. To
present a different run, update that explicit input and re-review the examples.

The report summarizes repetition zero only. The runner's `--repeats` option
retains additional timing observations for a future warm-latency study; the
current report is not a repeated-load benchmark. Dependency initialization is
recorded separately; request latency includes generation, validation and retries.
Scores are custom RAGAS-style calculations, not official RAGAS judge outputs.
