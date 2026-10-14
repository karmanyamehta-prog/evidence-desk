# Evaluation results

Model: `qwen2.5:1.5b-instruct-q4_K_M`. 24 real local queries. One pass; not laptop benchmarks.

Labels are assistant-reviewed, not independent human judgments. Faithfulness is scored only on factual statements; abstentions and heading-only outputs have N/A, and completeness penalizes missed answers. Latency includes cold first-query effects and any retries. Separate p50/p95 are descriptive, not stable warm-load estimates.

| Query | Split | P@5 | Faithfulness | Completeness | Abstained | Seconds |
|---|---|---:|---:|---:|---|---:|
| q01 | dev | 0.400 | N/A | 0.000 | True | 43.76 |
| q02 | test | 0.800 | 1.000 | 1.000 | False | 33.04 |
| q03 | test | 0.600 | N/A | 0.000 | True | 49.37 |
| q04 | dev | 0.600 | 1.000 | 0.500 | False | 36.49 |
| q05 | test | 0.200 | 1.000 | 1.000 | False | 35.88 |
| q06 | test | 0.000 | 1.000 | 0.000 | False | 69.13 |
| q07 | dev | 0.200 | 1.000 | 1.000 | False | 25.75 |
| q08 | test | 0.200 | 1.000 | 1.000 | False | 29.27 |
| q09 | dev | 0.200 | 1.000 | 1.000 | False | 17.27 |
| q10 | test | 0.200 | 1.000 | 0.500 | False | 15.17 |
| q11 | test | 0.200 | N/A | 0.000 | False | 23.78 |
| q12 | test | 0.200 | N/A | 0.000 | False | 22.60 |
| q13 | test | 0.200 | 0.500 | 1.000 | False | 37.51 |
| q14 | dev | 0.200 | 0.500 | 1.000 | False | 29.33 |
| q15 | test | 0.400 | 1.000 | 1.000 | False | 31.51 |
| q16 | test | 0.600 | 0.000 | 0.000 | False | 48.93 |
| q17 | dev | 0.200 | 1.000 | 0.500 | False | 22.34 |
| q18 | test | 0.400 | N/A | 0.000 | True | 66.31 |
| q19 | dev | N/A | N/A | N/A | True | 67.21 |
| q20 | test | N/A | 0.000 | N/A | False | 46.86 |
| q21 | test | N/A | N/A | N/A | True | 124.81 |
| q22 | test | N/A | 1.000 | N/A | False | 34.94 |
| q23 | dev | 0.200 | N/A | 0.000 | True | 95.47 |
| q24 | test | 0.200 | 1.000 | 1.000 | False | 29.81 |

## Aggregate

| Metric | Value |
|---|---:|
| queries | 24 |
| precision_at_1 | 0.900 |
| precision_at_3 | 0.433 |
| precision_at_5 | 0.310 |
| retrieval_scored_queries | 20 |
| hit_at_3 | 0.950 |
| hit_at_5 | 0.950 |
| mrr | 0.925 |
| gold_span_recall_at_5_by_quote | 0.850 |
| answered | 18 |
| model_abstentions | 0 |
| validation_rejections | 6 |
| model_abstentions_on_unanswerable | 0 |
| gold_span_recall_at_5 | 0.850 |
| faithfulness_macro | 0.812 |
| faithfulness_scored_answers | 16 |
| factual_statements | 19 |
| supported_statements | 15 |
| faithfulness_micro | 0.789 |
| completeness | 0.525 |
| correct_abstentions | 2 |
| unanswerable_queries | 4 |
| false_abstentions | 4 |
| answerable_queries | 20 |
| errors | 0 |
| latency_p50_s | 35.411 |
| latency_p95_s | 91.518 |
| reviewer | assistant; independent human validation pending |
| quality_protocol | exploratory, single pass; dev/test assignments retained but not a sealed holdout |

![Measured scores](metrics.png)

All raw answers, citations, and timings are retained in the run JSON; review.json supplies semantic labels and rationales. P@5 uses judged relevance, not same-page matching or citation validity. Alternate relevant passages count even when not in the selected gold spans. The errors count covers requests that raised exceptions; validation-rejection abstentions still count as failures in answer metrics.
