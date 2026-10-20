"""Tests for the context-noise analysis tools.

These cover the statistics that the noise conclusion rests on. The exact
binomial test is implemented in the tool rather than in the library, so it
has to be tested here; a wrong p-value is exactly the kind of bug that
would have shipped a wrong conclusion.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    """Import a tool script by path; the tools are not a package."""
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load("analyze_noise_effect")
claims_effect = _load("analyze_claims_effect")


# ---------------------------------------------------------------------------
# exact_binomial_two_sided
# ---------------------------------------------------------------------------


def test_exact_binomial_symmetric_case():
    # 5 vs 5 -> p = 1.0, nothing is significant.
    assert analysis.exact_binomial_two_sided(5, 5) == pytest.approx(1.0)


def test_exact_binomial_extreme_split():
    # 0 vs 10: P(X<=0) under p=0.5 is 1/1024, doubled.
    assert analysis.exact_binomial_two_sided(0, 10) == pytest.approx(2 / 1024)


def test_exact_binomial_known_value():
    # 4 vs 15 is the discordant count the noise probe produced.
    # P(X<=4) with n=19 is 0.009605; doubled gives 0.01921.
    assert analysis.exact_binomial_two_sided(4, 15) == pytest.approx(0.01921, abs=1e-5)


def test_exact_binomial_is_symmetric_in_its_arguments():
    assert analysis.exact_binomial_two_sided(3, 7) == pytest.approx(
        analysis.exact_binomial_two_sided(7, 3)
    )


def test_exact_binomial_no_discordant_pairs():
    assert analysis.exact_binomial_two_sided(0, 0) == 1.0


def test_exact_binomial_never_exceeds_one():
    for a in range(0, 10):
        for b in range(0, 10):
            assert analysis.exact_binomial_two_sided(a, b) <= 1.0


# ---------------------------------------------------------------------------
# The analysis of the checked-in probe output
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report():
    return json.loads(
        (TOOLS / "context_noise_analysis.json").read_text(encoding="utf-8")
    )


def test_budget_confound_question_is_excluded(report):
    assert report["excluded_because_gold_span_not_in_retrieved_prompt"] == ["q06"]


def test_round_level_p_value_matches_the_raw_probe(report):
    """The all-questions round-level figure reproduces the probe's own count."""
    assert report["round_level_all_questions"]["discordant_gold_only_only"] == 15
    assert report["round_level_all_questions"]["discordant_retrieved_only"] == 4
    assert report["round_level_all_questions"][
        "exact_binomial_two_sided_p"
    ] == pytest.approx(0.01921, abs=1e-5)


def test_excluding_the_confound_weakens_the_effect(report):
    """The confounded question was counted in the flattering direction."""
    raw = report["round_level_all_questions"]["exact_binomial_two_sided_p"]
    clean = report["round_level_excluding_budget_confound"][
        "exact_binomial_two_sided_p"
    ]
    assert clean > raw


def test_question_level_effect_is_not_significant(report):
    """At the unit the design samples, the distractors are not shown to matter."""
    clean = report["question_level_excluding_budget_confound"]
    assert clean["exact_binomial_two_sided_p"] > 0.05


def test_effect_is_not_one_directional(report):
    """A real distractor effect would rescue every question it touches."""
    clean = report["question_level_excluding_budget_confound"]
    assert clean["rescued_by_gold_only"]
    assert clean["lost_by_gold_only"], (
        "expected at least one question answered under the full context and "
        "abstained on the gold chunk alone"
    )


def test_q06_is_not_counted_as_a_distractor_rescue(report):
    clean = report["question_level_excluding_budget_confound"]
    assert "q06" not in clean["rescued_by_gold_only"]


# ---------------------------------------------------------------------------
# The gold-presence check
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def presence():
    return json.loads(
        (TOOLS / "context_noise_gold_presence.json").read_text(encoding="utf-8")
    )


def test_gold_presence_reports_one_budget_drop(presence):
    assert presence["summary"]["gold_span_missing_because_the_budget_dropped_it"] == [
        "q06"
    ]


def test_q06_gold_chunk_was_retrieved_but_not_packed(presence):
    row = next(r for r in presence["rows"] if r["id"] == "q06")
    assert row["gold_chunk_in_top_k"] is True
    assert row["gold_span_in_retrieved_prompt"] is False


def test_q23_gold_chunk_was_never_retrieved(presence):
    row = next(r for r in presence["rows"] if r["id"] == "q23")
    assert row["gold_chunk_in_top_k"] is False
    assert row["gold_chunk_rank"] is None


# ---------------------------------------------------------------------------
# The context-packing analysis
# ---------------------------------------------------------------------------

packing = _load("analyze_context_packing")


def test_greedy_pack_takes_chunks_in_rank_order_while_they_fit():
    assert packing.greedy_pack([10, 10, 10], 25) == [0, 1]
    assert packing.greedy_pack([10, 10, 10], 30) == [0, 1, 2]


def test_greedy_pack_skips_a_chunk_that_does_not_fit():
    assert packing.greedy_pack([100, 10], 50) == [1]


def test_knapsack_pack_respects_the_budget():
    chosen = packing.knapsack_pack([10, 10, 10], [1.0, 0.9, 0.8], 25)
    assert sum([10, 10, 10][i] for i in chosen) <= 25


def test_knapsack_pack_prefers_the_more_similar_chunk():
    # Chunk 0 is more similar, so with room for only one, chunk 0 wins.
    chosen = packing.knapsack_pack([10, 10], [1.0, 0.5], 10)
    assert chosen == [0]


def test_knapsack_pack_finds_the_better_combination():
    # Chunk 1 is the most valuable but too long to sit alongside chunk 0 in a
    # 20-character budget, so the knapsack takes chunks 0 and 2 (1.9) rather
    # than chunk 1 alone (1.2).
    assert packing.knapsack_pack([10, 15, 10], [1.0, 1.2, 0.9], 20) == [0, 2]


def test_knapsack_pack_prefers_real_similarity_over_rank():
    """A rank proxy would make the first chunk dominate; similarity does not.

    With rank weighting (1/(i+1)) the knapsack always reproduces the greedy
    answer, which would make the comparison with the shipped packer
    vacuous. With real similarity values it can pick a different set.
    """
    # Chunk 0 is long and slightly less similar; chunks 1 and 2 are short
    # and together are both closer and cheaper.
    chosen = packing.knapsack_pack([10, 4, 4], [0.90, 0.88, 0.86], 12)
    assert chosen == [1, 2]


def test_best_subset_with_gold_finds_the_cheapest_way_in():
    lengths = [10, 9, 10]
    values = [1.0, 0.9, 0.8]
    best = packing.best_subset_with_gold(lengths, values, 20, 2)
    assert best is not None
    assert 2 in best[1]
    assert sum(lengths[i] for i in best[1]) <= 20


def test_best_subset_with_gold_returns_none_when_infeasible():
    assert packing.best_subset_with_gold([100, 100], [1.0, 1.0], 50, 0) is None


@pytest.fixture(scope="module")
def packing_report():
    return json.loads(
        (TOOLS / "context_packing_analysis.json").read_text(encoding="utf-8")
    )


def test_knapsack_and_greedy_agree_on_every_question(packing_report):
    """The conclusion that the packer is not the culprit depends on this."""
    assert packing_report["summary"]["packings_that_differ"] == []


def test_knapsack_gains_and_loses_no_gold_span(packing_report):
    summary = packing_report["summary"]
    assert summary["questions_where_the_knapsack_gains_the_gold_span"] == []
    assert summary["questions_where_the_knapsack_loses_the_gold_span"] == []


def test_gold_span_reaches_the_prompt_for_18_of_20(packing_report):
    summary = packing_report["summary"]
    assert summary["gold_in_prompt_under_greedy"] == 18
    assert summary["gold_in_prompt_under_knapsack"] == 18


def test_q06_gold_chunk_was_within_reach_at_a_slightly_higher_budget(
    packing_report,
):
    """The gold chunk was reachable; the budget was the only thing stopping it.

    This is the evidence that q06 is a budget-boundary case rather than an
    unreachable one, and it is why raising `context_chars` to 4809 would
    recover it.
    """
    row = next(r for r in packing_report["rows"] if r["id"] == "q06")
    assert row["gold_chunk_rank"] == 5
    assert row["greedy_gold_in_prompt"] is False
    # A subset that includes the gold chunk exists and is feasible.
    assert row["best_with_gold_chunks"] is not None
    assert 4 in row["best_with_gold_chunks"]
    # And it costs almost nothing in similarity.
    assert row["similarity_given_up_to_include_gold"] < 0.01


def test_q06_rank_one_and_gold_are_within_a_tie(packing_report):
    """The gold chunk is not a weak match; the ranking simply carries no signal.

    This is what rules out the "a long weak chunk crowded out a short
    relevant one" explanation.
    """
    row = next(r for r in packing_report["rows"] if r["id"] == "q06")
    gap = row["distances"][4] - row["distances"][0]
    assert gap < 0.005


# ---------------------------------------------------------------------------
# The claims-quality analysis
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def claims_report():
    return json.loads(
        (TOOLS / "context_noise_claims_analysis.json").read_text(encoding="utf-8")
    )


def test_q06_is_excluded_from_the_rescued_set(claims_report):
    """q06 is rescued by the absence of the answer, not by the distractors."""
    assert claims_report["excluded_because_gold_span_not_in_retrieved_prompt"] == [
        "q06"
    ]
    assert "q06" not in claims_report["rescued_by_gold_only"]


def test_three_questions_are_genuinely_rescued(claims_report):
    assert claims_report["rescued_by_gold_only"] == ["q12", "q14", "q16"]


def test_rescued_answers_are_grounded(claims_report):
    """The distractors suppress usable answers, not wrong ones."""
    quality = claims_report["rescued_answer_quality"]
    assert quality["questions"] == 3
    assert quality["grounded"] == 3


def test_not_every_rescued_answer_is_complete(claims_report):
    """The extra answers are mostly good partial answers, not complete ones."""
    quality = claims_report["rescued_answer_quality"]
    assert quality["complete"] < quality["questions"]
    assert quality["grounded_and_complete"] == 1


def test_rescue_effect_is_not_significant(claims_report):
    assert claims_report["question_level_exact_binomial_two_sided_p"] > 0.05


@pytest.fixture(scope="module")
def raw_claims():
    return json.loads((TOOLS / "context_noise_claims.json").read_text(encoding="utf-8"))


def test_gold_only_answers_more_often(raw_claims):
    summary = raw_claims["summary"]
    assert summary["gold_only"]["answers"] > summary["retrieved"]["answers"]


def test_gold_only_answers_are_more_often_grounded(raw_claims):
    summary = raw_claims["summary"]
    retrieved = summary["retrieved"]
    gold = summary["gold_only"]
    retrieved_rate = retrieved["answers_with_grounded_quotes"] / retrieved["answers"]
    gold_rate = gold["answers_with_grounded_quotes"] / gold["answers"]
    assert gold_rate > retrieved_rate


def test_fact_recall_is_not_improved_by_the_gold_only_condition(raw_claims):
    """More answers, but not more complete ones. This is why the change is
    not simply applied."""
    summary = raw_claims["summary"]
    assert (
        summary["gold_only"]["answers_recalling_every_required_fact"]
        <= summary["retrieved"]["answers_recalling_every_required_fact"]
    )
