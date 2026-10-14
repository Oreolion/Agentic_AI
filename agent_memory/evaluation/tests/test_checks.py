"""Test the evaluator, not the live agent. All observations are synthetic fixtures."""

from dataclasses import replace

import pytest

from agent_memory.evaluation.checks import (
    ChunkReceipt,
    check_paper_write,
    check_summary_snapshot,
    exact_recall,
    judge_route,
    next_action,
    rubric_score,
)


pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "actual,expected,passed",
    [
        ("Can you get me the paper MemGPT", "Can you get me the paper MemGPT", True),
        ("Can you get me\n the paper MemGPT", "Can you get me the paper MemGPT", True),
        ("Use Oracle", "Do not use Oracle", False),
        ("M7-Q2", "M7-Q3", False),
        ("The first question was: MemGPT", "MemGPT", False),
    ],
)
def test_exact_recall_does_not_reward_keyword_overlap(actual, expected, passed):
    assert exact_recall(actual, expected) is passed


def test_summary_snapshot_keeps_a_concurrent_message_outside_the_tagged_set():
    result = check_summary_snapshot(
        ["m1", "m2"], ["m1", "m2"], ["m1", "m2", "m3"],
        {"m1": "A", "m2": "A", "m3": "A"}, "A",
    )
    assert result.passed


@pytest.mark.parametrize(
    "tagged,preserved,threads,error",
    [
        (["m1", "m2", "m3"], ["m1", "m2", "m3"], {"m1": "A", "m2": "A", "m3": "A"}, "tagged_ids_differ_from_snapshot"),
        (["m1", "m2"], ["m1"], {"m1": "A", "m2": "A"}, "original_messages_missing"),
        (["m1", "m2"], ["m1", "m2"], {"m1": "A", "m2": "B"}, "thread_ownership_mismatch"),
        (["m1", "m2"], ["m1", "m2"], {"m1": "A"}, "thread_ownership_mismatch"),
    ],
)
def test_summary_corruption_and_missing_ownership_fail(tagged, preserved, threads, error):
    result = check_summary_snapshot(["m1", "m2"], tagged, preserved, threads, "A")
    assert not result.passed
    assert error in result.errors


def test_a_complete_paper_manifest_verifies_success():
    rows = [ChunkReceipt("paper-A", "v2", "0", "hash-a"), ChunkReceipt("paper-A", "v2", "1", "hash-b")]
    assert check_paper_write(rows, list(reversed(rows)), True).passed


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_version", "changed_content"])
def test_save_claims_require_the_exact_persisted_paper(mutation):
    rows = [ChunkReceipt("paper-A", "v2", "0", "hash-a"), ChunkReceipt("paper-A", "v2", "1", "hash-b")]
    persisted = list(rows)
    if mutation == "missing":
        persisted.pop()
    elif mutation == "duplicate":
        persisted.append(rows[0])
    elif mutation == "wrong_version":
        persisted[0] = replace(rows[0], version="v1")
    else:
        persisted[0] = replace(rows[0], content_sha256="different-hash")
    result = check_paper_write(rows, persisted, True)
    assert not result.passed
    assert "false_success_claim" in result.errors


def test_honest_failure_is_still_a_failed_write_without_false_success():
    expected = [ChunkReceipt("paper-A", "v2", "0", "hash-a")]
    result = check_paper_write(expected, [], False)
    assert not result.passed
    assert "false_success_claim" not in result.errors


def test_summary_rubric_matches_the_guideline_example():
    ratings = dict(zip("abcde", [4, 3, 0, 3, 4]))
    weights = dict(zip("abcde", [30, 30, 20, 10, 10]))
    assert rubric_score(ratings, weights) == 70


@pytest.mark.parametrize(
    "ratings,weights",
    [
        ({"a": 4}, {"a": 50}),
        ({"a": 4}, {"a": 50, "b": 50}),
        ({"a": 5}, {"a": 100}),
        ({"a": float("nan")}, {"a": 100}),
        ({"a": 4, "b": 4}, {"a": -10, "b": 110}),
    ],
)
def test_invalid_rubrics_are_errors_not_silently_adjusted(ratings, weights):
    with pytest.raises(ValueError):
        rubric_score(ratings, weights)


@pytest.mark.parametrize(
    "hard,semantic,selected,budget,expected",
    [
        (False, True, True, True, "fail_without_judge"),
        (True, False, False, False, "deterministic_pass"),
        (True, True, False, True, "unscored_semantic"),
        (True, True, True, False, "blocked_judge_budget"),
        (True, True, True, True, "semantic_review"),
    ],
)
def test_judge_spend_is_gated_without_inventing_semantic_passes(hard, semantic, selected, budget, expected):
    assert judge_route(
        hard_checks_passed=hard,
        semantic_required=semantic,
        selected_for_review=selected,
        judge_budget_available=budget,
    ) == expected


@pytest.mark.parametrize(
    "status,attempts,expected",
    [
        ("ok", 1, "continue"),
        ("invalid_args", 1, "repair_arguments"),
        ("invalid_args", 2, "report_blocker"),
        ("transient_error", 1, "retry_transient"),
        ("transient_error", 2, "report_blocker"),
        ("permanent_error", 1, "change_strategy_or_report"),
        ("partial_write", 1, "reconcile_state"),
        ("empty", 1, "retrieve_alternative_or_abstain"),
        ("budget_exceeded", 1, "report_blocker"),
    ],
)
def test_known_observations_do_not_need_a_reflector(status, attempts, expected):
    assert next_action(
        status,
        attempts_used=attempts,
        adaptive_reflection_enabled=True,
        reflection_budget_available=True,
    ) == expected


def test_scope_denial_overrides_even_a_success_observation():
    assert next_action("ok", scope_allowed=False) == "deny_scope"


@pytest.mark.parametrize("enabled,budget", [(False, True), (True, False), (False, False)])
def test_unresolved_strategy_cannot_trigger_unbudgeted_reflection(enabled, budget):
    assert next_action(
        "unknown", ambiguous_strategy=True,
        adaptive_reflection_enabled=enabled,
        reflection_budget_available=budget,
    ) == "actor_uses_validated_observation"


def test_only_an_enabled_budgeted_unresolved_strategy_requests_reflection():
    assert next_action(
        "unknown", ambiguous_strategy=True,
        adaptive_reflection_enabled=True,
        reflection_budget_available=True,
    ) == "reflect_once"
