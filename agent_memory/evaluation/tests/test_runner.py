"""Pipeline acceptance tests use synthetic observations and temporary reports."""

from copy import deepcopy
import json
import socket
import sys

import pytest

from agent_memory.evaluation.checks import check_summary_snapshot, next_action
from agent_memory.evaluation.runner import ROOT, evaluate, main, write_reports
from agent_memory.evaluation.schema import InputError, load_inputs, load_json


pytestmark = pytest.mark.offline


@pytest.fixture
def inputs():
    return (
        load_json(ROOT / "datasets" / "starter.json"),
        load_json(ROOT / "fixtures" / "observations.good.json"),
    )


def test_starter_reports_pending_semantics_and_partial_coverage(inputs):
    report = evaluate(*inputs)
    assert report["counts"] == {"total": 14, "pass": 13, "fail": 0, "pending": 1}
    assert report["scope_decision"] == "inconclusive"
    assert report["release_ready"] is False
    assert report["coverage"]["uncovered_guideline_components"]
    assert report["critical_failures"] == []
    assert report["evaluator_usage"]["judge_calls"] == 0
    assert report["results"][-1]["errors"] == ["unscored_semantic"]


def test_injected_regressions_override_other_passing_cases(inputs):
    observations = load_json(ROOT / "fixtures" / "observations.regression.json")
    report = evaluate(inputs[0], observations)
    assert report["counts"] == {"total": 14, "pass": 7, "fail": 6, "pending": 1}
    assert report["scope_decision"] == "fail"
    critical = {flag for case in report["critical_failures"] for flag in case["flags"]}
    assert {"thread_ownership_mismatch", "false_success_claim", "unallowed_tool"} <= critical


def test_retrieval_metrics_use_gold_ids_not_model_calls(inputs):
    report = evaluate(*inputs, components=["C08"])
    assert report["results"][0]["metrics"] == {
        "recall_at_k": 1, "precision_at_k": 0.5, "returned_count": 2, "reciprocal_rank": 0.5,
    }
    assert report["scope_decision"] == "pass"


def test_missing_observation_is_pending_and_stays_in_denominator(inputs):
    dataset, observations = inputs
    del observations["records"]["paper-save"]
    report = evaluate(dataset, observations, components=["C07"])
    assert report["counts"] == {"total": 1, "pass": 0, "fail": 0, "pending": 1}
    assert report["fully_scored_fraction"] == 0
    assert report["scope_decision"] == "inconclusive"


def test_identical_case_inputs_have_identical_fingerprints(inputs):
    first, second = evaluate(*inputs), evaluate(*deepcopy(inputs))
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["observations_sha256"] == second["observations_sha256"]
    inputs[1]["records"]["recall-first"]["quote"] = "Changed question"
    assert evaluate(*inputs)["observations_sha256"] != first["observations_sha256"]


@pytest.mark.parametrize("mutation", [
    "duplicate_case", "unknown_case", "wrong_dataset", "invalid_digest",
    "bool_tokens", "negative_tokens", "unknown_kind", "kind_not_string",
    "schema_typo", "duplicate_rank", "empty_gold_quote", "unknown_status",
    "unknown_component", "mixed_paper_versions", "malformed_trace",
])
def test_bad_annotations_and_observations_are_input_errors(inputs, mutation):
    dataset, observations = inputs
    if mutation == "duplicate_case":
        dataset["cases"].append(deepcopy(dataset["cases"][0]))
    elif mutation == "unknown_case":
        observations["records"]["unknown"] = {}
    elif mutation == "wrong_dataset":
        observations["dataset_id"] = "other-dataset"
    elif mutation == "invalid_digest":
        observations["records"]["paper-save"]["chunks"][0]["content_sha256"] = "not-a-hash"
    elif mutation == "bool_tokens":
        observations["records"]["context-boundary"]["input_tokens"] = True
    elif mutation == "negative_tokens":
        observations["records"]["context-boundary"]["output_reserve"] = -1
    elif mutation == "unknown_kind":
        dataset["cases"][0]["kind"] = "invented"
    elif mutation == "kind_not_string":
        dataset["cases"][0]["kind"] = []
    elif mutation == "schema_typo":
        observations["records"]["recall-first"]["qoute"] = "typo"
    elif mutation == "duplicate_rank":
        observations["records"]["retrieval-evidence"]["ranked_ids"] = ["a2", "a2"]
    elif mutation == "empty_gold_quote":
        dataset["cases"][0]["expected"]["quote"] = ""
    elif mutation == "unknown_status":
        observations["records"]["policy-success"]["status"] = []
    elif mutation == "unknown_component":
        dataset["cases"][0]["component_ids"] = ["C99"]
    elif mutation == "mixed_paper_versions":
        dataset["cases"][2]["expected"]["chunks"][1]["version"] = "v1"
    else:
        observations["records"]["tool-protocol"]["results"] = "not-a-list"
    with pytest.raises(InputError):
        load_inputs(dataset, observations)


@pytest.mark.parametrize("payload", ['{"key":1,"key":2}', '{"value":NaN}', '{"value":Infinity}'])
def test_json_loader_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path, payload):
    source = tmp_path / "bad.json"
    source.write_text(payload, encoding="utf-8")
    with pytest.raises(InputError):
        load_json(source)


def test_unchanged_ids_do_not_conceal_modified_original_content():
    result = check_summary_snapshot(
        ["m1"], ["m1"], ["m1"], {"m1": "A"}, "A",
        source_hashes={"m1": "original"}, preserved_hashes={"m1": "changed"},
    )
    assert not result.passed
    assert "original_content_changed_or_missing" in result.errors


def test_second_reflection_is_blocked_even_when_enabled_and_budgeted():
    assert next_action(
        "unknown", ambiguous_strategy=True, adaptive_reflection_enabled=True,
        reflection_budget_available=True, reflection_calls_used=1,
    ) == "actor_uses_validated_observation"


def test_empty_retrieval_cannot_pass_precision_threshold(inputs):
    inputs[1]["records"]["retrieval-evidence"]["ranked_ids"] = []
    result = evaluate(*inputs, components=["C08"])["results"][0]
    assert result["status"] == "fail"
    assert result["metrics"]["precision_at_k"] == 0
    assert result["metrics"]["recall_at_k"] == 0


def test_failed_tool_observation_does_not_break_protocol_by_itself(inputs):
    inputs[1]["records"]["tool-protocol"]["results"][1]["status"] = "permanent_error"
    report = evaluate(*inputs, components=["C18"])
    assert report["scope_decision"] == "pass"
    assert report["results"][0]["metrics"]["non_ok_results"] == 1
    assert report["release_ready"] is False


def test_reports_are_machine_readable_and_not_overwritten(inputs, tmp_path):
    report = evaluate(*inputs)
    write_reports(report, tmp_path)
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["counts"] == report["counts"]
    assert "not a live agent" in (tmp_path / "report.md").read_text(encoding="utf-8")
    with pytest.raises(InputError, match="already exist"):
        write_reports(report, tmp_path)
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == saved


@pytest.mark.parametrize("selection,fixture,exit_code", [
    ([], "observations.good.json", 3),
    (["--component", "C07"], "observations.good.json", 0),
    ([], "observations.regression.json", 1),
])
def test_cli_exit_codes_and_report_creation(tmp_path, selection, fixture, exit_code):
    assert main([
        "--observations", str(ROOT / "fixtures" / fixture),
        "--output", str(tmp_path / "run"), *selection,
    ]) == exit_code
    assert (tmp_path / "run" / "report.json").is_file()


def test_empty_component_selection_is_not_a_vacuous_pass(tmp_path):
    assert main(["--component", "C02", "--output", str(tmp_path / "run")]) == 2
    assert not (tmp_path / "run").exists()


def test_malformed_input_stops_before_creating_report(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert main(["--dataset", str(broken), "--output", str(tmp_path / "run")]) == 2
    assert not (tmp_path / "run").exists()


def test_offline_boundary_rejects_network_and_does_not_import_live_agent():
    with pytest.raises(AssertionError, match="Network access"):
        socket.create_connection(("example.com", 443))
    assert "agent_memory.memory_agent" not in sys.modules
    assert "agent_memory.helper" not in sys.modules


def test_starter_declares_all_supported_checker_kinds(inputs):
    # Semantic review intentionally has no automatic pass until a reviewer exists.
    cases = load_inputs(*inputs)
    assert {case.kind for case in cases} == {
        "exact_recall", "summary_snapshot", "paper_write", "retrieval",
        "context_budget", "tool_trace", "controller", "semantic_review",
    }


def test_controller_probe_failure_is_reported_not_scored_as_an_actor_result(inputs):
    inputs[1]["records"]["policy-success"]["status"] = "permanent_error"
    report = evaluate(*inputs, components=["C20"])
    assert report["scope_decision"] == "fail"
    assert any("controller_action_mismatch" in row["errors"] for row in report["results"])
    assert report["evaluator_usage"]["reflection_calls"] == 0


def test_duplicate_tool_results_are_protocol_failure(inputs):
    results = inputs[1]["records"]["tool-protocol"]["results"]
    results.append(deepcopy(results[0]))
    row = evaluate(*inputs, components=["C18"])["results"][0]
    assert row["status"] == "fail"
    assert "duplicate_tool_result" in row["errors"]


def test_one_token_above_the_inclusive_capacity_boundary_fails(inputs):
    inputs[1]["records"]["context-boundary"]["input_tokens"] += 1
    row = evaluate(*inputs, components=["C17"])["results"][0]
    assert row["errors"] == ["context_capacity_exceeded"]


def test_modified_source_content_is_a_critical_failure_even_with_intact_ids(inputs):
    inputs[1]["records"]["summary-snapshot"]["preserved_hashes"]["m1"] = "e" * 64
    report = evaluate(*inputs, components=["C15"])
    assert report["critical_failures"] == [{
        "case_id": "summary-snapshot", "flags": ["original_content_changed_or_missing"],
    }]


def test_duplicate_persisted_chunk_is_scored_failure_not_a_schema_error(inputs):
    rows = inputs[1]["records"]["paper-save"]["chunks"]
    rows.append(deepcopy(rows[0]))
    report = evaluate(*inputs, components=["C07"])
    assert report["scope_decision"] == "fail"
    assert "false_success_claim" in report["critical_failures"][0]["flags"]


def test_retrieval_return_limit_is_a_separate_contract(inputs):
    inputs[1]["records"]["retrieval-evidence"]["ranked_ids"] = ["a2", "negative-1", "negative-2", "negative-3"]
    row = evaluate(*inputs, components=["C08"])["results"][0]
    assert row["status"] == "fail"
    assert "too_many_retrieval_results" in row["errors"]
