"""Strict, dependency-free input contracts for artifact evaluation."""

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import re


KINDS = {
    "exact_recall", "summary_snapshot", "paper_write", "retrieval",
    "context_budget", "tool_trace", "controller", "semantic_review",
}
STATUSES = {
    "ok", "invalid_args", "transient_error", "permanent_error",
    "partial_write", "empty", "not_found", "budget_exceeded", "unknown",
}


class InputError(ValueError):
    """Invalid evaluation inputs; never a passing or failing agent result."""


@dataclass(frozen=True)
class Case:
    case_id: str
    component_ids: tuple[str, ...]
    kind: str
    expected: dict
    description: str


def fields(value, required, optional=(), *, where="record"):
    if not isinstance(value, dict):
        raise InputError(f"{where}: expected an object")
    missing, extra = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or extra:
        raise InputError(f"{where}: missing={sorted(missing)}, unknown={sorted(extra)}")


def text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{where}: expected a nonempty string")


def boolean(value, where):
    if type(value) is not bool:
        raise InputError(f"{where}: expected a boolean")


def number(value, where, *, integer=False, low=0, high=None):
    if type(value) not in (int, float) or (integer and type(value) is not int):
        raise InputError(f"{where}: expected {'integer' if integer else 'number'}")
    if (type(value) is float and not isfinite(value)) or value < low or (high is not None and value > high):
        raise InputError(f"{where}: value outside allowed range")


def string_list(value, where, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise InputError(f"{where}: expected {'nonempty ' if nonempty else ''}list")
    for item in value:
        text(item, where)
    if len(value) != len(set(value)):
        raise InputError(f"{where}: duplicate identifiers")


def hash_map(value, where):
    if not isinstance(value, dict):
        raise InputError(f"{where}: expected ID-to-SHA256 map")
    for key, digest in value.items():
        text(key, where)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise InputError(f"{where}: invalid lowercase SHA256 digest")


def chunk_rows(value, where, *, nonempty=False, unique=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise InputError(f"{where}: expected chunk manifest")
    identities = []
    for row in value:
        fields(row, {"source_id", "version", "chunk_id", "content_sha256"}, where=where)
        for key in ("source_id", "version", "chunk_id"):
            text(row[key], where)
        hash_map({row["chunk_id"]: row["content_sha256"]}, where)
        identities.append((row["source_id"], row["version"], row["chunk_id"]))
    if unique and len(identities) != len(set(identities)):
        raise InputError(f"{where}: duplicate gold chunk identity")
    if unique and len({key[:2] for key in identities}) != 1:
        raise InputError(f"{where}: gold manifest must identify exactly one paper version")


def validate_expected(kind, expected):
    where = f"{kind}.expected"
    if kind == "exact_recall":
        fields(expected, {"quote"}, where=where)
        text(expected["quote"], where)
    elif kind == "summary_snapshot":
        fields(expected, {"source_ids", "thread_id", "source_hashes"}, where=where)
        string_list(expected["source_ids"], where, nonempty=True)
        text(expected["thread_id"], where)
        hash_map(expected["source_hashes"], where)
        if set(expected["source_ids"]) != set(expected["source_hashes"]):
            raise InputError(f"{where}: source IDs and hashes must match")
    elif kind == "paper_write":
        fields(expected, {"chunks"}, where=where)
        chunk_rows(expected["chunks"], where, nonempty=True, unique=True)
    elif kind == "retrieval":
        fields(expected, {"relevant_ids", "k", "min_recall", "min_precision"}, where=where)
        string_list(expected["relevant_ids"], where, nonempty=True)
        number(expected["k"], where, integer=True, low=1)
        for key in ("min_recall", "min_precision"):
            number(expected[key], where, high=1)
    elif kind == "context_budget":
        fields(expected, {"window_tokens", "required_ids"}, where=where)
        number(expected["window_tokens"], where, integer=True, low=1)
        string_list(expected["required_ids"], where)
    elif kind == "tool_trace":
        fields(expected, {"allowed_tools", "required_tools", "max_calls"}, where=where)
        string_list(expected["allowed_tools"], where)
        string_list(expected["required_tools"], where)
        number(expected["max_calls"], where, integer=True)
        if not set(expected["required_tools"]).issubset(expected["allowed_tools"]):
            raise InputError(f"{where}: required tools are outside allowlist")
    elif kind == "controller":
        fields(expected, {"action"}, where=where)
        if not isinstance(expected["action"], str) or expected["action"] not in {
            "deny_scope", "continue", "repair_arguments", "report_blocker",
            "retry_transient", "change_strategy_or_report", "reconcile_state",
            "retrieve_alternative_or_abstain", "reflect_once", "actor_uses_validated_observation",
        }:
            raise InputError(f"{where}: unknown action")
    else:
        fields(expected, {"required_facts"}, where=where)
        string_list(expected["required_facts"], where, nonempty=True)


def validate_observed(kind, observed):
    where = f"{kind}.observed"
    if kind == "exact_recall":
        fields(observed, {"quote"}, where=where)
        if not isinstance(observed["quote"], str):
            raise InputError(f"{where}: quote must be a string")
    elif kind == "summary_snapshot":
        fields(observed, {"tagged_ids", "preserved_ids", "row_threads", "preserved_hashes"}, where=where)
        # Repeated observed IDs are scored as corruption, not silently deduplicated.
        for key in ("tagged_ids", "preserved_ids"):
            if not isinstance(observed[key], list):
                raise InputError(f"{where}: {key} must be a list")
            for item in observed[key]:
                text(item, where)
        if not isinstance(observed["row_threads"], dict):
            raise InputError(f"{where}: row_threads must be an object")
        for key, thread in observed["row_threads"].items():
            text(key, where)
            text(thread, where)
        hash_map(observed["preserved_hashes"], where)
    elif kind == "paper_write":
        fields(observed, {"chunks", "claimed_success"}, where=where)
        chunk_rows(observed["chunks"], where)
        boolean(observed["claimed_success"], where)
    elif kind == "retrieval":
        fields(observed, {"ranked_ids"}, where=where)
        string_list(observed["ranked_ids"], where)
    elif kind == "context_budget":
        fields(observed, {"input_tokens", "output_reserve", "included_ids"}, where=where)
        for key in ("input_tokens", "output_reserve"):
            number(observed[key], where, integer=True)
        string_list(observed["included_ids"], where)
    elif kind == "tool_trace":
        fields(observed, {"calls", "results"}, where=where)
        for key in ("calls", "results"):
            if not isinstance(observed[key], list):
                raise InputError(f"{where}: {key} must be a list")
            for row in observed[key]:
                fields(row, {"call_id", "tool_name"} if key == "calls" else {"call_id", "status"}, where=where)
                text(row["call_id"], where)
                if key == "calls":
                    text(row["tool_name"], where)
                elif not isinstance(row["status"], str) or row["status"] not in STATUSES:
                    raise InputError(f"{where}: unknown result status")
    elif kind == "controller":
        fields(observed, {"status"}, {
            "attempts_used", "attempt_cap", "scope_allowed", "ambiguous_strategy",
            "adaptive_reflection_enabled", "reflection_budget_available", "reflection_calls_used",
        }, where=where)
        if not isinstance(observed["status"], str) or observed["status"] not in STATUSES:
            raise InputError(f"{where}: unknown status")
        for key in ("attempts_used", "attempt_cap", "reflection_calls_used"):
            if key in observed:
                number(observed[key], where, integer=True, low=1 if key == "attempt_cap" else 0)
        for key in ("scope_allowed", "ambiguous_strategy", "adaptive_reflection_enabled", "reflection_budget_available"):
            if key in observed:
                boolean(observed[key], where)
    else:
        fields(observed, {"answer", "evidence"}, where=where)
        text(observed["answer"], where)
        string_list(observed["evidence"], where, nonempty=True)


def load_json(path: Path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InputError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise InputError(f"Non-finite JSON number: {value}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys, parse_constant=nonfinite)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"Cannot load {path.name}: {exc}") from exc


def load_inputs(dataset: dict, observations: dict) -> list[Case]:
    fields(dataset, {"schema_version", "dataset_id", "cases"}, where="dataset")
    fields(observations, {"schema_version", "dataset_id", "provenance", "records"}, where="observations")
    for record in (dataset, observations):
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise InputError("Unsupported schema_version")
        text(record["dataset_id"], "dataset_id")
    if dataset["dataset_id"] != observations["dataset_id"]:
        raise InputError("Dataset IDs do not match")
    if not isinstance(observations["provenance"], str) or observations["provenance"] not in {"synthetic_fixture", "captured_artifacts"}:
        raise InputError("Unknown observation provenance")
    if not isinstance(dataset["cases"], list) or not dataset["cases"]:
        raise InputError("Dataset cases must be a nonempty list")
    if not isinstance(observations["records"], dict):
        raise InputError("Observation records must be an object")
    cases, seen = [], set()
    for record in dataset["cases"]:
        fields(record, {"case_id", "component_ids", "kind", "expected", "description"}, where="case")
        text(record["case_id"], "case_id")
        if record["case_id"] in seen:
            raise InputError(f"Duplicate case_id: {record['case_id']}")
        seen.add(record["case_id"])
        string_list(record["component_ids"], "component_ids", nonempty=True)
        if any(not re.fullmatch(r"C(?:0[1-9]|1[0-9]|20)", key) for key in record["component_ids"]):
            raise InputError("Unknown component ID")
        if not isinstance(record["kind"], str) or record["kind"] not in KINDS:
            raise InputError(f"Unknown checker kind: {record['kind']}")
        text(record["description"], "description")
        validate_expected(record["kind"], record["expected"])
        if record["case_id"] in observations["records"]:
            validate_observed(record["kind"], observations["records"][record["case_id"]])
        cases.append(Case(record["case_id"], tuple(record["component_ids"]), record["kind"], record["expected"], record["description"]))
    unknown = observations["records"].keys() - seen
    if unknown:
        raise InputError(f"Observations reference unknown cases: {sorted(unknown)}")
    return cases
