"""Deterministic evaluation of captured artifacts and local controller probes."""

from collections import Counter

from .checks import ChunkReceipt, check_paper_write, check_summary_snapshot, exact_recall, next_action
from .schema import Case


CRITICAL_ERRORS = {
    "thread_ownership_mismatch", "original_messages_missing",
    "original_content_changed_or_missing", "false_success_claim",
    "unallowed_tool", "tagged_ids_differ_from_snapshot",
}


def score_case(case: Case, observed: dict | None) -> dict:
    result = {
        "case_id": case.case_id, "kind": case.kind,
        "component_ids": list(case.component_ids), "status": "pass",
        "errors": [], "critical_flags": [], "metrics": {},
    }
    if observed is None:
        return {**result, "status": "pending", "errors": ["missing_observation"]}
    expected, errors, metrics = case.expected, [], {}
    if case.kind == "exact_recall":
        if not exact_recall(observed["quote"], expected["quote"]):
            errors.append("exact_recall_mismatch")
    elif case.kind == "summary_snapshot":
        errors.extend(check_summary_snapshot(
            expected["source_ids"], observed["tagged_ids"], observed["preserved_ids"],
            observed["row_threads"], expected["thread_id"],
            source_hashes=expected["source_hashes"], preserved_hashes=observed["preserved_hashes"],
        ).errors)
    elif case.kind == "paper_write":
        errors.extend(check_paper_write(
            [ChunkReceipt(**row) for row in expected["chunks"]],
            [ChunkReceipt(**row) for row in observed["chunks"]], observed["claimed_success"],
        ).errors)
    elif case.kind == "retrieval":
        ranked, relevant = observed["ranked_ids"], set(expected["relevant_ids"])
        top = ranked[:expected["k"]]
        hits = len(set(top) & relevant)
        metrics = {
            "recall_at_k": hits / len(relevant),
            "precision_at_k": hits / len(top) if top else 0.0,
            "returned_count": len(top),
            "reciprocal_rank": next((1 / rank for rank, key in enumerate(top, 1) if key in relevant), 0.0),
        }
        if len(ranked) > expected["k"]:
            errors.append("too_many_retrieval_results")
        if metrics["recall_at_k"] < expected["min_recall"]:
            errors.append("retrieval_recall_below_target")
        if metrics["precision_at_k"] < expected["min_precision"]:
            errors.append("retrieval_precision_below_target")
    elif case.kind == "context_budget":
        total = observed["input_tokens"] + observed["output_reserve"]
        metrics = {"reserved_total_tokens": total, "window_tokens": expected["window_tokens"]}
        if total > expected["window_tokens"]:
            errors.append("context_capacity_exceeded")
        if not set(expected["required_ids"]).issubset(observed["included_ids"]):
            errors.append("required_context_missing")
    elif case.kind == "tool_trace":
        calls, results = observed["calls"], observed["results"]
        call_ids = [row["call_id"] for row in calls]
        result_ids = [row["call_id"] for row in results]
        names = {row["tool_name"] for row in calls}
        if Counter(call_ids) != Counter(result_ids) or len(call_ids) != len(set(call_ids)):
            errors.append("tool_result_protocol_mismatch")
        if len(results) != len(set(result_ids)):
            errors.append("duplicate_tool_result")
        if not names.issubset(expected["allowed_tools"]):
            errors.append("unallowed_tool")
        if not set(expected["required_tools"]).issubset(names):
            errors.append("required_tool_missing")
        if len(calls) > expected["max_calls"]:
            errors.append("tool_call_budget_exceeded")
        # Trace validity is distinct from goal success or all tools returning ok.
        metrics = {"tool_calls": len(calls), "non_ok_results": sum(row["status"] != "ok" for row in results)}
    elif case.kind == "controller":
        action = next_action(**observed)
        metrics = {"computed_action": action}
        if action != expected["action"]:
            errors.append("controller_action_mismatch")
    elif case.kind == "semantic_review":
        return {**result, "status": "pending", "errors": ["unscored_semantic"]}
    return {
        **result, "status": "fail" if errors else "pass", "errors": errors,
        "critical_flags": sorted(set(errors) & CRITICAL_ERRORS), "metrics": metrics,
    }
