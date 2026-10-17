"""Pure checks for observed artifacts, without database, network, or LLM calls.

These utilities evaluate supplied snapshots. A live adapter must collect and
authenticate those snapshots independently of an actor's assertions.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
import unicodedata


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ChunkReceipt:
    source_id: str
    version: str
    chunk_id: str
    content_sha256: str


def exact_recall(actual_quote: str, original_quote: str) -> bool:
    """Compare extracted quotes, preserving case, words, negations, and codes."""
    def normalize(text: str) -> str:
        return " ".join(unicodedata.normalize("NFC", text).split())

    return normalize(actual_quote) == normalize(original_quote)


def check_summary_snapshot(
    source_ids: Sequence[str],
    tagged_ids: Sequence[str],
    preserved_ids: Sequence[str],
    row_threads: Mapping[str, str],
    active_thread: str,
    *,
    source_hashes: Mapping[str, str] | None = None,
    preserved_hashes: Mapping[str, str] | None = None,
) -> CheckResult:
    """Check one summary's snapshot membership, originals, and ownership."""
    source, tagged, preserved = set(source_ids), set(tagged_ids), set(preserved_ids)
    errors = []
    if not source:
        errors.append("empty_snapshot")
    if len(source_ids) != len(source) or len(tagged_ids) != len(tagged):
        errors.append("duplicate_message_ids")
    if source != tagged:
        errors.append("tagged_ids_differ_from_snapshot")
    if not source.issubset(preserved):
        errors.append("original_messages_missing")
    if any(row_threads.get(row_id) != active_thread for row_id in source | tagged):
        errors.append("thread_ownership_mismatch")
    if source_hashes is not None or preserved_hashes is not None:
        if source_hashes is None or preserved_hashes is None:
            raise ValueError("Both original and preserved content hashes are required")
        if any(
            row_id not in source_hashes
            or source_hashes[row_id] != preserved_hashes.get(row_id)
            for row_id in source
        ):
            errors.append("original_content_changed_or_missing")
    return CheckResult(not errors, tuple(errors))


def check_paper_write(
    expected: Sequence[ChunkReceipt],
    persisted: Sequence[ChunkReceipt],
    claimed_success: bool,
) -> CheckResult:
    """Check a complete paper-version manifest read from independent storage.

    The adapter must scope persisted rows to the requested paper/version. Other
    corpus documents are not evidence of incomplete ingestion.
    """
    errors = []
    identities = [(row.source_id, row.version, row.chunk_id) for row in persisted]
    complete = bool(expected) and set(expected) == set(persisted)
    if len(identities) != len(set(identities)):
        errors.append("duplicate_chunk_identity")
        complete = False
    expected_keys = [(row.source_id, row.version, row.chunk_id) for row in expected]
    if len(expected_keys) != len(set(expected_keys)):
        errors.append("invalid_expected_manifest")
        complete = False
    if not complete:
        errors.append("incomplete_or_different_paper_manifest")
    if claimed_success and not complete:
        errors.append("false_success_claim")
    return CheckResult(not errors, tuple(errors))


def rubric_score(ratings: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Score a complete 0-4 rubric; missing dimensions and NaNs are errors."""
    if not weights or set(ratings) != set(weights):
        raise ValueError("Ratings and weights must have identical nonempty dimensions")
    if any(not isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("Weights must be finite and positive")
    if abs(sum(weights.values()) - 100) > 1e-9:
        raise ValueError("Weights must sum to 100")
    if any(not isfinite(r) or not 0 <= r <= 4 for r in ratings.values()):
        raise ValueError("Ratings must be finite and between 0 and 4")
    return sum(weights[key] * ratings[key] for key in weights) / 4


def judge_route(
    *,
    hard_checks_passed: bool,
    semantic_required: bool,
    selected_for_review: bool,
    judge_budget_available: bool,
) -> str:
    """Route scoring without letting a sampled-out semantic check become a pass."""
    if not hard_checks_passed:
        return "fail_without_judge"
    if not semantic_required:
        return "deterministic_pass"
    if not selected_for_review:
        return "unscored_semantic"
    if not judge_budget_available:
        return "blocked_judge_budget"
    return "semantic_review"


def next_action(
    status: str,
    *,
    attempts_used: int = 1,
    attempt_cap: int = 2,
    scope_allowed: bool = True,
    ambiguous_strategy: bool = False,
    adaptive_reflection_enabled: bool = False,
    reflection_budget_available: bool = False,
    reflection_calls_used: int = 0,
) -> str:
    """Proposed cheap controller policy; not yet connected to call_agent.

    Attempt counts include SDK retries. No status grants new tool authority.
    """
    if attempts_used < 0 or attempt_cap < 1 or reflection_calls_used < 0:
        raise ValueError("Invalid attempt budget")
    if not scope_allowed:
        return "deny_scope"
    if status == "ok":
        return "continue"
    if status == "invalid_args":
        return "repair_arguments" if attempts_used < attempt_cap else "report_blocker"
    if status == "transient_error":
        return "retry_transient" if attempts_used < attempt_cap else "report_blocker"
    if status == "permanent_error":
        return "change_strategy_or_report"
    if status == "partial_write":
        return "reconcile_state"
    if status in {"empty", "not_found"}:
        return "retrieve_alternative_or_abstain"
    if status == "budget_exceeded":
        return "report_blocker"
    if status != "unknown":
        raise ValueError(f"Unrecognized observation status: {status}")
    if (
        ambiguous_strategy and adaptive_reflection_enabled
        and reflection_budget_available and reflection_calls_used == 0
    ):
        return "reflect_once"
    return "actor_uses_validated_observation"
