"""Local artifact evaluation CLI. Never imports the live memory-agent module."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

from .schema import InputError, load_inputs, load_json
from .scorers import score_case


ROOT = Path(__file__).resolve().parent
COMPONENTS = tuple(f"C{value:02d}" for value in range(1, 21))


def fingerprint(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate(dataset: dict, observations: dict, *, components: list[str] | None = None) -> dict:
    started = perf_counter()
    cases = load_inputs(dataset, observations)
    if components:
        if not set(components).issubset(COMPONENTS):
            raise InputError("Unknown requested component")
        cases = [case for case in cases if set(case.component_ids) & set(components)]
    if not cases:
        raise InputError("No annotated cases cover the requested selection")
    results = [score_case(case, observations["records"].get(case.case_id)) for case in cases]
    counts = {status: 0 for status in ("pass", "fail", "pending")}
    counts.update(Counter(result["status"] for result in results))
    decision = "fail" if counts["fail"] else "inconclusive" if counts["pending"] else "pass"
    covered = sorted({key for case in cases for key in case.component_ids})
    per_component = {}
    for key in covered:
        component_results = [row for row in results if key in row["component_ids"]]
        per_component[key] = {
            status: sum(row["status"] == status for row in component_results)
            for status in ("pass", "fail", "pending")
        }
    return {
        "report_schema_version": 1,
        "evaluator_version": "offline-checkpoint-1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset["dataset_id"],
        "dataset_sha256": fingerprint(dataset),
        "observations_sha256": fingerprint(observations),
        "observation_provenance": observations["provenance"],
        "mode": "offline_artifact_contracts",
        "artifact_authentication": "not_verified_by_this_runner",
        "scope_decision": decision,
        "release_ready": False,
        "release_note": "Artifact checks only; no live agent, Oracle, semantic judge, or end-to-end evaluation was run.",
        "counts": {"total": len(results), **counts},
        "fully_scored_fraction": (counts["pass"] + counts["fail"]) / len(results),
        "all_selected_cases_passed": decision == "pass",
        "critical_failures": [
            {"case_id": row["case_id"], "flags": row["critical_flags"]}
            for row in results if row["critical_flags"]
        ],
        "coverage": {
            "covered_components": covered,
            "uncovered_guideline_components": sorted(set(COMPONENTS) - set(covered)),
            "per_component": per_component,
            "coverage_note": "Case labels indicate partial contract coverage, not complete component validation.",
        },
        "evaluator_usage": {
            "actor_calls": 0, "judge_calls": 0, "reflection_calls": 0,
            "network_calls": 0, "oracle_calls": 0,
            "duration_ms": round((perf_counter() - started) * 1000, 3),
        },
        "evaluated_system_usage": "not_collected; do not infer product cost or latency from scorer usage",
        "results": results,
    }


def markdown_report(report: dict) -> str:
    def cell(value):
        return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    counts = report["counts"]
    lines = [
        "# Offline Evaluation Checkpoint", "",
        f"Dataset: {cell(report['dataset_id'])}",
        f"Observation provenance: {cell(report['observation_provenance'])}",
        f"Decision for selected artifact checks: **{report['scope_decision']}**", "",
        "This is not a live agent or release evaluation. Artifact authenticity is not verified by this runner.",
        "A passing result applies only to declared checks on the supplied snapshots.", "",
        f"Cases: {counts['total']} | Pass: {counts['pass']} | Fail: {counts['fail']} | Pending: {counts['pending']}",
        f"Fully scored fraction: {report['fully_scored_fraction']:.1%}",
        "Evaluator network/model/Oracle calls: 0. Product usage was not collected.", "",
        "## Critical failures", "",
    ]
    lines.extend(
        f"- {cell(row['case_id'])}: {', '.join(row['flags'])}"
        for row in report["critical_failures"]
    )
    if not report["critical_failures"]:
        lines.append("None observed in the supplied artifacts.")
    lines.extend(["", "## Cases", "", "| Case | Components | Status | Findings |", "|---|---|---|---|"])
    for row in report["results"]:
        lines.append(f"| {cell(row['case_id'])} | {', '.join(row['component_ids'])} | {row['status']} | {cell(', '.join(row['errors']) or 'Declared contract satisfied')} |")
    lines.extend([
        "", "## Coverage limits", "",
        "Partial contract coverage: " + ", ".join(report["coverage"]["covered_components"]),
        "No cases selected for: " + ", ".join(report["coverage"]["uncovered_guideline_components"]),
        "Semantic review remains pending wherever required; it is never assumed to pass.",
        "", f"Dataset SHA256: `{report['dataset_sha256']}`",
        f"Observations SHA256: `{report['observations_sha256']}`", "",
    ])
    return "\n".join(lines)


def write_reports(report: dict, output: Path) -> None:
    targets = [output / "report.json", output / "report.md"]
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids silently replacing earlier reports or user files.
    if any(path.exists() for path in targets):
        raise InputError("Report files already exist; choose a new output directory")
    payloads = [json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", markdown_report(report)]
    for path, payload in zip(targets, payloads):
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score offline artifacts; never runs the agent or external services.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "starter.json")
    parser.add_argument("--observations", type=Path, default=ROOT / "fixtures" / "observations.good.json")
    parser.add_argument("--output", type=Path, help="New local report directory; existing report files are not overwritten")
    parser.add_argument("--component", action="append", choices=COMPONENTS, help="Select cases by component; repeat as needed")
    args = parser.parse_args(argv)
    try:
        report = evaluate(load_json(args.dataset), load_json(args.observations), components=args.component)
        output = args.output or ROOT / "reports" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
        write_reports(report, output)
    except (InputError, OSError) as exc:
        print(f"Evaluation input/report error: {exc}", file=sys.stderr)
        return 2
    counts = report["counts"]
    print(f"{report['scope_decision'].upper()}: {counts['pass']} pass, {counts['fail']} fail, {counts['pending']} pending")
    print(f"Local reports: {output.resolve()}")
    print("Scope: supplied artifacts only; release_ready=false; zero external evaluation calls.")
    return {"pass": 0, "fail": 1, "inconclusive": 3}[report["scope_decision"]]


if __name__ == "__main__":
    raise SystemExit(main())
