# Offline evaluation checkpoint

This checkpoint scores **supplied artifacts**, not a running memory agent. It does not import `memory_agent.py` or `helper.py`, initialize Oracle, call a model, download a paper, or upload LangSmith traces. All bundled observations are annotated synthetic fixtures, not results from your successful demo.

Implemented: strict input validation; exact recall; summary snapshot membership, ownership, and content hashes; paper-version/chunk manifests; annotated-ID retrieval metrics; context capacity; tool/result protocol; deterministic recovery-policy probes; local JSON/Markdown reports; and pytest acceptance tests.

Not implemented: live agent/Oracle capture adapters, source-artifact authentication, full C01-C20 coverage, full end-to-end scenarios, arbitrary prose faithfulness, human/model judge ingestion, or runtime reflection. Reports always have `release_ready=false`. Controller probes compute a local policy decision; they do not change your agent's behavior.

## Run from the project root

Your existing venv has pytest installed. In a new environment, install only the evaluation dependency file when needed:

```bash
./venv/Scripts/python.exe -m pip install -r agent_memory/evaluation/requirements-eval.txt
```

Run the deterministic artifact pipeline:

```bash
./venv/Scripts/python.exe -m agent_memory.evaluation
```

Expected: **13 pass, 0 fail, 1 pending**, exit code **3**. The open-ended summary case deliberately stays pending; no paid judge is invoked and no keyword match becomes semantic success. The command prints a new report directory under `agent_memory/evaluation/reports/`. Generated reports are git-ignored.

Evaluate just the paper-manifest contract:

```bash
./venv/Scripts/python.exe -m agent_memory.evaluation --component C07
```

Expected: **1 pass**, exit code **0**. This demonstrates a checker on a synthetic persisted manifest; it does not assert that your live database contains the paper.

Verify that injected regressions are caught:

```bash
./venv/Scripts/python.exe -m agent_memory.evaluation --observations agent_memory/evaluation/fixtures/observations.regression.json
```

Expected: **7 pass, 6 fail, 1 pending**, exit code **1**. Findings include wrong recall, cross-thread/tagging corruption, false save success, irrelevant retrieval, context overflow, and an unallowed tool/missing result.

Repeat `--component` to select the union of labeled cases. A component with no annotated cases returns an input error rather than a zero-case pass. A multi-component case supplies only its stated partial contract coverage.

Use `--dataset`, `--observations`, and `--output` for your own inputs and a new report directory. Existing `report.json`/`report.md` files are not silently overwritten. Dataset/observation fingerprints are recorded; timestamps and scoring duration naturally differ between runs.

| Exit code | Meaning |
|---:|---|
| 0 | All selected artifact contracts passed |
| 1 | At least one contract failed; passing cases cannot hide it |
| 2 | Input/schema/output error; no valid evaluation conclusion |
| 3 | No known failure, but missing observations or semantic review leave the result inconclusive |

## Run pytest without service access

Git Bash, from the project root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 LANGSMITH_TRACING=false LANGSMITH_TEST_TRACKING=false ./venv/Scripts/python.exe -m pytest -c agent_memory/evaluation/pytest.ini agent_memory/evaluation/tests -q
```

PowerShell:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:LANGSMITH_TRACING = 'false'
$env:LANGSMITH_TEST_TRACKING = 'false'
.\venv\Scripts\python.exe -m pytest -c agent_memory/evaluation/pytest.ini agent_memory/evaluation/tests -q
```

The explicit test path prevents collection of unrelated scripts elsewhere in this repository. Third-party plugin loading is disabled. Autouse fixtures block socket connection/DNS calls during tests. These tests validate the evaluator against synthetic positive, negative, and malformed examples; they are not a live-agent quality score.

## Data and observation boundary

`datasets/starter.json` holds 14 annotated cases and expected predicates. `fixtures/observations.good.json` and `observations.regression.json` hold separate supplied snapshots. Missing observations stay pending. Unknown case IDs, duplicate JSON keys, malformed digests, boolean token counts, invalid statuses, and misspelled fields are rejected.

The `captured_artifacts` provenance value is available for later externally collected snapshots. It is a label, not cryptographic proof. A future capture adapter must read original contents, persisted rows, serialized token counts, and tool traces independently of the actor's claims. Hashing invented data does not make it trustworthy. Do not paste model assertions into observed manifests and call that database verification.

Each checker has a narrow scope:

- `exact_recall`: compares an extracted quote with NFC/whitespace normalization only; it does not extract quotes from prose.
- `summary_snapshot`: checks exact tag membership, ownership, retained IDs and original-content hashes; it does not judge summary prose.
- `paper_write`: checks one complete paper/version manifest, duplicate identities and false success claims; actual durable state must come from an independent capture adapter.
- `retrieval`: computes binary Recall@k, Precision@k and reciprocal rank from annotated IDs; it does not judge unannotated passages.
- `context_budget`: compares supplied full-input count plus output reserve with a declared window and checks required IDs; it does not tokenize a live NVIDIA request yet.
- `tool_trace`: checks call/result matching, allowed/required names and call cap. Non-ok results can be valid observations; valid protocol does not mean task success. Ordering, prerequisites, argument schemas and dynamic availability need further checks.
- `controller`: tests bounded pure policy decisions, including a one-reflection-per-request allowance; it does not classify raw exceptions or invoke a reflector.
- `semantic_review`: remains pending regardless of how plausible an answer looks.

## Next meaningful checkpoint

Add a capture adapter against an explicitly disposable Oracle evaluation target and an import-safe agent factory. Compare actual tagged IDs/content hashes and stored chunk manifests with gold fixtures, then introduce an end-to-end replay scenario. Add optional LangSmith tracking when local capture/scoring works and uploads are deliberately enabled. The complete design remains in [EVALUATION_GUIDELINE.md](../EVALUATION_GUIDELINE.md).
