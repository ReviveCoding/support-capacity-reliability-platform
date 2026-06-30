# Improvement Report — 1.4.1rc1

## Executive assessment

The 1.4.0 archive already contained a strong research-to-decision pipeline, deterministic canonical and negative fixtures, release gates, a persisted model bundle, and package/API/CI scaffolding. The candidate audit found two high-impact reliability defects in the validation harness rather than in forecast or workforce logic: native subprocess descendants could keep inherited pipes open, and the entire pytest suite could retain incompatible native ML/solver runtimes in one interpreter. Version 1.4.1rc1 isolates process execution and pytest modules, preserves all baseline forecast and decision metrics, and adds a machine-verifiable release-candidate handoff.

This document records candidate-level evidence. It does **not** claim final release qualification; Docker, exact-commit GitHub-hosted execution, Windows, and Chronos/CUDA remain follow-up gates.

## Phase 0 baseline

### Repository and entrypoints

- Packaging: `pyproject.toml`, setuptools, wheel and sdist.
- CLI: `support-capacity` / `support_capacity_reliability.cli:main`.
- API: FastAPI application in `support_capacity_reliability.api.app`.
- Canonical pipeline: `make smoke`.
- Expected negative gate: `make stress`.
- Model artifact reload: `verify-model-bundle`.
- Published artifact verification: `verify-output`.
- CI: Python 3.11–3.13 quality matrix plus independent smoke and insufficient-workforce jobs.
- Docker: non-root, non-editable package install, release-required command; runtime build not available in the audit environment.

### Pipeline map

| Stage | Implementation | Input contract | Output contract | Failure handling | Tests |
|---|---|---|---|---|---|
| Configuration | `config.py` | validated YAML/Pydantic model | immutable application config | fail-fast validation | config tests |
| Data generation/adaptation | `data/synthetic.py`, `data/public_adapters.py` | configured regions, skills, horizon | interval and agent frames | schema errors | synthetic/contracts tests |
| Schema validation | `data/contracts.py` | interval/agent frames | validated typed frames | explicit validation error | contract tests |
| Feature construction | `data/features.py` | historical observations only | leakage-safe features | missing history error | feature tests |
| Forecast candidates | `forecasting/*` | train/calibration partitions | q10/q50/q90 | explicit candidate failure | forecast/Chronos/Torch tests |
| Reliability layer | `reliability/rcwe.py`, `calibration.py` | base forecast and reference states | calibrated quantiles/support | low-support widening | RCWE/calibration tests |
| Fixed-origin planning | `workflow/forecast_planning.py` | historical frame and locked model | recursive decision forecast | planning error | integration and axis tests |
| Scenario generation | `reliability/scenarios.py` | calibrated forecasts/residual history | coherent demand/AHT/patience/shrinkage tensors | contract error | scenario tests |
| Strategic optimization | `optimization/capacity.py` | scenario capacity requirements | skill capacity plan | solver status persisted | capacity tests |
| Tactical scheduling | `optimization/scheduler.py` | agents, shifts, skills, availability | assigned schedule | infeasible status | scheduler tests |
| Queue replay | `queueing/simulator.py` | offered contacts and assigned schedule | replication metrics | schema/finite-value failure | simulator tests |
| Policy selection/recourse | `workflow/policy_selection.py`, `optimization/recourse.py` | earlier replay candidates | locked base policy and repaired schedule | bounded fallback/ITERATE | recourse/decision tests |
| Release gate | `reliability/release_gate.py` | required predictive/operational evidence | PASS/PASS_WITH_RECOURSE/ITERATE | fail-closed missing evidence | gate tests |
| Artifact publication | `artifacts.py`, `pipeline.py` | complete staged run | bundle, index, summary, reports | lock, staging, atomic promotion | artifact/atomic tests |
| Artifact loading | `artifacts.py`, CLI | trusted checksummed bundle | replayed quantiles | checksum/schema/version failure | artifact tests |
| Monitoring/reporting | `monitoring.py`, `reporting.py` | final run evidence | PSI/bias/support and reports | explicit monitoring status | monitoring tests |

## Prioritized findings

Priority score uses severity × likelihood × affected scope × evidence confidence ÷ implementation cost.

| Finding | Severity | Likelihood | Scope | Confidence | Cost | Priority | Disposition |
|---|---:|---:|---:|---:|---:|---:|---|
| Inherited subprocess pipes can keep doctor/CLI/test parents waiting after a worker exits | High (4) | 4 | 5 | 1.0 | 2 | 40.0 | Fixed |
| Long-lived pytest process accumulates incompatible native ML/solver shutdown state | High (4) | 4 | 5 | 1.0 | 2 | 40.0 | Fixed |
| Baseline command evidence was prose-oriented rather than machine-verifiable handoff data | Medium (2) | 5 | 4 | 1.0 | 2 | 20.0 | Fixed |
| No fully pinned cross-platform dependency lock | Medium (2) | 3 | 3 | 1.0 | 3 | 6.0 | Follow-up |
| Docker, GitHub-hosted, Windows, and Chronos/CUDA qualification unavailable locally | High externally | — | — | 1.0 | external | — | Documented qualification gates |

## Major improvement rounds

### Round 1 — bounded subprocess lifecycle

**Problem.** `subprocess.run(..., stdout=PIPE, stderr=PIPE)` could wait indefinitely when native libraries or grandchildren retained inherited descriptors after the direct child exited.

**Root cause.** Pipe lifecycle was coupled to descendants, while timeout cleanup did not consistently terminate the process group.

**Change.** Added `process_utils.py` with temporary-file output capture, new process groups, bounded waits, partial-output recovery on timeout, and optional descendant cleanup for worker-style commands. Doctor probes, pipeline workers, and bundle replay use this utility.

**Regression risk.** Aggressive process-group cleanup can interrupt legitimate library cleanup. Normal probe success therefore does not kill descendants; worker contexts opt in explicitly.

**Validation.** Dedicated success, timeout, inherited-descriptor, environment, and working-directory tests; doctor and canonical CLI return normally.

### Round 2 — isolated test modules and merged coverage

**Problem.** Assertions passed, but a single long-lived pytest interpreter could fail to terminate after importing combinations of Torch, LightGBM, SciPy, HiGHS, OR-Tools, and SimPy.

**Root cause.** Native runtime shutdown interactions accumulated across unrelated test modules.

**Change.** `scripts/run_test_suite.py` executes every test module in a bounded isolated process. Coverage shards are merged, and a machine-readable test summary records module and test counts.

**Regression risk.** Isolation can hide accidental module-order dependencies. This is desirable: tests must be independently reproducible. The integration pipeline remains a real non-mocked end-to-end test.

**Validation.** All isolated modules pass; merged source coverage remains at the required threshold; integration and unit suites both terminate normally.

### Round 3 — release-candidate evidence and handoff

**Problem.** Existing reports documented results but did not provide the complete candidate manifest required for downstream qualification.

**Change.** Added `scripts/create_release_handoff.py`, `release_candidate_handoff.json`, source manifest/diff checksums, dependency snapshot, entrypoint inventory, baseline/final metrics, command evidence, evidence-level labeling, unresolved severity, and next qualification gates.

**Validation.** The generator re-reads the handoff and verifies every referenced local path and SHA-256 before success.

## Scientific and decision validity

- Candidate selection and final frozen evaluation remain temporally separated.
- Baseline and candidate use identical deterministic fixtures, split, metrics, and thresholds.
- No forecast, service-level, abandonment, cost, or release threshold was relaxed.
- Canonical and insufficient-workforce release statuses remain unchanged.
- The high-risk stress path still fails only workforce feasibility rather than model or pipeline integrity.
- Synthetic validation demonstrates correctness and pipeline integration, not external operational impact.

## Structural optimization

The codebase already separates forecasting, reliability, queueing, optimization, policy selection, artifact publication, and runtime isolation. The audit removed no public API and performed no broad cosmetic rewrite. The remaining large orchestration function is covered by a real integration test; splitting it further would provide lower value than the added cross-module regression risk at candidate freeze.

## Candidate conclusion

- Confirmed Critical defects: 0.
- Executable High defects in the current environment: 0 after Rounds 1–2.
- External High qualification items: remote GitHub-hosted run and Docker runtime run, explicitly retained for the next qualification stage.
- Candidate designation: `1.4.1rc1`.
- Evidence ceiling in this phase: E3, not E4.
