# Reliability-Aware Support Forecasting & Workforce Optimization Platform

Version 1.4.1rc2 is a fully runnable, offline Applied Science framework that converts uncertain support demand into tested workforce decisions. The canonical pipeline reconstructs offered workload, benchmarks statistical, classical ML, and GPU-ready forecasting candidates, applies the research-derived Reference-Conditioned Workload Envelope (RCWE), generates coherent operational scenarios, validates schedules in a multi-skill queueing digital twin, and publishes results only after release checks pass.

## Decision problem

A forecast is useful only when it produces a feasible and reliable workforce decision.

```text
synthetic ACD ground truth or public workload proxy
-> schema, provenance, and target contracts
-> leakage-safe temporal features
-> temporal one-step model selection and calibration
-> fixed-origin recursive decision forecast
-> RCWE reference-state correction
-> peak-aware uncertainty calibration
-> coherent load, AHT, patience, and shrinkage scenarios
-> Erlang-A checks and SimPy queue replay
-> strategic capacity MILP
-> availability-aware tactical CP-SAT schedule
-> recourse-aware policy selection on an earlier horizon
-> locked-policy frozen evaluation
-> bounded cross-skill, reserve, overtime, and VTO recovery
-> release gate, persisted forecast bundle, monitoring snapshot, artifact index, and decision memo
```

## What this repository demonstrates

- **Core foundations:** seasonal heuristics, Poisson count regression, quantile loss, time-ordered validation, Erlang-C/A, discrete-event simulation, MILP, and CP-SAT.
- **Applied ML:** LightGBM quantile forecasting, GPU-ready PyTorch quantile training, optional Chronos-2, probabilistic calibration, and fixed-origin recursive inference.
- **Research specialty:** RCWE reference-state modeling, latent/offered/served workload separation, and explicit undercoverage versus excess-capacity trade-offs.
- **Decision science:** coherent joint scenarios, strategic capacity planning, tactical workforce scheduling, recourse-aware policy selection, tail-risk gates, and cost/service-level trade-offs.
- **Production-style engineering:** strict configuration, deterministic seeds, isolated worker execution, CLI, FastAPI, dependency doctor, concurrency lock, staging outputs, atomic publication, persisted model bundle, monitoring snapshot, checksummed artifact index, CI, packaging, and failure-preserving execution.

## Quick start

Python 3.11 through 3.13 is supported.

```bash
python -m pip install .[dev]

# Optional neural forecaster (large dependency; CPU/GPU wheel follows PyTorch platform selection)
python -m pip install .[torch]
python -m pip check
python scripts/qualify_local.py --profile CORE
```

Installed wheels carry bundled smoke, stress, and full configurations. The default smoke and stress paths use the base statistical/LightGBM stack and do not require PyTorch; `configs/smoke_torch.yaml` and `configs/full.yaml` require the optional `torch` or `full` extra. Therefore default validation and canonical execution work outside a source checkout:

```bash
pip install dist/support_capacity_reliability-1.4.1rc2-py3-none-any.whl
cd /tmp
support-capacity validate-config
support-capacity run --require-release
```

Run the expected negative release path separately:

```bash
make stress
```

The commands intentionally separate quality checks and the end-to-end smoke process. This mirrors the independent GitHub Actions jobs and avoids sharing one long-lived native numerical runtime across PyTorch, LightGBM, SciPy, OR-Tools, and SimPy.

## Runtime modes

### Canonical smoke mode

```bash
support-capacity run --config configs/smoke.yaml --require-release
```

The canonical run:

- uses deterministic synthetic ACD and workforce data;
- creates 42 agents, including a planned reserve pool;
- trains seasonal, Poisson, LightGBM, and PyTorch candidates;
- evaluates RCWE variants without forcing an advanced model to win;
- uses no external downloads or proprietary data;
- performs recourse-aware policy selection before the frozen horizon;
- writes to a staging directory and atomically promotes only a completed run;
- preserves the last successful output if a later run fails;
- rejects concurrent writers targeting the same output directory.

### Insufficient-workforce stress mode

```bash
support-capacity run \
  --config configs/stress_insufficient_workforce.yaml \
  --expected-status ITERATE
```

This configuration uses 36 agents and is expected to retain a structural coverage violation after bounded recourse. It verifies that the release gate rejects an infeasible workforce rather than relaxing constraints or hiding the shortage.

### Full local/GPU mode

```bash
pip install .[full]
support-capacity run --config configs/full.yaml
```

Chronos-2 is optional. The adapter requires a dense 96-step context and fails explicitly when its dependency or weights are unavailable. GPU or Chronos execution should be claimed only when a run produces the corresponding hardware and training artifacts.

### Public workload proxy

```bash
python scripts/download_nyc311_sample.py --limit 50000
```

NYC 311 is treated only as a public request-arrival proxy, not as contact-center ACD data. The canonical validation remains synthetic because public request data lacks agent service time, patience, skills, schedules, and counterfactual staffing truth.

## Forecast and selection protocol

The implementation deliberately separates four stages:

1. **Time-ordered one-step holdout evaluation** for candidate comparison.
2. **Calibration-only fitting** for global and peak upper-tail interval adjustments.
3. **Fixed-origin recursive forecasting** for the downstream decision horizon.
4. **Earlier policy-tuning replay followed by a locked, non-overlapping frozen evaluation.**

Future observed AHT, patience, shrinkage, and regime labels are never forecast inputs. RCWE reference retrieval is restricted to historical training states. The served-target comparator and offered-load model use the same RCWE stack so the offered-load ablation is controlled.

Policy selection applies the same bounded recourse mechanism to every candidate before comparing one-sided Student-t cost bounds. The final frozen result is not used to choose the policy.

## Workload and queue semantics

The following quantities are intentionally separate:

- `latent_demand`: generated underlying need, used for controlled diagnostics;
- `offered_contacts`: contacts that enter the simulated ACD, used by queue replay;
- `observed_served`: completed contacts, used for the censored-target comparator;
- `offered_load_estimate`: corrected planning target used by forecasting and capacity planning.

Redials and recontacts return to future intervals. The queue simulator respects the scheduler's assigned skill, agent availability, shift duration, overtime eligibility, service time, patience, and shrinkage scenarios.

## Main artifacts

```text
outputs/smoke/
├── data/
├── metrics/
│   ├── validation_leaderboard.csv
│   ├── fixed_origin_predictions.csv
│   ├── offered_load_ablation.json
│   ├── scenario_diagnostics.json
│   ├── strategic_capacity_plan.csv
│   ├── strategic_tactical_alignment.csv
│   ├── policy_selection_replay_base.csv
│   ├── policy_selection_replay.csv
│   ├── policy_comparison.csv
│   ├── intraday_recourse_actions.csv
│   └── decision_diagnostics.json
├── artifacts/
│   ├── selected_forecast_bundle.joblib
│   ├── selected_forecast_bundle_manifest.json
│   ├── selected_forecast_bundle_verification_input.csv
│   └── selected_forecast_bundle_verification_expected.csv
├── reports/
│   ├── validation_report.md
│   ├── model_selection_report.md
│   ├── decision_memo.md
│   ├── digital_twin_card.md
│   ├── limitations.md
│   └── release_gate_decision.json
├── artifact_index.json
├── run_manifest.json
├── run_summary.json
├── stage.log
└── stage_timing.jsonl
```

## Release gates

The final decision checks:

- overall interval calibration and fixed-origin calibration;
- worst supported-slice WAPE;
- high-load q90 coverage and incident q90 coverage;
- exact cross-sectional scenario coherence;
- strategic MILP success and strategic-to-tactical alignment;
- schedule feasibility and hard coverage violations;
- offered-contact service-level lower confidence bound;
- abandonment upper confidence bound;
- p95 waiting-time upper confidence bound and all-replication queue flow conservation;
- recourse action-rate and recourse-cost-share limits.

Possible outcomes are:

- `PASS`: the locked base schedule passes without recovery;
- `PASS_WITH_RECOURSE`: bounded recovery is required and all recourse gates pass;
- `ITERATE`: at least one predictive, operational, feasibility, or recourse gate fails.

## Local and GitHub verification

```bash
make doctor      # dependencies, writable paths, HiGHS, CP-SAT, SimPy
make lint        # Ruff format and lint
make coverage    # tests plus >=85% source coverage gate
make validate    # configuration contracts
make smoke       # canonical PASS/PASS_WITH_RECOURSE requirement
make stress      # expected ITERATE negative path
make package     # clean wheel/sdist plus archive and isolated-wheel smoke
make wheel-e2e   # installed-wheel CLI and API pipelines outside the repository
make verify-output       # verify every published artifact checksum
make verify-model-bundle # replay the persisted forecast stack
make api-smoke   # real Uvicorn health and staffing HTTP checks
make ci-static   # workflow and Docker contract checks
```

GitHub Actions runs:

- an Ubuntu Python 3.11, 3.12, and 3.13 quality matrix plus Windows Python 3.11;
- an isolated canonical smoke job;
- an isolated insufficient-workforce negative-gate job;
- immutable-SHA-pinned actions, least-privilege permissions, CodeQL, Dependabot, and artifact upload for reports, metrics, model bundles, manifests, release decisions, and package distributions;
- installed-wheel CLI and API E2E execution in the isolated smoke job.


### Cross-platform qualification

The canonical entrypoint is implemented in Python and is shared by local execution and GitHub Actions:

```bash
python scripts/qualify_local.py --profile CORE
python scripts/qualify_local.py --profile STANDARD
python scripts/qualify_local.py --profile EXTENDED
```

POSIX and PowerShell wrappers are provided as `scripts/qualify_local.sh` and
`scripts/qualify_local.ps1`. The runner writes a machine-readable command ledger and separate logs.
A workflow file or local parity run is not evidence of a GitHub-hosted pass; hosted evidence must refer
to the exact source commit.

## API

```bash
make api
```

Endpoints:

- `GET /health`
- `POST /required-staffing`
- `POST /run-pipeline`

The pipeline endpoint accepts YAML files only from the active workspace `configs/` directory, with bundled defaults available when running an installed wheel outside a checkout. API outputs are constrained to the workspace `outputs/` directory. Unexpected internal errors are logged server-side and returned as a generic HTTP 500 response. Concurrent runs for the same output target return a conflict rather than corrupting outputs.

## Claim boundaries

- All canonical outcomes are offline simulations.
- No AWS internal, customer, or proprietary contact-center data is used.
- Public datasets are labeled as proxies.
- Simulated cost, SLA, abandonment, and staffing results are not live business impact.
- Chronos or GPU execution is claimed only when corresponding artifacts exist.
- `PASS_WITH_RECOURSE` is a frozen two-stage recoverability experiment, not a live causal intraday deployment.

## Current validated behavior

The latest checked implementation has:

- a canonical 42-agent path that reaches `PASS_WITH_RECOURSE`;
- a 36-agent stress path that correctly remains `ITERATE`;
- `lightgbm+rcwe` selected in the latest deterministic smoke data;
- exact scenario coherence;
- optimal strategic HiGHS status;
- availability-aware CP-SAT scheduling;
- queue flow conservation;
- fail-closed release evidence, closing-horizon queue semantics, and replication schema contracts;
- automated unit, integration, contract, lifecycle, and E2E coverage above the repository 85% release gate; consult `qualification_manifest.json` for the exact final count and consecutive-pass evidence;
- installed-wheel CLI and API pipeline execution outside the repository;
- persisted selected forecast bundle replay, release-time PSI/bias/support monitoring, and full published-artifact checksum verification.

Exact metrics and test counts are generated from the current code and should be read from `outputs/smoke/run_summary.json`, `outputs/smoke/reports/release_gate_decision.json`, and the verification reports rather than copied from this README.

## License

MIT License. See `LICENSE`.

## Dependency reproducibility

Runtime dependencies use tested upper bounds to prevent unqualified major/minor upgrades. The exact Python 3.13 qualification set is recorded in `constraints/qualification-py313.txt`; use `python -m pip install -c constraints/qualification-py313.txt .` to reproduce the qualified environment.
