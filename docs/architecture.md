# Architecture

```text
Synthetic ACD or public workload proxy
  -> schema, provenance, and target contracts
  -> leakage-safe temporal feature mart
  -> heuristic / Poisson / LightGBM / PyTorch / optional Chronos-2 forecasts
  -> time-ordered one-step candidate evaluation
  -> fixed-origin recursive decision forecast
  -> RCWE and peak-aware interval calibration
  -> coherent load, AHT, patience, and shrinkage scenarios
  -> Erlang-A staffing checks and SimPy multi-skill voice digital twin
  -> strategic capacity-unit MILP with shortage and excess penalties
  -> availability-aware tactical CP-SAT schedule
  -> recourse-aware policy-selection replay
  -> locked-policy non-overlapping frozen replay
  -> bounded cross-skill, reserve, overtime, and VTO recovery
  -> release gate, model bundle, monitoring snapshot, artifact contract, and decision memo
```

## Package boundaries

- `data/`: synthetic ACD generation, public adapters, contracts, and feature construction.
- `forecasting/`: statistical, classical ML, PyTorch, and optional Chronos-2 candidates.
- `reliability/`: RCWE, interval calibration, coherent scenarios, and release gates.
- `queueing/`: Erlang-C/A and discrete-event simulation.
- `optimization/`: strategic capacity, tactical scheduling, and recourse.
- `evaluation/`: forecast and operational policy metrics.
- `workflow/forecast_planning.py`: fixed-origin recursion and scenario-to-capacity planning.
- `workflow/policy_selection.py`: base and repaired policy comparison.
- `artifacts.py`: trusted model-bundle serialization, portable replay, and published-tree checksums.
- `monitoring.py`: release-time PSI, bias, interval-coverage, and RCWE-support diagnostics.
- `runtime.py`: isolated worker execution for CLI and API pipeline requests.
- `pipeline.py`: orchestration, stage recording, locking, artifact validation, and atomic publication.

## Interfaces

- Forecast output: q10, q50, q90 for every timestamp-region-skill leaf.
- RCWE output: corrected quantiles, reference expectation, support score, and low-support flag.
- Scenario output: `[scenario, horizon, leaf]` demand plus AHT, patience, and shrinkage tensors.
- Strategic output: skill-level capacity units, expected shortage, and expected excess.
- Tactical output: one assigned skill for every scheduled agent-shift row.
- Recourse output: hold decisions and applied cross-skill, reserve, overtime, or VTO changes.
- Decision output: cost, service level, abandonment, waiting time, utilization, feasibility, and confidence bounds.
- Model artifact: selected fitted forecaster, calibrator, optional RCWE layer, feature contract, and replay sample.
- Published artifact contract: exact file list, byte sizes, SHA-256 checksums, release status, and selected variant.

## Operational safety

- Each run acquires an output-target lock.
- Work is written to a unique staging directory.
- Only a complete run whose stage sequence, release report, summary hash, model-bundle hash, and artifact tree agree is atomically promoted.
- The previous successful output is retained if a later run fails.
- Stale locks and staging directories are recovered safely.
- API configuration paths are constrained to workspace YAML files under `configs/`; packaged defaults support installed-wheel execution.
- Canonical CLI and Docker execution require an acceptable release state.
- Docker runs as a non-root user with a writable outputs directory.

## Fallback hierarchy

1. selected validated candidate;
2. LightGBM quantile forecast;
3. seasonal forecast;
4. conservative Erlang-A staffing;
5. the last successfully published output and schedule artifacts.
