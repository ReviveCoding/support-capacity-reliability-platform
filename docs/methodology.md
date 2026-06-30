# Methodology

## Workload definitions

The synthetic generator separates:

- latent intended demand;
- contacts actually offered to the ACD queue;
- served and abandoned contacts;
- future redials and recontacts;
- effective offered-load estimate.

The served-only comparator and offered-load model use the same RCWE stack so the reconstruction ablation is controlled. Frozen queue replay uses actual `offered_contacts`, because those are the contacts entering the simulated ACD.

## Leakage-safe forecasting

The supervised feature set excludes future regime labels and current realized operational quantities. Available predictors are:

- calendar encodings;
- target lags;
- shifted rolling means and standard deviations;
- lagged AHT, patience, and shrinkage measurements;
- deterministic region and skill encodings.

Time-ordered one-step holdout results are used for candidate comparison. Calibration is fitted on a separate calibration partition. Workforce decisions use a fixed-origin recursive forecast, so later horizon lags contain earlier predictions rather than future truth.

## Reference-Conditioned Workload Envelope

RCWE retrieves historical states from the training partition only. Similarity uses lagged demand, shifted rolling statistics, calendar encoding, and lagged operational measurements. Support is calibrated relative to each region-skill historical neighbor-distance distribution, preventing raw feature dimensionality from making every state look out of distribution. Low-support states widen intervals rather than forcing confident extrapolation.

## Peak-aware calibration

Global residual calibration controls overall interval coverage. A separate calibration-only upper-tail adjustment is activated only when the predicted q90 crosses a training-derived peak threshold. This preserves interval efficiency while protecting high-load and incident slices. Peak and incident q90 coverage are blocking release checks.

## Coherent joint uncertainty scenarios

Calibration residuals estimate cross-sectional dependence across region-skill leaves and temporal persistence. Generated load trajectories are nonnegative. Load-correlated AHT, patience, and shrinkage tensors are passed to capacity planning. Aggregate global, region, and skill totals are derived directly from leaf samples, and exact leaf-to-global coherence is a blocking gate.

## Queue and workforce evaluation

Erlang-A supplies analytical staffing approximations. The discrete-event simulator evaluates actual offered contacts under assigned skills, heterogeneous service times, patience, abandonment, shrinkage, and agent proficiency. Service level uses all offered contacts as the denominator; answered-only service level is diagnostic. New service cannot start after the staffed horizon closes, although calls already in service may finish. Lognormal service times are parameterized so the configured AHT remains the expected service duration.

Strategic capacity planning produces capacity units for the configured planning horizon, not weekly FTE claims. Tactical scheduling uses CP-SAT skill, availability, shift, daily-hour, overtime-eligibility, and preference constraints. Shift duration is derived from the decision horizon and used consistently in capacity, schedule, and labor-cost calculations.

## Policy selection and bounded recourse

The policy-tuning replay temporally precedes the final frozen replay. Every candidate receives the same bounded recourse mechanism before comparison. One-sided Student-t bounds use sample standard deviations for small simulation samples, including total cost and p95 waiting time. Flow conservation must hold in every replication. The selected base policy is locked before final evaluation.

The realized-offered staffing reference is diagnostic only. It is not labeled as a perfect-information oracle, so EVPI or VSS are not claimed.

The frozen recoverability experiment can apply same-shift cross-skill reassignment, planned reserve, overtime, and VTO after the offered-demand scenario is revealed. Availability, overtime eligibility, daily hours, action rate, and positive recourse-cost share remain bounded. Release artifacts distinguish `PASS`, `PASS_WITH_RECOURSE`, and `ITERATE`.

## Claim discipline

The canonical run is an offline synthetic operational replay. NYC 311 and CFPB are workload-pattern proxies only. No proprietary customer, agent, AWS, or production contact-center data is used. GPU, Chronos, or live operational-impact claims require separate run evidence.


## Model artifact and monitoring contract

The selected fitted forecasting stack is persisted as a trusted `ForecastModelBundle` containing the forecaster, interval calibrator, optional RCWE layer, feature names, state features, lags, rolling windows, target, package version, and schema version. A held-out feature sample and expected quantiles are stored beside the bundle. The pipeline and a standalone CLI replay the bundle in an isolated process and require numerical equality within a strict tolerance. Because `joblib` deserialization can execute Python objects, bundles must only be loaded from trusted runs whose checksum matches the manifest.

Release-time monitoring stores PSI for the target, AHT, patience, and shrinkage across calibration and test partitions, plus signed forecast bias, interval coverage, mean RCWE support, and low-support rate. These are warning diagnostics rather than causal evidence or automatic production retraining triggers.

Before atomic promotion, the pipeline checks required artifacts, non-empty files, summary/release agreement, summary hash, exact stage order, selected-variant/bundle agreement, bundle and replay-file checksums, and then writes an artifact index covering the published tree. `verify-output` detects missing, added, resized, or modified files after publication.
