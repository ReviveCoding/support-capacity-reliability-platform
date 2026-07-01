# Six-Hour Micro-Shift Validation Evidence

## Scope

This compact package retains immutable, reviewable validation artifacts for the
six-hour micro-shift staffing implementation at source commit
`ef9bcfa162630c35f379c6321e79919a50b64792`.

It keeps the Docker requalification manifest, frozen LightGBM/Torch benchmark
configuration and report, matched full-scale legacy and micro-shift configs,
the final confirmation report, selected-system summaries, policy replay
tables, selected schedules, and release-gate decisions.

Large synthetic contact tables, interval tables, model bundles, virtual
environments, container images, and transient logs are intentionally excluded.
The retained reports carry their associated hashes and identity checks.

## Matched Full-Scale Result

- Exact agents, intervals, split-manifest, frozen-prediction, and
  fixed-origin-prediction identity: `True`.
- Policy-fixed comparator: `probabilistic_safety`.
- Policy-fixed non-compensatory checks: `True`.
- Selected-system non-compensatory checks: `True`.
- Hard violations: `61` legacy to
  `18` micro-shift.
- Schedule feasibility: `0.5986842105263158` legacy to
  `0.9203539823008849` micro-shift.
- Total cost: `67367.01000000001` legacy to
  `48729.0` micro-shift.

## Claim Boundary

The evidence concerns an offline synthetic operational replay. It is not
production contact-center validation or production release approval. The
micro-shift selected system remains `ITERATE` with residual hard violations.

## Reproduction Notes

The historical configurations are retained under `frozen_geometry` and
`fullscale`. Their historical `project.output_dir` values are preserved for
provenance. Before a new run, replace only `project.output_dir` with a fresh
local destination. Do not overwrite evidence files in this directory.