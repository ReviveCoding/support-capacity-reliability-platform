# Release Qualification Report

## Current qualification identity

| Item | Value |
|---|---|
| Project version | `1.4.1rc2` |
| Functional release tag | [`v1.4.1rc2-windows-qualified-v25`](https://github.com/ReviveCoding/support-capacity-reliability-platform/releases/tag/v1.4.1rc2-windows-qualified-v25) |
| Qualified source commit | [`e66caf8cef9b6cefa677a0d90e29d010b79ae9a5`](https://github.com/ReviveCoding/support-capacity-reliability-platform/commit/e66caf8cef9b6cefa677a0d90e29d010b79ae9a5) |
| GitHub Actions CI | [`28462033837`](https://github.com/ReviveCoding/support-capacity-reliability-platform/actions/runs/28462033837) â€” PASS |
| CodeQL Python analysis | [`28462033948`](https://github.com/ReviveCoding/support-capacity-reliability-platform/actions/runs/28462033948) â€” PASS |
| Local Windows evidence | Clean short-root STANDARD controller result â€” PASS |
| Claim boundary | Offline synthetic operational replay; not live customer traffic, AWS production deployment, or a causal real-time rollout. |

## Verdict

**Qualified for the documented local Windows clean-release and GitHub-hosted CI scope.**

The functional release was qualified locally on Windows in a fresh short physical root and then validated on GitHub-hosted Windows and Ubuntu runners. The qualification does not claim Docker, CUDA/GPU, Chronos-2, live production, or external vulnerability-database execution.

## Passed evidence

### Windows local clean-release qualification

The final short-root controller completed successfully after fresh extraction and environment bootstrap. The run validated:

- Python 3.11 availability, fresh virtual environment, package installation, qualification-tool installation, and `pip check`.
- STANDARD qualification: doctor, formatting, lint, test/coverage gate, smoke validation, build, distribution verification, installed-wheel E2E, API and pipeline smoke, artifact/output verification, and the stress gate.
- Immutable identity evidence for the release ZIP and controller, recorded in the frozen local evidence snapshot.

### GitHub-hosted CI

The exact qualified commit passed all seven jobs in CI run `28462033837`:

| Job | Platform / runtime | Result |
|---|---|---|
| `smoke-ubuntu-py3.11` | Ubuntu, Python 3.11 | PASS |
| `insufficient-workforce-ubuntu-py3.11` | Ubuntu, Python 3.11 | PASS; expected `ITERATE` negative gate |
| `torch-quality-ubuntu-py3.11` | Ubuntu, Python 3.11, CPU PyTorch | PASS |
| `quality-windows-latest-py3.11` | Windows, Python 3.11 | PASS |
| `quality-ubuntu-latest-py3.11` | Ubuntu, Python 3.11 | PASS |
| `quality-ubuntu-latest-py3.12` | Ubuntu, Python 3.12 | PASS |
| `quality-ubuntu-latest-py3.13` | Ubuntu, Python 3.13 | PASS |

The smoke job verified doctor, configuration validation, canonical `PASS_WITH_RECOURSE` execution, output verification, model-bundle replay, build/distribution verification, and installed-wheel CLI/API execution. The CI workflow also uses pinned actions, least-privilege permissions, and artifact uploads for qualification evidence.

### Static security analysis

CodeQL Python analysis completed successfully in run `28462033948` for the same commit.

## Current canonical outcome

The deterministic canonical smoke replay selects `lightgbm+rcwe` and produces a `PASS_WITH_RECOURSE` outcome. Its operational results are artifacts of offline synthetic replay and are not production business metrics.

| Metric | Canonical artifact value |
|---|---:|
| Fixed-origin WAPE | 38.13% |
| Fixed-origin 80% interval coverage | 86.11% |
| Worst supported-slice WAPE | 43.67% |
| Peak q90 coverage | 71.21% |
| Incident q90 coverage | 91.67% |
| Strategic capacity solver | HiGHS optimal |
| Schedule feasibility | 100% |
| Hard violations | 0 |
| Simulated service level | 99.47% |
| Simulated abandonment | 0.36% |
| Intraday recourse actions | 4 |
| Recourse action rate | 9.52% |
| Recourse cost share | 9.73% |

The 36-agent insufficient-workforce negative control remains `ITERATE`. This is intentional evidence that the release gate fails closed when bounded recourse cannot restore feasibility.

## Explicit non-claims

| Surface | Status |
|---|---|
| Docker build/run qualification | Not executed |
| CUDA / GPU training qualification | Not executed |
| Chronos-2 full-mode qualification | Not executed |
| Live customer or proprietary contact-center data | Not used |
| AWS production deployment | Not executed |
| Causal real-time intraday recourse rollout | Not claimed |
| External vulnerability-database audit | Not claimed by this release |

## Evidence artifacts and reproducibility

- The exact functional release is pinned by the annotated tag and GitHub Release above.
- `scripts/qualify_local.py` is the shared cross-platform qualification entrypoint.
- GitHub Actions retains CI artifacts according to workflow retention settings; the README figures are generated from the exact smoke and stress artifacts, with provenance stored in `docs/figures/readme_figure_manifest.json`.
- Local Windows release ZIP/controller hashes and final controller state are retained outside the public source tree as local evidence artifacts and are not represented as GitHub release assets.

## Historical report

The detailed pre-upload qualification report, including earlier Linux-only and pre-hosted-CI findings, is retained in [`docs/historical_pre_github_qualification.md`](historical_pre_github_qualification.md). It is historical context, not the current final verdict.