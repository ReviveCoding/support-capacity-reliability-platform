# Known Limitations — 1.4.1rc1

## Release-candidate boundary

This repository is a candidate, not a final production release. `PASS_WITH_RECOURSE` is an offline frozen-scenario recoverability result and is not evidence of a live causal staffing intervention.

## Data and external validity

- Canonical and stress results use deterministic synthetic ACD, workforce, incident, redial, recontact, service-time, patience, and shrinkage data.
- Optional NYC 311 data is a workload-arrival proxy, not contact-center ACD data.
- No AWS internal, customer, agent, or proprietary contact-center data is included.
- Synthetic cost, SLA, abandonment, and staffing outcomes do not establish real business impact.

## Environment qualification gaps

- Docker image build and runtime were not executed because Docker is unavailable in the current environment.
- Remote GitHub Actions were not executed on an exact candidate commit; local workflow syntax and referenced commands were checked only.
- Windows execution was not performed. Windows support should remain provisional until setup, tests, smoke, stress, and wheel E2E pass on Windows Python 3.11.
- Chronos-2 model weights, CUDA training, and GPU inference were not executed. The optional dependency seam, dense-context contract, and explicit failure path are tested.

## Dependency reproducibility

- `pyproject.toml`, `requirements.txt`, and `requirements-dev.txt` are checksummed, but no fully pinned cross-platform lockfile is present.
- The current environment dependency snapshot is included in the handoff; it is evidence, not a portable solver for every platform.
- Editable installation showed a setuptools editable-wheel shutdown delay in the audit container. Standard wheel and sdist builds and installed-wheel execution are the candidate installation evidence.

## Artifact trust and security

- The selected forecast stack is stored with `joblib`; only trusted, checksummed bundles should be deserialized.
- SHA-256 detects corruption or modification but is not an external cryptographic signature or provenance authority.
- The FastAPI service constrains configuration and output paths, but it does not implement authentication, authorization, TLS termination, or multi-tenant isolation.

## Operational scope intentionally excluded

- Real-time streaming ingestion and online recalibration.
- Chat concurrency and silent-abandonment modeling.
- Reinforcement-learning scheduling.
- Kubernetes orchestration and production autoscaling.
- External model registry and signed artifact provenance.
- Multiple production-grade foundation forecasting models.

## Required next qualification

1. Commit the exact candidate source fingerprint and run all remote GitHub Actions jobs.
2. Build and run the Docker image as non-root, verifying writable outputs and both canonical and negative gates.
3. Execute Windows qualification before making unconditional Windows claims.
4. Execute Chronos/CUDA full-mode qualification before making GPU or foundation-model performance claims.
5. Confirm the final delivery ZIP and handoff checksums after transfer.
