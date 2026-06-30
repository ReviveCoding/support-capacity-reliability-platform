# Release Qualification Report - support-capacity-reliability 1.4.1rc2

## Current qualification verdict

**DOCKER-QUALIFIED FOLLOW-UP RELEASE**

- Current Docker-qualified tag: `v1.4.1rc2-docker-qualified-v26`
- Docker-qualified source commit: `1c03678859c9c1cebb98abba62a916a3fc3bbece`
- Original Windows-qualified functional tag: `v1.4.1rc2-windows-qualified-v25`
- Hosted CI: **PASS** - run `28472655707`
- CodeQL Python analysis: **PASS** - run `28472655791`
- Docker local runtime qualification: **PASS**
- Canonical offline release outcome inside Docker: **PASS_WITH_RECOURSE**

The original Windows-qualified v25 tag remains immutable. The v26 follow-up adds Docker runtime evidence without rewriting v25.

## Executed evidence

| Scope | Result | Evidence |
|---|---|---|
| Windows local clean short-root STANDARD qualification | PASS | v25 functional release lineage |
| GitHub-hosted quality validation | PASS | Windows Python 3.11; Ubuntu Python 3.11, 3.12, and 3.13 |
| GitHub-hosted release smoke | PASS | Ubuntu Python 3.11 |
| Optional CPU PyTorch forecaster validation | PASS | Ubuntu Python 3.11 |
| Insufficient-workforce negative release gate | PASS | Ubuntu Python 3.11 |
| CodeQL Python analysis | PASS | GitHub-hosted run `28472655791` |
| Docker Desktop WSL2 Linux-container qualification | PASS | Local Docker evidence manifest SHA-256: `95516aaf915688ffcf88daa0af0d5d0c0d737612f5da1f1e09dae0494e57b833` |

## Docker qualification details

The v26 Docker qualification was executed from a clean source checkout at commit `1c03678859c9c1cebb98abba62a916a3fc3bbece` using a Linux container built from `python:3.11-slim`.

The runtime path verified:

1. clean Docker image build;
2. GNU OpenMP runtime `libgomp1` for LightGBM;
3. offline container execution with `--network none`;
4. `--cap-drop ALL` and `no-new-privileges`;
5. `support-capacity doctor --config configs/smoke.yaml`;
6. canonical smoke execution with `--require-release`;
7. output integrity verification;
8. persisted model-bundle replay verification;
9. container exit code `0`.

The Docker runtime produced the canonical `PASS_WITH_RECOURSE` decision. The image and container were removed after evidence capture; logs, copied runtime outputs, inspect records, and the manifest are retained locally under the Docker qualification evidence directory.

## Claim boundary

This is a qualified **offline synthetic operational replay** and portability/reliability artifact. It does not establish:

- live customer traffic or live contact-center business impact;
- AWS production deployment;
- causal real-time value of intraday recourse;
- CUDA/GPU training or inference qualification;
- Chronos-2 qualification;
- an external vulnerability-database audit.

GPU-ready PyTorch and optional Chronos pathways exist, but neither receives a performance or deployment claim without separate executed evidence.

## Historical document

The archive-era pre-GitHub report is preserved in [historical_pre_github_qualification.md](historical_pre_github_qualification.md). It is historical context only and must not be read as the current qualification scope.