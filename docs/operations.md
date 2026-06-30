# Operations Guide

## Supported execution contract

- Python: 3.11 through 3.13.
- Primary release runtime: Python 3.11.
- Canonical offline path: CPU-only synthetic smoke configuration.
- Optional paths: Chronos-2 and GPU execution; neither is required for the base install.
- External services: none for the canonical smoke.
- Runtime writes: only beneath the configured output directory and qualification report directories.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .[dev]
python -m pip check
```

PowerShell activation is `.\.venv\Scripts\Activate.ps1`. The repository does not require an
editable install, `PYTHONPATH`, a pre-existing model, a user-home configuration, or credentials.

## Canonical release checks

```bash
python scripts/qualify_local.py --profile CORE
python scripts/qualify_local.py --profile STANDARD
python scripts/qualify_local.py --profile EXTENDED
```

CORE covers dependency/runtime doctor checks, formatting/linting, isolated tests with coverage,
configuration validation, package build, distribution inspection, and installed-wheel E2E. STANDARD
adds repeated API and pipeline lifecycle checks plus the negative workforce gate. EXTENDED adds CI,
security, and regression-gate checks.

## Runtime entrypoints

```bash
support-capacity --version
support-capacity --help
support-capacity doctor --config configs/smoke.yaml
support-capacity run --config configs/smoke.yaml --require-release
support-capacity verify-output --output outputs/smoke
support-capacity verify-model-bundle --artifact-dir outputs/smoke/artifacts
```

Use `support-capacity --debug <command> ...` only when a traceback is required for diagnosis. Normal
operational failures return a non-zero exit code and a concise message without internal paths.

## API lifecycle

```bash
uvicorn support_capacity_reliability.api.app:app --host 127.0.0.1 --port 8000
```

- Liveness/readiness: `GET /health`.
- Staffing request: `POST /required-staffing`.
- Pipeline request: `POST /run-pipeline`.
- Graceful stop: SIGTERM or Ctrl+C.
- A second pipeline writer targeting the same output is rejected by an exclusive lock.

## Artifact publication

Each run writes into a unique staging directory. A completed staging tree is validated before the
existing output is moved to a backup and the new tree is renamed into place. If promotion fails, the
previous output is restored. Readers must call `verify-output` before trusting a published tree. Model
bundles are checksummed and are intended to be loaded only from a trusted pipeline output.

## Optional GPU/Chronos path

```bash
python -m pip install .[chronos]
support-capacity doctor --config configs/full.yaml
support-capacity run --config configs/full.yaml
```

The command fails explicitly when the optional dependency, weights, or requested device is not
available. CPU smoke success is not evidence of GPU or Chronos qualification.

## Logs and cleanup

- Pipeline: `outputs/<run>/stage.log`, `stage_timing.jsonl`, `run_manifest.json`.
- Qualification: `reports/qualification_logs/` and the selected JSON manifest.
- API smoke: `reports/api_server.log`.

```bash
make clean
```

## Optional PyTorch installation

The base install intentionally excludes PyTorch so CPU-only local setup remains bounded. Install `.[torch]` for `torch_quantile` or `.[full]` for the neural plus Chronos paths. On Linux, choose the CPU or CUDA wheel using the official PyTorch installation selector before installing the project extra when platform-specific control is required.

## Dependency reproducibility

Runtime dependencies use tested upper bounds to prevent unqualified major/minor upgrades. The exact Python 3.13 qualification set is recorded in `constraints/qualification-py313.txt`; use `python -m pip install -c constraints/qualification-py313.txt .` to reproduce the qualified environment.
