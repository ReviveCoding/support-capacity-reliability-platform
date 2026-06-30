# Troubleshooting

## `doctor` reports a missing dependency

Install from the repository source of truth and rerun dependency consistency checks:

```bash
python -m pip install .
python -m pip check
support-capacity doctor --config configs/smoke.yaml
```

The base dependency set is CPU-capable. Chronos is optional and should be installed only for the full
configuration.

## A command appears to finish but the process does not exit

The doctor and persisted-bundle checks isolate native numerical libraries in bounded worker process
groups. Use the current release rather than adding shell-level force termination. Capture debug evidence
with `support-capacity --debug ...`; do not replace normal CLI shutdown with `os._exit`.

## `temporary failure` / exit code 75

Another process owns the output lock. Confirm that a legitimate run is active. Stale locks owned by a
dead process are recovered by the pipeline; an active writer must not be bypassed.

## `verification error` or checksum mismatch

Do not use the output. Preserve it for diagnosis, run `verify-output`, and regenerate the artifact from a
clean pipeline execution. A failed verification is never converted into a normal forecast response.

## Canonical smoke returns `ITERATE`

`configs/smoke.yaml` is required to produce `PASS` or `PASS_WITH_RECOURSE`. Inspect
`outputs/smoke/reports/release_gate_decision.json` and `run_summary.json`. Do not change thresholds or
test data merely to force a pass. The stress configuration is intentionally expected to return `ITERATE`.

## Port conflict

Choose an unused local port:

```bash
uvicorn support_capacity_reliability.api.app:app --host 127.0.0.1 --port 8010
```

A port conflict should fail startup; it is not an application-health success.

## Installed wheel works only inside the repository

This is a release blocker. Rebuild and run the installed-wheel smoke from a directory outside the source
tree:

```bash
python -m build
python scripts/verify_distribution.py
python scripts/smoke_installed_wheel.py --run-pipeline --run-api-pipeline
```

## Windows notes

Use Python 3.11 and PowerShell. Run `scripts\qualify_local.ps1`. Make is not required by the portable
runner. Windows is qualified only when an actual Windows run is recorded; workflow configuration alone
is not execution evidence.

## Optional PyTorch installation

The base install intentionally excludes PyTorch so CPU-only local setup remains bounded. Install `.[torch]` for `torch_quantile` or `.[full]` for the neural plus Chronos paths. On Linux, choose the CPU or CUDA wheel using the official PyTorch installation selector before installing the project extra when platform-specific control is required.

## Dependency reproducibility

Runtime dependencies use tested upper bounds to prevent unqualified major/minor upgrades. The exact Python 3.13 qualification set is recorded in `constraints/qualification-py313.txt`; use `python -m pip install -c constraints/qualification-py313.txt .` to reproduce the qualified environment.
