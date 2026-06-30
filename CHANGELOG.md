# Changelog

## 1.4.1rc2 - 2026-06-18

### Fixed

- Removed public CLI hard-exit behavior and isolated native-library shutdown workarounds inside bounded
  worker processes only.
- Added clean operational error messages, `--debug`, and `--version` CLI contracts.
- Replaced editable installation instructions and CI installs with normal non-editable installs.
- Added regression coverage for runtime-probe shutdown, pip-check timeout, version output, missing
  artifact errors, and debug traceback preservation.
- Declared the missing `tabulate` runtime dependency used by core Markdown reporting.
- Removed eager PyTorch imports from the base runtime, moved PyTorch to an explicit optional extra, and
  added a dedicated CPU-PyTorch CI job and `smoke_torch.yaml` contract.
- Added tested dependency upper bounds and an exact Python 3.13 qualification constraints file after
  unqualified NumPy/pandas/scikit-learn upgrades caused a release-gate regression.
- Made API request schemas reject unknown fields and corrected the API benchmark to treat the
  qualification-owned POSIX SIGTERM shutdown as an expected lifecycle result.

### Hardened

- Added cross-platform CORE/STANDARD/EXTENDED qualification runners and shell/PowerShell wrappers.
- Added Ubuntu Python 3.11-3.13 and Windows Python 3.11 CI coverage.
- Pinned third-party GitHub Actions to immutable commit SHAs and disabled persisted checkout credentials.
- Added CodeQL, Dependabot, local secret scanning, runner-environment evidence, and pip consistency gates.
- Added operations and troubleshooting documentation plus machine-readable release qualification outputs.

### Claim boundary

The canonical path remains an offline synthetic CPU validation. GitHub-hosted, Windows, Docker, GPU, and
Chronos claims require their own recorded execution evidence.
