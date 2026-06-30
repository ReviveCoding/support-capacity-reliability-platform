# Release Notes - 1.4.1rc2

This candidate is an operational-hardening release of the reliability-aware support forecasting, queueing
digital twin, and workforce optimization platform. It does not introduce a new scientific model or change
release thresholds.

The release focuses on clean installation, installed-wheel execution, deterministic local qualification,
process lifecycle safety, artifact verification, CI parity, and supply-chain configuration. The public CLI
now provides `--version`, concise non-debug failures, and an explicit debug traceback path. Native-library
shutdown handling is confined to trusted isolated probes rather than the public CLI process.

The base installation is now intentionally lightweight: PyTorch is an explicit optional extra, while the
canonical smoke/stress paths use the statistical and LightGBM stack. The release also declares the
previously missing `tabulate` reporting dependency, constrains runtime libraries to tested version lines,
rejects unknown API request fields, and records exact Python 3.13 qualification constraints. These are
operational reproducibility changes; release metrics and thresholds were not relaxed.

Use `qualification_manifest.json`, `release_bundle_manifest.json`, and
`docs/release_qualification.md` as the source of truth for the exact qualified source fingerprint, commands,
environments, checksums, unresolved items, and final verdict.
