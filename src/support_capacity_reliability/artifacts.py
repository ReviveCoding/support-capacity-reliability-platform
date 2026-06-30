from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from support_capacity_reliability import __version__
from support_capacity_reliability.forecasting.base import ForecastOutput
from support_capacity_reliability.process_utils import (
    IsolatedCommandTimeout,
    run_isolated_command,
)
from support_capacity_reliability.reliability.calibration import IntervalCalibrator
from support_capacity_reliability.reliability.rcwe import ReferenceConditionedWorkloadEnvelope

MODEL_BUNDLE_SCHEMA_VERSION = "1.0"
RUN_ARTIFACT_SCHEMA_VERSION = "1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ForecastModelBundle:
    schema_version: str
    package_version: str
    selected_variant: str
    target: str
    feature_columns: list[str]
    state_features: list[str]
    lags: list[int]
    rolling_windows: list[int]
    model: Any
    calibrator: IntervalCalibrator
    rcwe: ReferenceConditionedWorkloadEnvelope | None
    created_at_utc: str

    def predict(self, frame: pd.DataFrame) -> ForecastOutput:
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(f"Model bundle input is missing required features: {preview}")
        base = self.model.predict(frame, self.feature_columns)
        uncalibrated = self.rcwe.transform(frame, base).forecast if self.rcwe is not None else base
        return self.calibrator.transform(uncalibrated)


def _verify_model_bundle_isolated(artifact_dir: Path) -> dict[str, Any]:
    code = (
        "import json, os, sys; "
        "from support_capacity_reliability.artifacts import verify_model_bundle; "
        f"result=verify_model_bundle({str(artifact_dir)!r}); "
        "sys.stdout.write(json.dumps(result)); sys.stdout.flush(); os._exit(0)"
    )
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        completed = run_isolated_command(
            [sys.executable, "-c", code],
            timeout_seconds=120,
            env=environment,
            terminate_group_on_success=True,
        )
    except IsolatedCommandTimeout as exc:
        raise RuntimeError("Isolated model-bundle verification timed out") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(f"Isolated model-bundle verification failed: {detail}")
    return json.loads(completed.stdout)


def persist_and_verify_model_bundle(
    *,
    output_dir: Path,
    selected_variant: str,
    target: str,
    feature_columns: list[str],
    state_features: list[str],
    lags: list[int],
    rolling_windows: list[int],
    model: Any,
    calibrator: IntervalCalibrator,
    rcwe: ReferenceConditionedWorkloadEnvelope | None,
    verification_frame: pd.DataFrame,
    expected_forecast: ForecastOutput,
) -> dict[str, Any]:
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = artifact_dir / "selected_forecast_bundle.joblib"
    manifest_path = artifact_dir / "selected_forecast_bundle_manifest.json"
    verification_input_path = artifact_dir / "selected_forecast_bundle_verification_input.joblib"
    verification_expected_path = artifact_dir / "selected_forecast_bundle_verification_expected.csv"

    bundle = ForecastModelBundle(
        schema_version=MODEL_BUNDLE_SCHEMA_VERSION,
        package_version=__version__,
        selected_variant=selected_variant,
        target=target,
        feature_columns=list(feature_columns),
        state_features=list(state_features),
        lags=list(lags),
        rolling_windows=list(rolling_windows),
        model=model,
        calibrator=calibrator,
        rcwe=rcwe,
        created_at_utc=datetime.now(UTC).isoformat(),
    )

    verification_columns = list(dict.fromkeys([*feature_columns, "region", "skill"]))
    missing_verification_columns = [
        column for column in verification_columns if column not in verification_frame.columns
    ]
    if missing_verification_columns:
        raise ValueError(
            "Bundle verification frame is missing columns: "
            + ", ".join(missing_verification_columns[:8])
        )
    joblib.dump(
        verification_frame[verification_columns].copy(), verification_input_path, compress=3
    )
    pd.DataFrame(
        {
            "q10": expected_forecast.q10,
            "q50": expected_forecast.q50,
            "q90": expected_forecast.q90,
        }
    ).to_csv(verification_expected_path, index=False)

    try:
        joblib.dump(bundle, bundle_path, compress=3)
    except Exception as exc:
        if selected_variant.split("+", 1)[0] == "chronos2":
            manifest = {
                "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
                "package_version": __version__,
                "selected_variant": selected_variant,
                "serialization_status": "EXTERNAL_MODEL_REFERENCE",
                "reason": f"{type(exc).__name__}: {exc}",
                "model_id": getattr(model, "model_id", None),
                "feature_count": len(feature_columns),
                "verification_input_path": str(verification_input_path.relative_to(output_dir)),
                "verification_expected_path": str(
                    verification_expected_path.relative_to(output_dir)
                ),
                "load_trust_boundary": "Only load model artifacts produced by a trusted pipeline run.",
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            return manifest
        raise RuntimeError(
            f"Selected model bundle serialization failed: {type(exc).__name__}: {exc}"
        ) from exc

    manifest = {
        "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
        "package_version": __version__,
        "selected_variant": selected_variant,
        "serialization_status": "VERIFIED",
        "bundle_path": str(bundle_path.relative_to(output_dir)),
        "bundle_bytes": bundle_path.stat().st_size,
        "bundle_sha256": _sha256(bundle_path),
        "feature_count": len(feature_columns),
        "state_feature_count": len(state_features),
        "verification_rows": len(verification_frame),
        "verification_max_abs_error": None,
        "verification_input_path": str(verification_input_path.relative_to(output_dir)),
        "verification_input_sha256": _sha256(verification_input_path),
        "verification_input_format": "joblib_dataframe_exact_v1",
        "verification_expected_path": str(verification_expected_path.relative_to(output_dir)),
        "verification_expected_sha256": _sha256(verification_expected_path),
        "load_trust_boundary": "Only load model artifacts produced by a trusted pipeline run.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    replay_report = _verify_model_bundle_isolated(artifact_dir)
    manifest["verification_max_abs_error"] = replay_report["maximum_absolute_error"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_run_artifacts(output_dir: Path) -> dict[str, Any]:
    required_files = [
        "run_summary.json",
        "run_manifest.json",
        "stage.log",
        "stage_timing.jsonl",
        "split_manifest.json",
        "metrics/validation_leaderboard.csv",
        "metrics/frozen_test_metrics.csv",
        "metrics/policy_comparison.csv",
        "metrics/decision_diagnostics.json",
        "metrics/scenario_diagnostics.json",
        "metrics/monitoring_snapshot.json",
        "reports/release_gate_decision.json",
        "reports/decision_memo.md",
        "artifacts/selected_forecast_bundle_manifest.json",
    ]
    missing = [relative for relative in required_files if not (output_dir / relative).is_file()]
    if missing:
        raise RuntimeError("Run artifact contract is missing files: " + ", ".join(missing))

    empty = [relative for relative in required_files if (output_dir / relative).stat().st_size == 0]
    if empty:
        raise RuntimeError("Run artifact contract contains empty files: " + ", ".join(empty))

    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    release = json.loads(
        (output_dir / "reports/release_gate_decision.json").read_text(encoding="utf-8")
    )
    bundle_manifest = json.loads(
        (output_dir / "artifacts/selected_forecast_bundle_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    if summary.get("release_status") != release.get("status"):
        raise RuntimeError("Run summary and release-gate status disagree")
    if manifest.get("summary_hash") is None:
        raise RuntimeError("Run manifest is missing summary_hash")
    from support_capacity_reliability.utils import stable_hash

    if manifest["summary_hash"] != stable_hash(summary):
        raise RuntimeError("Run manifest summary_hash does not match run_summary.json")
    if summary.get("selected_variant") != bundle_manifest.get("selected_variant"):
        raise RuntimeError("Selected variant and persisted model bundle disagree")
    serialization_status = bundle_manifest.get("serialization_status")
    if serialization_status not in {"VERIFIED", "EXTERNAL_MODEL_REFERENCE"}:
        raise RuntimeError(f"Unsupported model-bundle serialization status: {serialization_status}")
    if serialization_status == "VERIFIED":
        checksum_specs = [
            ("bundle_path", "bundle_sha256", "bundle"),
            ("verification_input_path", "verification_input_sha256", "verification input"),
            (
                "verification_expected_path",
                "verification_expected_sha256",
                "verification expected output",
            ),
        ]
        for path_key, checksum_key, label in checksum_specs:
            artifact_path = output_dir / str(bundle_manifest.get(path_key, ""))
            if not artifact_path.is_file():
                raise RuntimeError(f"Verified model-bundle manifest points to a missing {label}")
            if bundle_manifest.get(checksum_key) != _sha256(artifact_path):
                raise RuntimeError(
                    f"Persisted model-bundle {label} checksum does not match its manifest"
                )

    stages = [
        line.strip() for line in (output_dir / "stage.log").read_text().splitlines() if line.strip()
    ]
    expected_stages = [
        "01_configured",
        "02_data_generated",
        "03_data_saved",
        "04_features_split",
        "05_models_validated",
        "06_frozen_test_predicted",
        "07_offered_load_ablation",
        "08_slices_complete",
        "09_scenarios_generated",
        "10_capacity_planned",
        "11_policies_simulated",
        "12_release_gate",
        "13_before_reports",
        "14_complete",
    ]
    if stages != expected_stages:
        raise RuntimeError(f"Unexpected stage sequence: {stages}")

    file_entries: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in {"artifact_index.json", "run_manifest.json"}:
            continue
        file_entries.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    contract = {
        "schema_version": RUN_ARTIFACT_SCHEMA_VERSION,
        "status": "PASS",
        "release_status": summary["release_status"],
        "selected_variant": summary["selected_variant"],
        "file_count": len(file_entries),
        "files": file_entries,
    }
    (output_dir / "artifact_index.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "schema_version": RUN_ARTIFACT_SCHEMA_VERSION,
        "status": "PASS",
        "file_count": len(file_entries),
        "artifact_index_sha256": _sha256(output_dir / "artifact_index.json"),
    }


def verify_model_bundle(artifact_dir: str | Path) -> dict[str, Any]:
    root = Path(artifact_dir).expanduser().resolve()
    manifest_path = root / "selected_forecast_bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model-bundle manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = manifest.get("serialization_status")
    if status == "EXTERNAL_MODEL_REFERENCE":
        return {
            "status": "EXTERNAL_MODEL_REFERENCE",
            "selected_variant": manifest.get("selected_variant"),
            "reason": manifest.get("reason"),
        }
    if status != "VERIFIED":
        raise RuntimeError(f"Unsupported model-bundle status: {status}")

    output_root = root.parent
    bundle_path = output_root / str(manifest["bundle_path"])
    input_path = output_root / str(manifest["verification_input_path"])
    expected_path = output_root / str(manifest["verification_expected_path"])
    for path in [bundle_path, input_path, expected_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Model-bundle replay artifact is missing: {path}")
    if _sha256(bundle_path) != manifest["bundle_sha256"]:
        raise RuntimeError("Model-bundle checksum does not match manifest")
    if _sha256(input_path) != manifest["verification_input_sha256"]:
        raise RuntimeError("Model-bundle verification input checksum does not match manifest")
    if _sha256(expected_path) != manifest["verification_expected_sha256"]:
        raise RuntimeError("Model-bundle expected-output checksum does not match manifest")

    loaded = joblib.load(bundle_path)
    if not isinstance(loaded, ForecastModelBundle):
        raise TypeError("Loaded model artifact is not a ForecastModelBundle")
    if loaded.schema_version != MODEL_BUNDLE_SCHEMA_VERSION:
        raise RuntimeError("Model-bundle schema version is unsupported")
    if loaded.package_version != manifest.get("package_version"):
        raise RuntimeError("Model-bundle package version differs from its manifest")
    if input_path.suffix == ".joblib":
        input_frame = joblib.load(input_path)
        if not isinstance(input_frame, pd.DataFrame):
            raise TypeError("Model-bundle verification input is not a DataFrame")
    else:
        # Compatibility path for bundles emitted before exact binary verification inputs.
        input_frame = pd.read_csv(input_path)
    expected_frame = pd.read_csv(expected_path)
    replay = loaded.predict(input_frame)
    expected = expected_frame[["q10", "q50", "q90"]].to_numpy(float)
    observed = np.column_stack([replay.q10, replay.q50, replay.q90])
    if expected.shape != observed.shape or not np.allclose(
        expected, observed, rtol=1e-9, atol=1e-9
    ):
        maximum_error = (
            float(np.max(np.abs(expected - observed))) if expected.shape == observed.shape else None
        )
        raise RuntimeError(f"Portable model-bundle replay mismatch: max_abs_error={maximum_error}")
    return {
        "status": "PASS",
        "selected_variant": manifest["selected_variant"],
        "verification_rows": len(input_frame),
        "maximum_absolute_error": float(np.max(np.abs(expected - observed))),
        "bundle_sha256": manifest["bundle_sha256"],
    }


def verify_published_artifacts(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    index_path = root / "artifact_index.json"
    manifest_path = root / "run_manifest.json"
    summary_path = root / "run_summary.json"
    for path in [index_path, manifest_path, summary_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required published artifact is missing: {path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_index_hash = manifest.get("artifact_contract", {}).get("artifact_index_sha256")
    observed_index_hash = _sha256(index_path)
    if expected_index_hash != observed_index_hash:
        raise RuntimeError("Published artifact index checksum does not match run manifest")
    from support_capacity_reliability.utils import stable_hash

    if manifest.get("summary_hash") != stable_hash(summary):
        raise RuntimeError("Published run summary checksum does not match run manifest")
    if manifest.get("artifact_contract", {}).get("status") != "PASS":
        raise RuntimeError("Published run manifest does not contain a passing artifact contract")

    indexed_paths: set[str] = set()
    for entry in index.get("files", []):
        relative = str(entry["path"])
        indexed_paths.add(relative)
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"Indexed artifact is missing: {relative}")
        if path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"Indexed artifact size changed: {relative}")
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Indexed artifact checksum changed: {relative}")

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"artifact_index.json", "run_manifest.json"}
    }
    if actual_paths != indexed_paths:
        missing_from_index = sorted(actual_paths - indexed_paths)
        missing_from_tree = sorted(indexed_paths - actual_paths)
        raise RuntimeError(
            "Published artifact tree differs from its index: "
            f"unindexed={missing_from_index[:5]}, missing={missing_from_tree[:5]}"
        )
    if summary.get("release_status") != index.get("release_status"):
        raise RuntimeError("Published summary and artifact index release status disagree")
    if summary.get("selected_variant") != index.get("selected_variant"):
        raise RuntimeError("Published summary and artifact index selected variant disagree")

    return {
        "status": "PASS",
        "output_dir": str(root),
        "release_status": summary["release_status"],
        "selected_variant": summary["selected_variant"],
        "file_count": len(indexed_paths),
        "artifact_index_sha256": observed_index_hash,
    }
