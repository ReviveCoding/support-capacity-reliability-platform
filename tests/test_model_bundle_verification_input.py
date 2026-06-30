from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from support_capacity_reliability import __version__, artifacts
from support_capacity_reliability.artifacts import (
    ForecastModelBundle,
    persist_and_verify_model_bundle,
    verify_model_bundle,
)
from support_capacity_reliability.forecasting.base import ForecastOutput
from support_capacity_reliability.reliability.calibration import IntervalCalibrator


class ThresholdSensitiveModel:
    """Minimal deterministic model that magnifies an IEEE-754 branch change."""

    name = "threshold_sensitive"

    def predict(self, frame: pd.DataFrame, features: list[str]) -> ForecastOutput:
        values = frame[features[0]].to_numpy(dtype=float)
        median = np.where(values > 1.0, 10.0, 1.0)
        return ForecastOutput(
            model_name=self.name,
            q10=median - 0.5,
            q50=median,
            q90=median + 0.5,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forecast(frame: pd.DataFrame) -> ForecastOutput:
    return ThresholdSensitiveModel().predict(frame, ["signal"])


def _persist_bundle(
    tmp_path: Path,
    frame: pd.DataFrame,
    monkeypatch,
) -> tuple[Path, dict[str, object]]:
    monkeypatch.setattr(
        artifacts,
        "_verify_model_bundle_isolated",
        lambda artifact_dir: {
            "status": "PASS",
            "selected_variant": "threshold_sensitive",
            "verification_rows": len(frame),
            "maximum_absolute_error": 0.0,
            "bundle_sha256": "deferred_to_same_process_verification",
        },
    )
    expected = _forecast(frame)
    manifest = persist_and_verify_model_bundle(
        output_dir=tmp_path,
        selected_variant="threshold_sensitive",
        target="offered_load_estimate",
        feature_columns=["signal"],
        state_features=[],
        lags=[1],
        rolling_windows=[1],
        model=ThresholdSensitiveModel(),
        calibrator=IntervalCalibrator(),
        rcwe=None,
        verification_frame=frame,
        expected_forecast=expected,
    )
    return tmp_path / "artifacts", manifest


def test_persisted_bundle_uses_exact_binary_dataframe_and_replays(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "signal": np.array(
                [
                    np.nextafter(1.0, 2.0),
                    np.nextafter(1.0, 0.0),
                    2.0,
                ],
                dtype=float,
            ),
            "region": ["north", "south", "east"],
            "skill": ["billing", "technical", "fraud"],
        }
    )

    artifact_dir, manifest = _persist_bundle(tmp_path, frame, monkeypatch)
    input_path = artifact_dir / "selected_forecast_bundle_verification_input.joblib"

    assert manifest["verification_input_format"] == "joblib_dataframe_exact_v1"
    assert manifest["verification_input_path"] == str(input_path.relative_to(tmp_path))
    assert input_path.is_file()
    assert not (artifact_dir / "selected_forecast_bundle_verification_input.csv").exists()

    restored = joblib.load(input_path)
    assert isinstance(restored, pd.DataFrame)
    assert_frame_equal(restored, frame, check_exact=True, check_dtype=True)

    replay = verify_model_bundle(artifact_dir)

    assert replay["status"] == "PASS"
    assert replay["maximum_absolute_error"] == 0.0


def test_existing_csv_bundle_inputs_remain_verifiable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "signal": np.array([1.0, 2.0, 3.0], dtype=float),
            "region": ["north", "south", "east"],
            "skill": ["billing", "technical", "fraud"],
        }
    )

    artifact_dir, manifest = _persist_bundle(tmp_path, frame, monkeypatch)
    exact_input_path = artifact_dir / "selected_forecast_bundle_verification_input.joblib"
    legacy_input_path = artifact_dir / "legacy_verification_input.csv"
    frame.to_csv(legacy_input_path, index=False)

    manifest_path = artifact_dir / "selected_forecast_bundle_manifest.json"
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    persisted["verification_input_path"] = str(legacy_input_path.relative_to(tmp_path))
    persisted["verification_input_sha256"] = _sha256(legacy_input_path)
    persisted.pop("verification_input_format", None)
    manifest_path.write_text(json.dumps(persisted, indent=2) + "\n", encoding="utf-8")

    replay = verify_model_bundle(artifact_dir)

    assert replay["status"] == "PASS"
    assert exact_input_path.is_file()


def test_binary_input_contract_is_declared_in_manifest() -> None:
    bundle = ForecastModelBundle(
        schema_version="1.0",
        package_version=__version__,
        selected_variant="threshold_sensitive",
        target="offered_load_estimate",
        feature_columns=["signal"],
        state_features=[],
        lags=[1],
        rolling_windows=[1],
        model=ThresholdSensitiveModel(),
        calibrator=IntervalCalibrator(),
        rcwe=None,
        created_at_utc="2026-06-30T00:00:00+00:00",
    )

    assert bundle.schema_version == "1.0"
    assert bundle.selected_variant == "threshold_sensitive"
