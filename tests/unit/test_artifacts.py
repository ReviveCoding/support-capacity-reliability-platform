from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from support_capacity_reliability.artifacts import (
    ForecastModelBundle,
    persist_and_verify_model_bundle,
    validate_run_artifacts,
    verify_model_bundle,
)
from support_capacity_reliability.forecasting.models import SeasonalNaiveForecaster
from support_capacity_reliability.reliability.calibration import IntervalCalibrator
from support_capacity_reliability.utils import stable_hash


def _fitted_bundle_inputs():
    frame = pd.DataFrame(
        {
            "lag_48": np.arange(20, dtype=float),
            "region": ["north"] * 20,
            "skill": ["billing"] * 20,
            "target": np.arange(20, dtype=float),
        }
    )
    model = SeasonalNaiveForecaster(seasonal_lag=48).fit(frame, ["lag_48"], "target")
    base = model.predict(frame, ["lag_48"])
    calibrator = IntervalCalibrator().fit(frame["target"].to_numpy(float), base)
    expected = calibrator.transform(base)
    return frame, model, calibrator, expected


def test_model_bundle_round_trip_is_exact(tmp_path: Path):
    frame, model, calibrator, expected = _fitted_bundle_inputs()
    manifest = persist_and_verify_model_bundle(
        output_dir=tmp_path,
        selected_variant="seasonal",
        target="target",
        feature_columns=["lag_48"],
        state_features=["lag_48"],
        lags=[48],
        rolling_windows=[4],
        model=model,
        calibrator=calibrator,
        rcwe=None,
        verification_frame=frame,
        expected_forecast=expected,
    )
    assert manifest["serialization_status"] == "VERIFIED"
    assert manifest["verification_max_abs_error"] == 0.0
    assert (tmp_path / manifest["bundle_path"]).is_file()
    replay = verify_model_bundle(tmp_path / "artifacts")
    assert replay["status"] == "PASS"
    assert replay["maximum_absolute_error"] == 0.0


def test_model_bundle_rejects_missing_features():
    frame, model, calibrator, _ = _fitted_bundle_inputs()
    bundle = ForecastModelBundle(
        schema_version="1.0",
        package_version="test",
        selected_variant="seasonal",
        target="target",
        feature_columns=["lag_48"],
        state_features=["lag_48"],
        lags=[48],
        rolling_windows=[4],
        model=model,
        calibrator=calibrator,
        rcwe=None,
        created_at_utc="now",
    )
    with pytest.raises(ValueError, match="missing required features"):
        bundle.predict(frame.drop(columns=["lag_48"]))


def test_run_artifact_contract_rejects_summary_hash_mismatch(tmp_path: Path):
    required = {
        "stage.log": "\n".join(
            [
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
        )
        + "\n",
        "stage_timing.jsonl": '{"stage":"14_complete"}\n',
        "split_manifest.json": "{}\n",
        "metrics/validation_leaderboard.csv": "variant,status\nseasonal,OK\n",
        "metrics/frozen_test_metrics.csv": "variant,wape\nseasonal,0.1\n",
        "metrics/policy_comparison.csv": "policy,total_cost\nfixed,1\n",
        "metrics/decision_diagnostics.json": "{}\n",
        "metrics/scenario_diagnostics.json": "{}\n",
        "metrics/monitoring_snapshot.json": "{}\n",
        "reports/decision_memo.md": "ok\n",
    }
    for relative, content in required.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    summary = {"release_status": "PASS", "selected_variant": "seasonal"}
    (tmp_path / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"summary_hash": "wrong"}), encoding="utf-8"
    )
    (tmp_path / "reports/release_gate_decision.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    bundle_path = artifact_dir / "selected_forecast_bundle.joblib"
    bundle_path.write_bytes(b"bundle")
    import hashlib

    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    (artifact_dir / "selected_forecast_bundle_manifest.json").write_text(
        json.dumps(
            {
                "selected_variant": "seasonal",
                "serialization_status": "VERIFIED",
                "bundle_path": "artifacts/selected_forecast_bundle.joblib",
                "bundle_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    assert stable_hash(summary) != "wrong"
    with pytest.raises(RuntimeError, match="summary_hash"):
        validate_run_artifacts(tmp_path)
