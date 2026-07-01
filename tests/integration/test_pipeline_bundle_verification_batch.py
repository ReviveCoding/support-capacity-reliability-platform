from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from support_capacity_reliability.artifacts import verify_model_bundle
from support_capacity_reliability.pipeline import run_pipeline

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_persists_full_frozen_test_batch_for_bundle_replay(tmp_path: Path) -> None:
    source_config = REPOSITORY_ROOT / "configs" / "smoke.yaml"
    payload = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    output_dir = tmp_path / "full_batch_bundle_output"

    payload["project"]["output_dir"] = str(output_dir)
    payload["forecast"]["models"] = ["seasonal"]
    payload["rcwe"]["enabled"] = False

    config_path = tmp_path / "full_batch_bundle_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    run_pipeline(config_path)

    split_manifest = json.loads((output_dir / "split_manifest.json").read_text(encoding="utf-8"))
    test_rows = int(split_manifest["rows"]["test"])
    artifact_dir = output_dir / "artifacts"
    manifest = json.loads(
        (artifact_dir / "selected_forecast_bundle_manifest.json").read_text(encoding="utf-8")
    )

    verification_input = joblib.load(output_dir / manifest["verification_input_path"])
    verification_expected = joblib.load(output_dir / manifest["verification_expected_path"])

    assert isinstance(verification_input, pd.DataFrame)
    assert isinstance(verification_expected, pd.DataFrame)
    assert test_rows > 64
    assert len(verification_input) == test_rows
    assert len(verification_expected) == test_rows
    assert verify_model_bundle(artifact_dir)["status"] == "PASS"
