from pathlib import Path

import pytest
import yaml

from support_capacity_reliability.artifacts import verify_published_artifacts
from support_capacity_reliability.pipeline import run_pipeline


def test_end_to_end_pipeline(tmp_path: Path):
    with open("configs/smoke.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["project"]["output_dir"] = str(tmp_path / "run")
    config["data"]["days"] = 20
    config["data"]["agents"] = 24
    config["data"]["event_days"] = [12, 13]
    config["forecast"]["models"] = ["seasonal", "poisson"]
    config["forecast"]["lags"] = [1, 2, 48, 96]
    config["forecast"]["rolling_windows"] = [6, 48]
    config["queue"]["replications"] = 2
    config["scenarios"]["count"] = 8
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = run_pipeline(path)
    assert summary["selected_variant"]
    assert summary["selected_policy_from_replay"]
    assert summary["deployed_policy"].endswith("+intraday_recourse")
    assert summary["recourse_remaining_hard_violations"] >= 0
    assert len(summary["reproducibility"]["config_hash"]) == 64
    assert len(summary["reproducibility"]["source_tree_hash"]) == 64
    recourse = __import__("pandas").read_csv(
        tmp_path / "run" / "metrics" / "intraday_recourse_actions.csv"
    )
    # Diagnostic unrepaired-undercoverage rows have a positive shortage amount,
    # but they do not represent an applied schedule-repair action.
    applied_action_rows = int(
        ((recourse["amount"] > 0) & (~recourse["action"].eq("unrepaired_undercoverage"))).sum()
    )
    assert summary["intraday_recourse_actions"] == applied_action_rows
    assert summary["intraday_recourse_decision_rows"] == len(recourse)
    assert (tmp_path / "run" / "reports" / "validation_report.md").exists()
    assert (tmp_path / "run" / "reports" / "model_selection_report.md").exists()
    assert (tmp_path / "run" / "reports" / "intraday_recourse_report.md").exists()
    assert (tmp_path / "run" / "metrics" / "selected_agent_schedule.csv").exists()
    assert (tmp_path / "run" / "metrics" / "decision_diagnostics.json").exists()
    assert (tmp_path / "run" / "metrics" / "intraday_recourse_actions.csv").exists()
    assert (tmp_path / "run" / "metrics" / "fixed_origin_predictions.csv").exists()
    assert (tmp_path / "run" / "metrics" / "policy_selection_replay.csv").exists()
    assert (tmp_path / "run" / "metrics" / "scenario_diagnostics.json").exists()
    assert (tmp_path / "run" / "metrics" / "offered_load_ablation.json").exists()
    assert (tmp_path / "run" / "metrics" / "scenario_aht_multipliers.npy").exists()
    assert (tmp_path / "run" / "metrics" / "scenario_patience_multipliers.npy").exists()
    assert (tmp_path / "run" / "metrics" / "scenario_shrinkage_rates.npy").exists()
    assert (tmp_path / "run" / "metrics" / "monitoring_snapshot.json").exists()
    assert (tmp_path / "run" / "metrics" / "monitoring_metrics.csv").exists()
    assert (tmp_path / "run" / "artifacts" / "selected_forecast_bundle.joblib").exists()
    assert (tmp_path / "run" / "artifacts" / "selected_forecast_bundle_manifest.json").exists()
    assert (tmp_path / "run" / "artifact_index.json").exists()
    assert summary["model_bundle"]["serialization_status"] == "VERIFIED"
    assert summary["monitoring"]["schema_version"] == "1.0"

    manifest = __import__("json").loads(
        (tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["config_hash"] == summary["reproducibility"]["config_hash"]
    assert manifest["source_tree_hash"] == summary["reproducibility"]["source_tree_hash"]
    assert manifest["artifact_contract"]["status"] == "PASS"
    assert len(manifest["artifact_contract"]["artifact_index_sha256"]) == 64
    verification = verify_published_artifacts(tmp_path / "run")
    assert verification["status"] == "PASS"
    monitored = tmp_path / "run" / "metrics" / "monitoring_snapshot.json"
    monitored.write_text(monitored.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="(size|checksum) changed"):
        verify_published_artifacts(tmp_path / "run")
