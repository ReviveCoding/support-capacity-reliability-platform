from pathlib import Path

import pytest
from pydantic import ValidationError

from support_capacity_reliability.config import AppConfig, load_config, resolve_config_path


def test_smoke_config_loads():
    config = load_config("configs/smoke.yaml")
    assert config.data.interval_minutes == 30
    assert "technical" in config.data.skills
    assert config.rcwe.enabled


def test_config_has_temporal_partitions():
    config = load_config("configs/smoke.yaml")
    assert config.forecast.train_fraction + config.forecast.calibration_fraction < 1


def test_invalid_temporal_split_is_rejected():
    config = load_config("configs/smoke.yaml")
    raw = config.model_dump()
    raw["forecast"]["train_fraction"] = 0.9
    raw["forecast"]["calibration_fraction"] = 0.2
    with pytest.raises(ValidationError, match="must be less than 1"):
        AppConfig.model_validate(raw)


def test_invalid_model_and_quantile_contracts_are_rejected():
    config = load_config("configs/smoke.yaml")
    raw = config.model_dump()
    raw["forecast"]["models"] = ["unknown_model"]
    with pytest.raises(ValidationError, match="unsupported forecast models"):
        AppConfig.model_validate(raw)

    raw = config.model_dump()
    raw["forecast"]["quantiles"] = [0.05, 0.5, 0.95]
    with pytest.raises(ValidationError, match="quantiles must be exactly"):
        AppConfig.model_validate(raw)


def test_chronos_model_requires_explicit_enablement():
    config = load_config("configs/smoke.yaml")
    raw = config.model_dump()
    raw["forecast"]["models"].append("chronos2")
    with pytest.raises(ValidationError, match="chronos.enabled"):
        AppConfig.model_validate(raw)


def test_cross_section_time_grid_and_horizon_contracts_are_rejected_early():
    config = load_config("configs/smoke.yaml")
    raw = config.model_dump()
    raw["data"]["interval_minutes"] = 37
    with pytest.raises(ValidationError, match="divide 1440"):
        AppConfig.model_validate(raw)

    raw = config.model_dump()
    raw["forecast"]["horizon_intervals"] = 1
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_optimization_cost_ordering_is_validated():
    config = load_config("configs/smoke.yaml")
    raw = config.model_dump()
    raw["optimization"]["overtime_hourly_cost"] = 20
    with pytest.raises(ValidationError, match="overtime_hourly_cost"):
        AppConfig.model_validate(raw)


def test_data_source_parameters_are_validated():
    config = load_config("configs/smoke.yaml")
    raw = config.model_dump()
    raw["data"]["observed_reach_rate"] = 0
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)

    raw = config.model_dump()
    raw["data"]["start_date"] = "not-a-date"
    with pytest.raises(ValidationError, match="start_date"):
        AppConfig.model_validate(raw)


def test_bundled_default_config_resolves_outside_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with resolve_config_path("configs/smoke.yaml") as resolved:
        assert resolved.is_file()
        config = load_config(resolved)
    assert config.project.name == "support-capacity-reliability"


def test_unknown_missing_config_does_not_silently_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        with resolve_config_path("configs/typo.yaml"):
            pass


def test_bundled_defaults_match_repository_configs():
    from importlib import resources

    for name in ["smoke.yaml", "stress_insufficient_workforce.yaml", "full.yaml"]:
        repository_text = (Path("configs") / name).read_text(encoding="utf-8")
        bundled_text = (
            resources.files("support_capacity_reliability")
            .joinpath("default_configs", name)
            .read_text(encoding="utf-8")
        )
        assert bundled_text == repository_text
