from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    name: str
    seed: int = 42
    output_dir: str = "outputs/smoke"


class DataConfig(StrictModel):
    days: int = Field(ge=14)
    interval_minutes: int = Field(default=30, gt=0)
    regions: list[str]
    skills: list[str]
    agents: int = Field(gt=0)
    start_date: str
    event_days: list[int] = Field(default_factory=list)
    redial_probability: float = Field(default=0.35, ge=0, le=1)
    recontact_probability: float = Field(default=0.08, ge=0, le=1)
    observed_reach_rate: float = Field(default=0.97, gt=0, le=1)
    save_event_level: bool = True

    @model_validator(mode="after")
    def validate_dimensions(self) -> DataConfig:
        if not self.regions or not self.skills:
            raise ValueError("regions and skills must be non-empty")
        if len(set(self.regions)) != len(self.regions):
            raise ValueError("regions must be unique")
        if len(set(self.skills)) != len(self.skills):
            raise ValueError("skills must be unique")
        if any(day < 0 or day >= self.days for day in self.event_days):
            raise ValueError("event_days must be within the generated day range")
        try:
            import pandas as pd

            pd.Timestamp(self.start_date)
        except (TypeError, ValueError) as exc:
            raise ValueError("start_date must be parseable as a timestamp") from exc
        return self


class TorchConfig(StrictModel):
    epochs: int = Field(default=8, ge=1)
    hidden_dim: int = Field(default=64, ge=8)
    batch_size: int = Field(default=256, ge=8)
    learning_rate: float = Field(default=0.003, gt=0)
    device: str = "auto"


class ChronosConfig(StrictModel):
    enabled: bool = False
    model_id: str = "amazon/chronos-2"
    context_length: int = Field(default=96, ge=8, le=8192)


class ForecastConfig(StrictModel):
    horizon_intervals: int = Field(default=48, ge=2)
    train_fraction: float = Field(default=0.65, gt=0, lt=1)
    calibration_fraction: float = Field(default=0.17, gt=0, lt=1)
    target: str = "offered_load_estimate"
    lags: list[int]
    rolling_windows: list[int]
    quantiles: list[float]
    models: list[str]
    torch: TorchConfig = Field(default_factory=TorchConfig)
    chronos: ChronosConfig = Field(default_factory=ChronosConfig)

    @model_validator(mode="after")
    def validate_forecast_contract(self) -> ForecastConfig:
        if self.train_fraction + self.calibration_fraction >= 1.0:
            raise ValueError("train_fraction + calibration_fraction must be less than 1")
        if not self.lags or any(lag <= 0 for lag in self.lags):
            raise ValueError("lags must contain positive integers")
        if len(set(self.lags)) != len(self.lags):
            raise ValueError("lags must be unique")
        if not self.rolling_windows or any(window <= 1 for window in self.rolling_windows):
            raise ValueError("rolling_windows must contain integers greater than 1")
        if len(set(self.rolling_windows)) != len(self.rolling_windows):
            raise ValueError("rolling_windows must be unique")
        expected_quantiles = [0.1, 0.5, 0.9]
        if len(self.quantiles) != 3 or any(
            abs(actual - expected) > 1e-9
            for actual, expected in zip(self.quantiles, expected_quantiles, strict=True)
        ):
            raise ValueError("quantiles must be exactly [0.1, 0.5, 0.9]")
        supported = {"seasonal", "poisson", "lightgbm", "torch_quantile", "chronos2"}
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("models must be a non-empty unique list")
        unknown = sorted(set(self.models) - supported)
        if unknown:
            raise ValueError(f"unsupported forecast models: {unknown}")
        if "chronos2" in self.models and not self.chronos.enabled:
            raise ValueError("chronos.enabled must be true when chronos2 is listed")
        return self


class RCWEConfig(StrictModel):
    enabled: bool = True
    neighbors: int = Field(default=12, ge=2)
    blend_strength: float = Field(default=0.45, ge=0, le=1)
    minimum_support: float = Field(default=0.2, ge=0, le=1)
    distance_temperature: float = Field(default=1.0, gt=0)
    low_support_interval_inflation: float = Field(default=1.3, ge=1)
    max_low_support_rate_for_selection: float = Field(default=0.75, ge=0, le=1)


class ScenarioConfig(StrictModel):
    count: int = Field(default=32, ge=4)
    correlation_shrinkage: float = Field(default=0.15, ge=0, le=1)


class QueueConfig(StrictModel):
    service_level_seconds: float = Field(default=120, gt=0)
    service_level_target: float = Field(default=0.8, ge=0, le=1)
    abandonment_target: float = Field(default=0.12, ge=0, le=1)
    max_agents_per_pool: int = Field(default=50, ge=1)
    simulation_hours: int = Field(default=12, ge=1)
    replications: int = Field(default=4, ge=2)
    policy_tuning_replications: int = Field(default=3, ge=2)
    staffing_load_quantile: float = Field(default=0.85, gt=0, lt=1)


class OptimizationConfig(StrictModel):
    regular_hourly_cost: float = Field(default=34, gt=0)
    overtime_hourly_cost: float = Field(default=52, gt=0)
    shortage_penalty: float = Field(default=180, gt=0)
    excess_capacity_penalty: float = Field(default=10, ge=0)
    schedule_preference_penalty: float = Field(default=2, ge=0)
    solver_time_limit_seconds: int = Field(default=15, ge=1)
    shrinkage_buffer: float = Field(default=0.12, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_cost_ordering(self) -> OptimizationConfig:
        if self.overtime_hourly_cost < self.regular_hourly_cost:
            raise ValueError("overtime_hourly_cost must be at least regular_hourly_cost")
        return self


class ReleaseGateConfig(StrictModel):
    max_interval_coverage_error: float = Field(default=0.15, ge=0)
    max_worst_slice_wape: float = Field(default=0.65, ge=0)
    min_slice_count_for_wape: int = Field(default=30, ge=1)
    min_peak_q90_coverage: float = Field(default=0.65, ge=0, le=1)
    min_incident_q90_coverage: float = Field(default=0.80, ge=0, le=1)
    min_peak_sample_count: int = Field(default=10, ge=1)
    min_incident_sample_count: int = Field(default=5, ge=1)
    min_schedule_feasibility: float = Field(default=1.0, ge=0, le=1)
    max_abandonment_rate: float = Field(default=0.25, ge=0, le=1)
    max_p95_wait_seconds: float = Field(default=900, ge=0)
    max_fixed_origin_wape: float = Field(default=0.65, ge=0)
    max_fixed_origin_coverage_error: float = Field(default=0.15, ge=0)
    max_scenario_coherence_error: float = Field(default=1e-8, ge=0)
    require_capacity_plan_success: bool = True
    min_strategic_tactical_alignment: float = Field(default=1.0, ge=0, le=1)
    max_recourse_action_rate: float = Field(default=0.40, ge=0, le=1)
    max_recourse_cost_share: float = Field(default=0.40, ge=0, le=1)
    max_hard_violations: int = Field(default=0, ge=0)


class AppConfig(StrictModel):
    project: ProjectConfig
    data: DataConfig
    forecast: ForecastConfig
    rcwe: RCWEConfig
    scenarios: ScenarioConfig
    queue: QueueConfig
    optimization: OptimizationConfig
    release_gate: ReleaseGateConfig

    @model_validator(mode="after")
    def validate_cross_section_contracts(self) -> AppConfig:
        minutes_per_day = 24 * 60
        if minutes_per_day % self.data.interval_minutes != 0:
            raise ValueError("interval_minutes must divide 1440 exactly")
        if (self.queue.simulation_hours * 60) % self.data.interval_minutes != 0:
            raise ValueError("queue.simulation_hours must contain an integer number of intervals")

        intervals_per_day = minutes_per_day // self.data.interval_minutes
        total_intervals = self.data.days * intervals_per_day
        history_requirement = max(
            max(self.forecast.lags),
            max(self.forecast.rolling_windows),
            48,
        )
        supervised_intervals = total_intervals - history_requirement
        test_fraction = 1.0 - self.forecast.train_fraction - self.forecast.calibration_fraction
        estimated_test_intervals = int(supervised_intervals * test_fraction)
        if supervised_intervals < 10 or estimated_test_intervals < 4:
            raise ValueError(
                "data horizon is too short for lag construction, temporal splits, and "
                "separate policy-tuning/frozen decision horizons"
            )
        return self


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return AppConfig.model_validate(raw)


@contextmanager
def resolve_config_path(path: str | Path) -> Iterator[Path]:
    """Resolve an explicit config path or a bundled default configuration.

    Explicit existing paths always win. Bundled fallback is intentionally limited to the
    three repository default filenames so misspelled custom paths fail rather than silently
    running a different configuration.
    """
    requested = Path(path)
    if requested.is_file():
        yield requested.resolve()
        return

    allowed_defaults = {
        "smoke.yaml",
        "stress_insufficient_workforce.yaml",
        "full.yaml",
    }
    if requested.name not in allowed_defaults or requested.parent not in {
        Path("."),
        Path("configs"),
    }:
        raise FileNotFoundError(f"Configuration file not found: {requested}")

    resource = resources.files("support_capacity_reliability").joinpath(
        "default_configs", requested.name
    )
    if not resource.is_file():
        raise FileNotFoundError(
            f"Configuration file not found and bundled default is unavailable: {requested}"
        )
    with resources.as_file(resource) as materialized:
        yield Path(materialized)
