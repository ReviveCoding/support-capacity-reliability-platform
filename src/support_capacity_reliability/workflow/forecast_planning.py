"""Forecast-planning helpers shared by the end-to-end orchestration pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from support_capacity_reliability.config import AppConfig
from support_capacity_reliability.data.features import (
    OPERATIONAL_COLUMNS,
    build_recursive_feature_rows,
)
from support_capacity_reliability.evaluation.metrics import forecast_metrics, slice_metrics
from support_capacity_reliability.forecasting.base import ForecastOutput
from support_capacity_reliability.queueing.erlang import required_agents_erlang_a
from support_capacity_reliability.reliability.calibration import IntervalCalibrator
from support_capacity_reliability.reliability.rcwe import ReferenceConditionedWorkloadEnvelope


class PlanningError(RuntimeError):
    """Raised when a planning forecast cannot be aligned or parameterized safely."""


def prediction_frame(source: pd.DataFrame, forecast: ForecastOutput, prefix: str) -> pd.DataFrame:
    out = source[["timestamp", "region", "skill", "regime"]].copy()
    out[f"{prefix}_q10"] = forecast.q10
    out[f"{prefix}_q50"] = forecast.q50
    out[f"{prefix}_q90"] = forecast.q90
    return out


def state_features(feature_columns: list[str]) -> list[str]:
    preferred = [
        column
        for column in feature_columns
        if column.startswith("lag_")
        or column.startswith("roll_mean_")
        or column.startswith("roll_std_")
        or "_lag_" in column
        or "_roll_mean_" in column
        or column in {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"}
    ]
    return preferred[: min(len(preferred), 18)]


def planning_parameter_table(
    historical_intervals: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """Return deployable operational assumptions without using realized future values."""
    history = historical_intervals.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    history["hour_slot"] = history["timestamp"].dt.hour * 2 + history["timestamp"].dt.minute // 30
    target = pd.Timestamp(timestamp)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")
    slot = target.hour * 2 + target.minute // 30
    seasonal = (
        history[history["hour_slot"] == slot]
        .groupby(["region", "skill"], as_index=False)[OPERATIONAL_COLUMNS]
        .median()
    )
    fallback = history.groupby(["region", "skill"], as_index=False)[OPERATIONAL_COLUMNS].median()
    merged = fallback.merge(
        seasonal,
        on=["region", "skill"],
        how="left",
        suffixes=("_fallback", ""),
    )
    for column in OPERATIONAL_COLUMNS:
        merged[column] = merged[column].fillna(merged[f"{column}_fallback"])
    return merged[["region", "skill", *OPERATIONAL_COLUMNS]]


def recursive_fixed_origin_forecast(
    *,
    historical_intervals: pd.DataFrame,
    future_timestamps: list[pd.Timestamp],
    model: Any,
    calibrator: IntervalCalibrator,
    rcwe: ReferenceConditionedWorkloadEnvelope | None,
    target: str,
    lags: list[int],
    rolling_windows: list[int],
    regions: list[str],
    skills: list[str],
) -> tuple[pd.DataFrame, ForecastOutput, np.ndarray, np.ndarray]:
    """Generate a leakage-safe fixed-origin recursive forecast for the decision horizon."""
    history = historical_intervals.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    output_frames: list[pd.DataFrame] = []
    q10_values: list[np.ndarray] = []
    q50_values: list[np.ndarray] = []
    q90_values: list[np.ndarray] = []
    support_values: list[np.ndarray] = []
    low_support_values: list[np.ndarray] = []

    for timestamp in future_timestamps:
        planning = planning_parameter_table(history, timestamp)
        feature_rows, recursive_features = build_recursive_feature_rows(
            history=history,
            timestamp=timestamp,
            target=target,
            lags=lags,
            rolling_windows=rolling_windows,
            regions=regions,
            skills=skills,
            planning_parameters=planning,
        )
        base = model.predict(feature_rows, recursive_features)
        if rcwe is not None:
            rcwe_output = rcwe.transform(feature_rows, base)
            uncalibrated = rcwe_output.forecast
            support = rcwe_output.support
            low_support = rcwe_output.low_support
        else:
            uncalibrated = base
            support = np.ones(len(feature_rows), dtype=float)
            low_support = np.zeros(len(feature_rows), dtype=bool)
        forecast = calibrator.transform(uncalibrated)

        step = feature_rows[["timestamp", "region", "skill", "regime"]].copy()
        for column in OPERATIONAL_COLUMNS:
            step[column] = feature_rows[f"planning_{column}"].to_numpy(float)
        step[target] = forecast.q50
        output_frames.append(step)
        q10_values.append(forecast.q10)
        q50_values.append(forecast.q50)
        q90_values.append(forecast.q90)
        support_values.append(support)
        low_support_values.append(low_support)

        history = pd.concat(
            [
                history,
                step[["timestamp", "region", "skill", "regime", target, *OPERATIONAL_COLUMNS]],
            ],
            ignore_index=True,
            sort=False,
        )

    forecast_frame = pd.concat(output_frames, ignore_index=True)
    combined = ForecastOutput(
        f"{getattr(model, 'name', 'model')}:fixed_origin_recursive",
        np.concatenate(q10_values),
        np.concatenate(q50_values),
        np.concatenate(q90_values),
    )
    return (
        forecast_frame,
        combined,
        np.concatenate(support_values),
        np.concatenate(low_support_values),
    )


def effective_lags(config: AppConfig) -> list[int]:
    lags = set(config.forecast.lags)
    if config.forecast.chronos.enabled and "chronos2" in config.forecast.models:
        lags.update(range(1, config.forecast.chronos.context_length + 1))
    return sorted(lags)


def prepare_decision_horizon(
    *,
    intervals: pd.DataFrame,
    future_timestamps: list[pd.Timestamp],
    model: Any,
    calibrator: IntervalCalibrator,
    rcwe: ReferenceConditionedWorkloadEnvelope | None,
    config: AppConfig,
) -> tuple[pd.DataFrame, ForecastOutput, dict[str, float], np.ndarray, np.ndarray, pd.Timestamp]:
    decision_origin = future_timestamps[0] - pd.Timedelta(minutes=config.data.interval_minutes)
    historical_intervals = intervals[
        pd.to_datetime(intervals["timestamp"], utc=True) <= decision_origin
    ].copy()
    recursive_frame, forecast, support, low_support = recursive_fixed_origin_forecast(
        historical_intervals=historical_intervals,
        future_timestamps=future_timestamps,
        model=model,
        calibrator=calibrator,
        rcwe=rcwe,
        target=config.forecast.target,
        lags=effective_lags(config),
        rolling_windows=config.forecast.rolling_windows,
        regions=config.data.regions,
        skills=config.data.skills,
    )
    horizon = intervals[
        pd.to_datetime(intervals["timestamp"], utc=True).isin(future_timestamps)
    ].copy()
    horizon = horizon.sort_values(["timestamp", "region", "skill"]).reset_index(drop=True)
    recursive_frame = recursive_frame.sort_values(["timestamp", "region", "skill"]).reset_index(
        drop=True
    )
    if len(horizon) != len(recursive_frame):
        raise PlanningError(
            "Fixed-origin forecast could not be aligned to the realized decision horizon"
        )
    for column in OPERATIONAL_COLUMNS:
        horizon[f"planning_{column}"] = recursive_frame[column].to_numpy(float)
    metrics = forecast_metrics(horizon[config.forecast.target].to_numpy(float), forecast)
    return horizon, forecast, metrics, support, low_support, decision_origin


def worst_supported_slice_wape(
    frame: pd.DataFrame,
    target: str,
    forecast: ForecastOutput,
    minimum_count: int,
) -> tuple[float, pd.DataFrame]:
    report = slice_metrics(frame, target, forecast).copy()
    report["supported_for_wape_gate"] = report["count"] >= minimum_count
    supported = report[report["supported_for_wape_gate"]]
    candidate = supported if not supported.empty else report
    return float(candidate["wape"].max()), report


def model_score(metrics: dict[str, float], worst_slice_wape: float) -> float:
    return (
        metrics["wape"]
        + 0.15 * worst_slice_wape
        + 0.35 * metrics["coverage_error"]
        + 0.01 * metrics["mean_interval_width"]
    )


def build_rcwe(config: AppConfig) -> ReferenceConditionedWorkloadEnvelope:
    settings = config.rcwe
    return ReferenceConditionedWorkloadEnvelope(
        neighbors=settings.neighbors,
        blend_strength=settings.blend_strength,
        minimum_support=settings.minimum_support,
        distance_temperature=settings.distance_temperature,
        low_support_interval_inflation=settings.low_support_interval_inflation,
    )


def capacity_requirements_from_scenarios(
    scenario_samples: np.ndarray,
    leaf_keys: list[str],
    horizon: pd.DataFrame,
    skills: list[str],
    interval_minutes: int,
    queue_settings: Any,
    shrinkage_buffer: float,
    staffing_load_quantile: float,
    *,
    operational_skill_keys: list[str],
    aht_multipliers: np.ndarray,
    patience_multipliers: np.ndarray,
    shrinkage_rates: np.ndarray,
) -> np.ndarray:
    """Convert joint load and operational scenarios into skill-level staffing needs."""
    scenario_count = scenario_samples.shape[0]
    skill_index = {skill: idx for idx, skill in enumerate(skills)}
    operational_skill_index = {skill: idx for idx, skill in enumerate(operational_skill_keys)}
    key_skill = [key.split("::", 1)[1] for key in leaf_keys]
    sorted_times = [
        pd.Timestamp(value)
        for value in sorted(pd.to_datetime(horizon["timestamp"], utc=True).unique())
    ]
    base_parameters = (
        horizon.assign(timestamp=pd.to_datetime(horizon["timestamp"], utc=True))
        .groupby(["timestamp", "skill"], as_index=False)
        .agg(
            planning_average_handle_time_seconds=(
                "planning_average_handle_time_seconds",
                "mean",
            ),
            planning_patience_mean_seconds=("planning_patience_mean_seconds", "mean"),
            planning_shrinkage_rate=("planning_shrinkage_rate", "mean"),
        )
    )
    result = np.zeros((scenario_count, len(skills)), dtype=float)
    for scenario_id in range(scenario_count):
        for skill in skills:
            leaf_ids = [idx for idx, leaf_skill in enumerate(key_skill) if leaf_skill == skill]
            if not leaf_ids:
                continue
            op_idx = operational_skill_index[skill]
            contacts_by_interval = scenario_samples[scenario_id][:, leaf_ids].sum(axis=1)
            skill_base = (
                base_parameters[base_parameters["skill"].astype(str) == skill]
                .set_index("timestamp")
                .reindex(sorted_times)
            )
            if skill_base.isna().any().any():
                raise PlanningError(
                    f"Operational planning parameters are incomplete for skill={skill}"
                )
            scenario_aht = (
                skill_base["planning_average_handle_time_seconds"].to_numpy(float)
                * aht_multipliers[scenario_id, :, op_idx]
            )
            scenario_patience = (
                skill_base["planning_patience_mean_seconds"].to_numpy(float)
                * patience_multipliers[scenario_id, :, op_idx]
            )
            scenario_shrinkage = shrinkage_rates[scenario_id, :, op_idx]
            contacts_per_interval = float(np.quantile(contacts_by_interval, staffing_load_quantile))
            arrival_rate = contacts_per_interval / (interval_minutes * 60.0)
            approximation = required_agents_erlang_a(
                arrival_rate_per_second=arrival_rate,
                average_handle_time_seconds=float(
                    np.quantile(scenario_aht, staffing_load_quantile)
                ),
                patience_mean_seconds=float(
                    np.quantile(scenario_patience, 1.0 - staffing_load_quantile)
                ),
                service_level_target=queue_settings.service_level_target,
                abandonment_target=queue_settings.abandonment_target,
                service_level_seconds=queue_settings.service_level_seconds,
                max_agents=queue_settings.max_agents_per_pool,
            )
            planned_shrinkage = float(np.quantile(scenario_shrinkage, staffing_load_quantile))
            effective_availability = max(
                (1.0 - planned_shrinkage) * (1.0 - shrinkage_buffer),
                0.1,
            )
            result[scenario_id, skill_index[skill]] = np.ceil(
                approximation.agents / effective_availability
            )
    return result
