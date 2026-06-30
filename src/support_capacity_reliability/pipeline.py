from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from support_capacity_reliability.artifacts import (
    persist_and_verify_model_bundle,
    validate_run_artifacts,
)
from support_capacity_reliability.config import AppConfig, load_config
from support_capacity_reliability.data.contracts import validate_agents, validate_intervals
from support_capacity_reliability.data.features import (
    OPERATIONAL_COLUMNS,
    build_supervised_frame,
    temporal_split,
)
from support_capacity_reliability.data.synthetic import generate_synthetic_acd
from support_capacity_reliability.evaluation.decision import evaluate_existing_schedule
from support_capacity_reliability.evaluation.metrics import forecast_metrics
from support_capacity_reliability.forecasting.base import ForecastOutput
from support_capacity_reliability.forecasting.chronos_adapter import Chronos2Adapter
from support_capacity_reliability.forecasting.factory import build_model
from support_capacity_reliability.monitoring import build_monitoring_snapshot
from support_capacity_reliability.optimization.capacity import StrategicCapacityPlanner
from support_capacity_reliability.optimization.recourse import apply_intraday_recourse
from support_capacity_reliability.reliability.calibration import IntervalCalibrator
from support_capacity_reliability.reliability.rcwe import ReferenceConditionedWorkloadEnvelope
from support_capacity_reliability.reliability.release_gate import evaluate_release_gate
from support_capacity_reliability.reliability.scenarios import generate_coherent_scenarios
from support_capacity_reliability.reporting import write_reports
from support_capacity_reliability.utils import ensure_dir, seed_everything, stable_hash, write_json
from support_capacity_reliability.workflow.forecast_planning import (
    build_rcwe as _build_rcwe,
)
from support_capacity_reliability.workflow.forecast_planning import (
    capacity_requirements_from_scenarios as _capacity_requirements_from_scenarios,
)
from support_capacity_reliability.workflow.forecast_planning import (
    effective_lags as _effective_lags,
)
from support_capacity_reliability.workflow.forecast_planning import (
    model_score as _score,
)
from support_capacity_reliability.workflow.forecast_planning import (
    prediction_frame as _prediction_frame,
)
from support_capacity_reliability.workflow.forecast_planning import (
    prepare_decision_horizon as _prepare_decision_horizon,
)
from support_capacity_reliability.workflow.forecast_planning import (
    state_features as _state_features,
)
from support_capacity_reliability.workflow.forecast_planning import (
    worst_supported_slice_wape as _worst_supported_slice_wape,
)
from support_capacity_reliability.workflow.policy_selection import (
    evaluate_policy_candidates,
    evaluate_recourse_aware_policy_candidates,
    select_policy_from_replay,
)


class PipelineError(RuntimeError):
    pass


class PipelineBusyError(PipelineError):
    pass


_RUN_STARTS: dict[str, float] = {}


def _mark(output_dir: Path, stage: str) -> None:
    key = str(output_dir.resolve())
    if stage == "01_configured" or key not in _RUN_STARTS:
        _RUN_STARTS[key] = time.perf_counter()
    elapsed = time.perf_counter() - _RUN_STARTS[key]
    path = output_dir / "stage.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(stage + "\n")
        handle.flush()
    timing_path = output_dir / "stage_timing.jsonl"
    with timing_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": stage,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "elapsed_seconds": round(elapsed, 6),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
    if stage == "14_complete":
        _RUN_STARTS.pop(key, None)


def _run_pipeline_impl(config: AppConfig) -> dict[str, Any]:
    seed_everything(config.project.seed)
    effective_lags = _effective_lags(config)
    output_dir = ensure_dir(config.project.output_dir)
    ensure_dir(output_dir / "data")
    ensure_dir(output_dir / "metrics")
    ensure_dir(output_dir / "models")
    _mark(output_dir, "01_configured")

    bundle = generate_synthetic_acd(config.data, seed=config.project.seed)
    _mark(output_dir, "02_data_generated")
    interval_contract = validate_intervals(bundle.intervals, target_column=config.forecast.target)
    agent_contract = validate_agents(bundle.agents, allowed_skills=config.data.skills)
    if not interval_contract.passed or not agent_contract.passed:
        raise PipelineError(
            f"Data contract failed: intervals={interval_contract.errors}; agents={agent_contract.errors}"
        )
    bundle.intervals.to_csv(output_dir / "data" / "intervals.csv", index=False)
    bundle.agents.to_csv(output_dir / "data" / "agents.csv", index=False)
    bundle.events.to_csv(output_dir / "data" / "events.csv", index=False)
    if config.data.save_event_level:
        bundle.contacts.to_csv(output_dir / "data" / "contacts.csv", index=False)

    _mark(output_dir, "03_data_saved")
    supervised, feature_columns = build_supervised_frame(
        bundle.intervals,
        config.forecast.target,
        effective_lags,
        config.forecast.rolling_windows,
    )
    train, calibration, test = temporal_split(
        supervised,
        config.forecast.train_fraction,
        config.forecast.calibration_fraction,
    )
    split_manifest = {
        "train_start": str(train["timestamp"].min()),
        "train_end": str(train["timestamp"].max()),
        "calibration_start": str(calibration["timestamp"].min()),
        "calibration_end": str(calibration["timestamp"].max()),
        "test_start": str(test["timestamp"].min()),
        "test_end": str(test["timestamp"].max()),
        "rows": {"train": len(train), "calibration": len(calibration), "test": len(test)},
        "feature_count": len(feature_columns),
        "manifest_hash": stable_hash(feature_columns),
    }
    write_json(output_dir / "split_manifest.json", split_manifest)

    _mark(output_dir, "04_features_split")
    state_features = _state_features(feature_columns)
    validation_rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    calibration_forecasts: dict[str, ForecastOutput] = {}
    calibrators: dict[str, IntervalCalibrator] = {}
    rcwe_objects: dict[str, ReferenceConditionedWorkloadEnvelope] = {}
    hardware_reports: dict[str, Any] = {}
    calibration_peak_threshold = float(
        np.quantile(train[config.forecast.target].to_numpy(float), 0.90)
    )
    calibration_peak_mask = (
        calibration[config.forecast.target].to_numpy(float) >= calibration_peak_threshold
    )
    shared_rcwe = (
        _build_rcwe(config).fit(train, state_features, config.forecast.target)
        if config.rcwe.enabled
        else None
    )

    model_names = list(config.forecast.models)
    for model_name in model_names:
        if model_name == "chronos2" and not config.forecast.chronos.enabled:
            continue
        if model_name == "chronos2":
            status = Chronos2Adapter.availability()
            if not status.available:
                raise PipelineError(
                    "Chronos-2 is enabled but unavailable. Install `.[chronos]`, cache the "
                    f"configured model, and rerun. Availability check: {status.reason}"
                )
        model = build_model(model_name, config)
        try:
            model.fit(train, feature_columns, config.forecast.target)
            base_calibration = model.predict(calibration, feature_columns)
        except Exception as exc:
            validation_rows.append(
                {
                    "variant": model_name,
                    "status": "FAILED",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        fitted[model_name] = model
        base_metrics = forecast_metrics(
            calibration[config.forecast.target].to_numpy(float), base_calibration
        )
        base_worst_slice_wape, _ = _worst_supported_slice_wape(
            calibration,
            config.forecast.target,
            base_calibration,
            config.release_gate.min_slice_count_for_wape,
        )
        validation_rows.append(
            {
                "variant": model_name,
                "status": "OK",
                "eligible_for_selection": True,
                "worst_regime_wape": base_worst_slice_wape,
                "score": _score(base_metrics, base_worst_slice_wape),
                **base_metrics,
            }
        )
        calibration_forecasts[model_name] = base_calibration
        calibrator = IntervalCalibrator(target_coverage=0.8).fit(
            calibration[config.forecast.target].to_numpy(float),
            base_calibration,
            peak_mask=calibration_peak_mask,
            peak_activation_threshold=calibration_peak_threshold,
        )
        calibrators[model_name] = calibrator
        if model_name == "torch_quantile" and hasattr(model, "hardware_summary"):
            hardware_reports[model_name] = model.hardware_summary()

        if config.rcwe.enabled:
            assert shared_rcwe is not None
            rcwe = shared_rcwe
            rcwe_output = rcwe.transform(calibration, base_calibration)
            variant = f"{model_name}+rcwe"
            metrics = forecast_metrics(
                calibration[config.forecast.target].to_numpy(float), rcwe_output.forecast
            )
            rcwe_worst_slice_wape, _ = _worst_supported_slice_wape(
                calibration,
                config.forecast.target,
                rcwe_output.forecast,
                config.release_gate.min_slice_count_for_wape,
            )
            low_support_rate = float(rcwe_output.low_support.mean())
            validation_rows.append(
                {
                    "variant": variant,
                    "status": "OK",
                    "eligible_for_selection": bool(
                        low_support_rate <= config.rcwe.max_low_support_rate_for_selection
                    ),
                    "score": _score(metrics, rcwe_worst_slice_wape),
                    "worst_regime_wape": rcwe_worst_slice_wape,
                    "mean_reference_support": float(rcwe_output.support.mean()),
                    "low_support_rate": low_support_rate,
                    "support_selection_threshold": (config.rcwe.max_low_support_rate_for_selection),
                    **metrics,
                }
            )
            calibration_forecasts[variant] = rcwe_output.forecast
            calibrators[variant] = IntervalCalibrator(target_coverage=0.8).fit(
                calibration[config.forecast.target].to_numpy(float),
                rcwe_output.forecast,
                peak_mask=calibration_peak_mask,
                peak_activation_threshold=calibration_peak_threshold,
            )
            rcwe_objects[model_name] = rcwe

    _mark(output_dir, "05_models_validated")
    validation_leaderboard = pd.DataFrame(validation_rows)
    valid_rows = validation_leaderboard[
        (validation_leaderboard["status"] == "OK")
        & validation_leaderboard["eligible_for_selection"].fillna(False).astype(bool)
    ].copy()
    if valid_rows.empty:
        raise PipelineError("No forecasting model completed successfully")
    selected_variant = str(valid_rows.sort_values("score").iloc[0]["variant"])
    selected_base_name = selected_variant.split("+", 1)[0]
    selected_model = fitted[selected_base_name]
    base_test = selected_model.predict(test, feature_columns)
    if "+rcwe" in selected_variant:
        selected_rcwe_output = rcwe_objects[selected_base_name].transform(test, base_test)
        selected_uncalibrated = selected_rcwe_output.forecast
        rcwe_support = selected_rcwe_output.support
        rcwe_low_support = selected_rcwe_output.low_support
    else:
        selected_uncalibrated = base_test
        rcwe_support = np.ones(len(test), dtype=float)
        rcwe_low_support = np.zeros(len(test), dtype=bool)
    selected_test = calibrators[selected_variant].transform(selected_uncalibrated)
    bundle_verification_rows = min(64, len(test))
    expected_bundle_forecast = ForecastOutput(
        selected_test.model_name,
        selected_test.q10[:bundle_verification_rows],
        selected_test.q50[:bundle_verification_rows],
        selected_test.q90[:bundle_verification_rows],
    )
    model_bundle_manifest = persist_and_verify_model_bundle(
        output_dir=output_dir,
        selected_variant=selected_variant,
        target=config.forecast.target,
        feature_columns=feature_columns,
        state_features=state_features,
        lags=effective_lags,
        rolling_windows=config.forecast.rolling_windows,
        model=selected_model,
        calibrator=calibrators[selected_variant],
        rcwe=rcwe_objects.get(selected_base_name) if "+rcwe" in selected_variant else None,
        verification_frame=test.iloc[:bundle_verification_rows].copy(),
        expected_forecast=expected_bundle_forecast,
    )

    _mark(output_dir, "06_frozen_test_predicted")
    # Offered-load controlled study: train the same selected model family on served contacts.
    served_frame, served_features = build_supervised_frame(
        bundle.intervals,
        "observed_served",
        effective_lags,
        config.forecast.rolling_windows,
    )
    served_train, served_calibration, served_test = temporal_split(
        served_frame,
        config.forecast.train_fraction,
        config.forecast.calibration_fraction,
    )
    served_model = build_model(selected_base_name, config)
    served_model.fit(served_train, served_features, "observed_served")
    served_prediction_base = served_model.predict(served_test, served_features)
    served_calibration_base = served_model.predict(served_calibration, served_features)
    served_variant = selected_base_name
    if "+rcwe" in selected_variant:
        # Keep the reliability stack identical across the served-target and offered-load
        # models so the ablation isolates target construction instead of RCWE usage.
        served_rcwe = _build_rcwe(config).fit(
            served_train,
            _state_features(served_features),
            "observed_served",
        )
        served_prediction_uncalibrated = served_rcwe.transform(
            served_test, served_prediction_base
        ).forecast
        served_calibration_uncalibrated = served_rcwe.transform(
            served_calibration, served_calibration_base
        ).forecast
        served_variant += "+rcwe"
    else:
        served_prediction_uncalibrated = served_prediction_base
        served_calibration_uncalibrated = served_calibration_base
    served_peak_threshold = float(
        np.quantile(served_train["observed_served"].to_numpy(float), 0.90)
    )
    served_peak_mask = (
        served_calibration["observed_served"].to_numpy(float) >= served_peak_threshold
    )
    served_prediction = (
        IntervalCalibrator(0.8)
        .fit(
            served_calibration["observed_served"].to_numpy(float),
            served_calibration_uncalibrated,
            peak_mask=served_peak_mask,
            peak_activation_threshold=served_peak_threshold,
        )
        .transform(served_prediction_uncalibrated)
    )

    test_metrics_rows: list[dict[str, Any]] = []
    selected_metrics = forecast_metrics(test[config.forecast.target].to_numpy(float), selected_test)
    test_metrics_rows.append(
        {
            "variant": f"{selected_variant}:calibrated",
            "evaluation_target": config.forecast.target,
            **selected_metrics,
        }
    )
    base_metrics_test = forecast_metrics(test[config.forecast.target].to_numpy(float), base_test)
    test_metrics_rows.append(
        {
            "variant": f"{selected_base_name}:uncalibrated",
            "evaluation_target": config.forecast.target,
            **base_metrics_test,
        }
    )
    latent_served_metrics = forecast_metrics(
        served_test["latent_demand"].to_numpy(float), served_prediction
    )
    test_metrics_rows.append(
        {
            "variant": f"{served_variant}:served_target",
            "evaluation_target": "latent_demand",
            **latent_served_metrics,
        }
    )
    latent_corrected_metrics = forecast_metrics(
        test["latent_demand"].to_numpy(float), selected_test
    )
    test_metrics_rows.append(
        {
            "variant": f"{selected_variant}:offered_load_corrected",
            "evaluation_target": "latent_demand",
            **latent_corrected_metrics,
        }
    )
    test_metrics = pd.DataFrame(test_metrics_rows)
    served_latent_bias = float(
        np.mean(served_prediction.q50 - served_test["latent_demand"].to_numpy(float))
    )
    corrected_latent_bias = float(
        np.mean(selected_test.q50 - test["latent_demand"].to_numpy(float))
    )
    write_json(
        output_dir / "metrics" / "offered_load_ablation.json",
        {
            "served_variant": served_variant,
            "offered_load_variant": selected_variant,
            "reliability_stack_matched": bool(
                ("+rcwe" in served_variant) == ("+rcwe" in selected_variant)
            ),
            "served_target_latent_wape": latent_served_metrics["wape"],
            "offered_load_latent_wape": latent_corrected_metrics["wape"],
            "served_target_latent_bias": served_latent_bias,
            "offered_load_latent_bias": corrected_latent_bias,
            "absolute_bias_reduction": abs(served_latent_bias) - abs(corrected_latent_bias),
            "claim_boundary": (
                "controlled synthetic ground-truth study; not an estimate of real latent "
                "customer demand"
            ),
        },
    )
    _mark(output_dir, "07_offered_load_ablation")

    test_prediction = _prediction_frame(test, selected_test, "selected")
    test_prediction["rcwe_support"] = rcwe_support
    test_prediction["rcwe_low_support"] = rcwe_low_support.astype(int)
    test_prediction[config.forecast.target] = test[config.forecast.target].to_numpy(float)
    test_prediction["latent_demand"] = test["latent_demand"].to_numpy(float)
    test_prediction.to_csv(output_dir / "metrics" / "frozen_test_predictions.csv", index=False)

    worst_slice_wape, slice_report = _worst_supported_slice_wape(
        test,
        config.forecast.target,
        selected_test,
        config.release_gate.min_slice_count_for_wape,
    )
    slice_report.to_csv(output_dir / "metrics" / "slice_metrics.csv", index=False)
    target_values = test[config.forecast.target].to_numpy(float)
    peak_threshold = float(np.quantile(train[config.forecast.target].to_numpy(float), 0.90))
    peak_mask = target_values >= peak_threshold
    peak_sample_count = int(peak_mask.sum())
    peak_q90_coverage = (
        float(np.mean(target_values[peak_mask] <= selected_test.q90[peak_mask]))
        if peak_sample_count > 0
        else 0.0
    )
    incident_mask = test["regime"].astype(str).eq("incident").to_numpy()
    incident_sample_count = int(incident_mask.sum())
    incident_q90_coverage = (
        float(np.mean(target_values[incident_mask] <= selected_test.q90[incident_mask]))
        if incident_sample_count > 0
        else 0.0
    )
    monitoring_snapshot, monitoring_frame = build_monitoring_snapshot(
        train=train,
        calibration=calibration,
        test=test,
        target=config.forecast.target,
        operational_columns=OPERATIONAL_COLUMNS,
        forecast=selected_test,
        rcwe_support=rcwe_support,
        rcwe_low_support=rcwe_low_support,
    )
    write_json(output_dir / "metrics" / "monitoring_snapshot.json", monitoring_snapshot)
    monitoring_frame.to_csv(output_dir / "metrics" / "monitoring_metrics.csv", index=False)

    # Residual correlation is estimated only from the calibration partition.
    calibration_reference = calibration[["timestamp", "region", "skill"]].copy()
    calibration_reference["leaf_key"] = (
        calibration_reference["region"].astype(str)
        + "::"
        + calibration_reference["skill"].astype(str)
    )
    calibration_reference["residual"] = (
        calibration[config.forecast.target].to_numpy(float)
        - calibration_forecasts[selected_variant].q50
    )
    for operational_column in OPERATIONAL_COLUMNS:
        calibration_reference[operational_column] = calibration[operational_column].to_numpy(float)
    test_timestamps = sorted(pd.to_datetime(test["timestamp"], utc=True).unique())
    simulation_steps = max(
        2,
        int(config.queue.simulation_hours * 60 / config.data.interval_minutes),
    )
    decision_steps = min(
        config.forecast.horizon_intervals,
        simulation_steps,
        max(2, len(test_timestamps) // 2),
    )
    if len(test_timestamps) < 2 * decision_steps:
        decision_steps = max(2, len(test_timestamps) // 2)
    policy_tuning_timestamps = [
        pd.Timestamp(value) for value in test_timestamps[-2 * decision_steps : -decision_steps]
    ]
    future_timestamps = [pd.Timestamp(value) for value in test_timestamps[-decision_steps:]]
    if len(policy_tuning_timestamps) < 2 or len(future_timestamps) < 2:
        raise PipelineError(
            "Test partition is too short for separated policy tuning and frozen evaluation"
        )

    selected_rcwe = rcwe_objects.get(selected_base_name) if "+rcwe" in selected_variant else None
    (
        tuning_horizon,
        tuning_forecast,
        tuning_forecast_metrics,
        _,
        _,
        tuning_origin,
    ) = _prepare_decision_horizon(
        intervals=bundle.intervals,
        future_timestamps=policy_tuning_timestamps,
        model=selected_model,
        calibrator=calibrators[selected_variant],
        rcwe=selected_rcwe,
        config=config,
    )
    (
        horizon,
        horizon_forecast,
        fixed_origin_metrics,
        fixed_support,
        fixed_low_support,
        decision_origin,
    ) = _prepare_decision_horizon(
        intervals=bundle.intervals,
        future_timestamps=future_timestamps,
        model=selected_model,
        calibrator=calibrators[selected_variant],
        rcwe=selected_rcwe,
        config=config,
    )

    test_metrics_rows.extend(
        [
            {
                "variant": f"{selected_variant}:policy_tuning_fixed_origin",
                "evaluation_target": config.forecast.target,
                **tuning_forecast_metrics,
            },
            {
                "variant": f"{selected_variant}:frozen_fixed_origin_recursive",
                "evaluation_target": config.forecast.target,
                **fixed_origin_metrics,
            },
        ]
    )
    test_metrics = pd.DataFrame(test_metrics_rows)
    fixed_origin_prediction = _prediction_frame(horizon, horizon_forecast, "fixed_origin")
    fixed_origin_prediction[config.forecast.target] = horizon[config.forecast.target].to_numpy(
        float
    )
    fixed_origin_prediction["latent_demand"] = horizon["latent_demand"].to_numpy(float)
    fixed_origin_prediction["rcwe_support"] = fixed_support
    fixed_origin_prediction["rcwe_low_support"] = fixed_low_support.astype(int)
    fixed_origin_prediction.to_csv(
        output_dir / "metrics" / "fixed_origin_predictions.csv",
        index=False,
    )

    _mark(output_dir, "08_slices_complete")
    scenario_bundle = generate_coherent_scenarios(
        horizon,
        horizon_forecast,
        config.scenarios.count,
        config.project.seed,
        residual_history=calibration_reference,
        operational_history=calibration_reference,
        correlation_shrinkage=config.scenarios.correlation_shrinkage,
    )
    np.save(output_dir / "metrics" / "coherent_leaf_scenarios.npy", scenario_bundle.leaf_samples)
    np.save(
        output_dir / "metrics" / "scenario_aht_multipliers.npy", scenario_bundle.aht_multipliers
    )
    np.save(
        output_dir / "metrics" / "scenario_patience_multipliers.npy",
        scenario_bundle.patience_multipliers,
    )
    np.save(
        output_dir / "metrics" / "scenario_shrinkage_rates.npy", scenario_bundle.shrinkage_rates
    )
    scenario_bundle.aggregate_samples.to_csv(
        output_dir / "metrics" / "aggregate_scenarios.csv",
        index=False,
    )
    scenario_coherence_error = float(
        np.max(
            np.abs(
                scenario_bundle.leaf_samples.sum(axis=2).reshape(-1)
                - scenario_bundle.aggregate_samples["global_total"].to_numpy(float)
            )
        )
    )
    write_json(
        output_dir / "metrics" / "scenario_diagnostics.json",
        {
            "scenario_count": config.scenarios.count,
            "horizon_intervals": len(future_timestamps),
            "leaf_count": len(scenario_bundle.leaf_keys),
            "temporal_autocorrelation": scenario_bundle.temporal_autocorrelation,
            "cross_sectional_correlation_min": float(
                np.min(scenario_bundle.cross_sectional_correlation)
            ),
            "cross_sectional_correlation_max": float(
                np.max(scenario_bundle.cross_sectional_correlation)
            ),
            "global_coherence_max_abs_error": scenario_coherence_error,
            "operational_skill_keys": scenario_bundle.operational_skill_keys,
            "aht_multiplier_min": float(np.min(scenario_bundle.aht_multipliers)),
            "aht_multiplier_max": float(np.max(scenario_bundle.aht_multipliers)),
            "patience_multiplier_min": float(np.min(scenario_bundle.patience_multipliers)),
            "patience_multiplier_max": float(np.max(scenario_bundle.patience_multipliers)),
            "shrinkage_rate_min": float(np.min(scenario_bundle.shrinkage_rates)),
            "shrinkage_rate_max": float(np.max(scenario_bundle.shrinkage_rates)),
            "operational_diagnostics": scenario_bundle.operational_diagnostics,
        },
    )

    scenario_requirements = _capacity_requirements_from_scenarios(
        scenario_bundle.leaf_samples,
        scenario_bundle.leaf_keys,
        horizon,
        config.data.skills,
        config.data.interval_minutes,
        config.queue,
        config.optimization.shrinkage_buffer,
        config.queue.staffing_load_quantile,
        operational_skill_keys=scenario_bundle.operational_skill_keys,
        aht_multipliers=scenario_bundle.aht_multipliers,
        patience_multipliers=scenario_bundle.patience_multipliers,
        shrinkage_rates=scenario_bundle.shrinkage_rates,
    )
    _mark(output_dir, "09_scenarios_generated")
    planning_horizon_hours = len(future_timestamps) * config.data.interval_minutes / 60.0
    capacity_plan = StrategicCapacityPlanner(
        regular_cost=config.optimization.regular_hourly_cost * planning_horizon_hours,
        shortage_penalty=config.optimization.shortage_penalty * planning_horizon_hours,
        excess_penalty=config.optimization.excess_capacity_penalty * planning_horizon_hours,
        time_limit_seconds=config.optimization.solver_time_limit_seconds,
    ).solve(scenario_requirements, config.data.skills)
    capacity_frame = pd.DataFrame(
        capacity_plan.to_records(planning_horizon_hours=planning_horizon_hours)
    )
    capacity_frame.to_csv(output_dir / "metrics" / "strategic_capacity_plan.csv", index=False)

    _mark(output_dir, "10_capacity_planned")
    safety_policy_name = (
        "rcwe_probabilistic_safety" if "+rcwe" in selected_variant else "probabilistic_safety"
    )
    tuning_base_results, tuning_schedules, tuning_requirements = evaluate_policy_candidates(
        horizon=tuning_horizon,
        horizon_forecast=tuning_forecast,
        historical_reference=supervised[
            pd.to_datetime(supervised["timestamp"], utc=True) <= tuning_origin
        ],
        agents=bundle.agents,
        config=config,
        safety_policy_name=safety_policy_name,
        seed=config.project.seed + 100_000,
        replications=config.queue.policy_tuning_replications,
    )
    tuning_base_results.sort_values("total_cost").to_csv(
        output_dir / "metrics" / "policy_selection_replay_base.csv",
        index=False,
    )
    tuning_policy_results, _, _ = evaluate_recourse_aware_policy_candidates(
        base_results=tuning_base_results,
        schedules=tuning_schedules,
        requirements=tuning_requirements,
        horizon=tuning_horizon,
        agents=bundle.agents,
        config=config,
        seed=config.project.seed + 200_000,
        replications=config.queue.policy_tuning_replications,
    )
    selected_policy, policy_selection_fallback_used = select_policy_from_replay(
        tuning_policy_results
    )
    tuning_policy_results["selected_for_frozen_evaluation"] = (
        tuning_policy_results["base_policy"] == selected_policy
    )
    tuning_policy_results.sort_values("total_cost").to_csv(
        output_dir / "metrics" / "policy_selection_replay.csv",
        index=False,
    )

    policy_results, policy_schedules, policy_requirements = evaluate_policy_candidates(
        horizon=horizon,
        horizon_forecast=horizon_forecast,
        historical_reference=supervised[
            pd.to_datetime(supervised["timestamp"], utc=True) <= decision_origin
        ],
        agents=bundle.agents,
        config=config,
        safety_policy_name=safety_policy_name,
        seed=config.project.seed,
        replications=config.queue.replications,
    )
    _mark(output_dir, "11_policies_simulated")
    non_reference = policy_results[
        policy_results["policy"] != "realized_offered_staffing_reference"
    ]
    hindsight_best_cost = float(non_reference["total_cost"].min())
    hindsight_best_policy = str(non_reference.sort_values("total_cost").iloc[0]["policy"])
    policy_results["cost_gap_vs_hindsight_best_candidate"] = (
        policy_results["total_cost"] - hindsight_best_cost
    )
    policy_results["selected_from_policy_replay"] = policy_results["policy"] == selected_policy
    policy_results = policy_results.sort_values("total_cost").reset_index(drop=True)

    selected_final_row = policy_results[policy_results["policy"] == selected_policy].iloc[0]
    reference_row = policy_results[
        policy_results["policy"] == "realized_offered_staffing_reference"
    ].iloc[0]
    fixed_cost = float(
        policy_results.loc[policy_results["policy"] == "fixed_ratio", "total_cost"].iloc[0]
    )
    point_cost = float(
        policy_results.loc[policy_results["policy"] == "point_forecast", "total_cost"].iloc[0]
    )
    safety_cost = float(
        policy_results.loc[policy_results["policy"] == safety_policy_name, "total_cost"].iloc[0]
    )
    decision_diagnostics = {
        "policy_selection_protocol": "recourse_aware_preceding_replay_then_frozen_evaluation",
        "policy_tuning_origin": str(tuning_origin),
        "frozen_evaluation_origin": str(decision_origin),
        "selected_policy": selected_policy,
        "policy_selection_fallback_used": policy_selection_fallback_used,
        "selected_policy_final_cost": float(selected_final_row["total_cost"]),
        "selected_policy_final_eligible": bool(selected_final_row["eligible_for_selection"]),
        "hindsight_best_candidate_policy": hindsight_best_policy,
        "hindsight_best_candidate_cost": hindsight_best_cost,
        "selection_cost_gap_vs_hindsight_best": float(
            selected_final_row["total_cost"] - hindsight_best_cost
        ),
        "fixed_ratio_cost": fixed_cost,
        "point_forecast_cost": point_cost,
        "probabilistic_safety_policy": safety_policy_name,
        "probabilistic_safety_cost": safety_cost,
        "incremental_cost_point_minus_fixed": point_cost - fixed_cost,
        "incremental_cost_probabilistic_minus_point": safety_cost - point_cost,
        "realized_offered_staffing_reference_cost": float(reference_row["total_cost"]),
        "selected_cost_minus_realized_reference": float(
            selected_final_row["total_cost"] - reference_row["total_cost"]
        ),
        "note": (
            "The realized-offered staffing reference is a diagnostic heuristic, not a global "
            "perfect-information oracle; VSS and EVPI are intentionally not claimed."
        ),
    }
    write_json(output_dir / "metrics" / "decision_diagnostics.json", decision_diagnostics)

    selected_required_coverage = policy_requirements[selected_policy]
    planned_capacity_by_skill = dict(
        zip(
            capacity_plan.skills,
            np.rint(capacity_plan.regular_capacity_units).astype(int),
            strict=True,
        )
    )
    strategic_bridge_rows: list[dict[str, object]] = []
    aligned_coverage = 0
    total_tactical_coverage = 0
    for skill in config.data.skills:
        tactical_peak = max(
            int(selected_required_coverage.get((shift, skill), 0)) for shift in ["early", "late"]
        )
        planned_units = int(planned_capacity_by_skill.get(skill, 0))
        primary_agents = int((bundle.agents["primary_skill"].astype(str) == skill).sum())
        eligible_agents = int(
            bundle.agents["skills"]
            .astype(str)
            .str.split("|")
            .map(lambda values, skill=skill: skill in values)
            .sum()
        )
        aligned_coverage += min(planned_units, tactical_peak)
        total_tactical_coverage += tactical_peak
        strategic_bridge_rows.append(
            {
                "skill": skill,
                "strategic_capacity_units": planned_units,
                "tactical_peak_required_units": tactical_peak,
                "strategic_minus_tactical": planned_units - tactical_peak,
                "current_primary_agents": primary_agents,
                "current_eligible_agents": eligible_agents,
                "primary_hiring_or_cross_training_gap": max(planned_units - primary_agents, 0),
                "maximum_eligibility_gap": max(planned_units - eligible_agents, 0),
                "planning_horizon_hours": planning_horizon_hours,
            }
        )
    strategic_tactical_alignment = (
        1.0 if total_tactical_coverage == 0 else aligned_coverage / total_tactical_coverage
    )
    strategic_bridge = pd.DataFrame(strategic_bridge_rows)
    strategic_bridge.to_csv(output_dir / "metrics" / "strategic_tactical_bridge.csv", index=False)

    pre_recourse_schedule = policy_schedules[selected_policy]
    pre_recourse_schedule.to_csv(
        output_dir / "metrics" / "selected_agent_schedule_pre_recourse.csv",
        index=False,
    )
    write_json(
        output_dir / "metrics" / "selected_required_coverage.json",
        {
            f"{shift}::{skill}": value
            for (shift, skill), value in policy_requirements[selected_policy].items()
        },
    )
    realized_required = policy_requirements.get(
        "realized_offered_staffing_reference",
        policy_requirements[selected_policy],
    )
    write_json(
        output_dir / "metrics" / "realized_required_coverage.json",
        {f"{shift}::{skill}": value for (shift, skill), value in realized_required.items()},
    )
    shift_duration_hours = planning_horizon_hours / 2.0
    repair = apply_intraday_recourse(
        schedule=pre_recourse_schedule,
        agents=bundle.agents,
        required_coverage=realized_required,
        regular_hourly_cost=config.optimization.regular_hourly_cost,
        overtime_hourly_cost=config.optimization.overtime_hourly_cost,
        shift_duration_hours=shift_duration_hours,
    )
    recourse_frame = repair.actions
    applied_recourse_actions = int(recourse_frame.loc[recourse_frame["amount"] > 0, "amount"].sum())
    recourse_decision_rows = int(len(recourse_frame))
    pre_recourse_assigned_shifts = max(int(pre_recourse_schedule["assigned"].sum()), 1)
    recourse_action_rate = applied_recourse_actions / pre_recourse_assigned_shifts
    recourse_positive_cost = float(
        recourse_frame.loc[recourse_frame["estimated_cost"] > 0, "estimated_cost"].sum()
    )
    recourse_frame.to_csv(
        output_dir / "metrics" / "intraday_recourse_actions.csv",
        index=False,
    )
    repair.schedule.to_csv(
        output_dir / "metrics" / "selected_agent_schedule.csv",
        index=False,
    )
    deployed_policy = f"{selected_policy}+intraday_recourse"
    deployed_result = evaluate_existing_schedule(
        policy_name=deployed_policy,
        horizon=horizon,
        schedule=repair.schedule,
        agents=bundle.agents,
        required=realized_required,
        interval_minutes=config.data.interval_minutes,
        service_level_seconds=config.queue.service_level_seconds,
        service_level_target=config.queue.service_level_target,
        regular_hourly_cost=config.optimization.regular_hourly_cost,
        overtime_hourly_cost=config.optimization.overtime_hourly_cost,
        shortage_penalty=config.optimization.shortage_penalty,
        seed=config.project.seed,
        replications=config.queue.replications,
        shift_duration_hours=shift_duration_hours,
    )
    decision_diagnostics.update(
        {
            "deployed_policy": deployed_policy,
            "recourse_applied_action_count": applied_recourse_actions,
            "recourse_decision_rows": recourse_decision_rows,
            "recourse_remaining_hard_violations": repair.remaining_hard_violations,
            "recourse_action_rate": recourse_action_rate,
            "recourse_positive_cost": recourse_positive_cost,
            "recourse_cost_share": recourse_positive_cost
            / max(float(deployed_result.labor_cost), 1.0),
            "deployed_policy_cost": deployed_result.total_cost,
            "deployed_policy_schedule_feasibility": deployed_result.schedule_feasibility,
            "deployed_policy_service_level_lcb95": deployed_result.service_level_lcb95,
            "deployed_policy_abandonment_rate_ucb95": deployed_result.abandonment_rate_ucb95,
        }
    )
    write_json(output_dir / "metrics" / "decision_diagnostics.json", decision_diagnostics)
    deployed_row = deployed_result.to_dict()
    deployed_row.update(
        {
            "eligible_for_selection": False,
            "eligible_for_release": bool(
                deployed_result.schedule_feasibility >= 1.0
                and deployed_result.hard_violations == 0
                and deployed_result.flow_conservation
            ),
            "cost_gap_vs_hindsight_best_candidate": float(
                deployed_result.total_cost - hindsight_best_cost
            ),
            "selected_from_policy_replay": False,
            "post_recourse": True,
            "recourse_applied_action_count": applied_recourse_actions,
            "recourse_action_rate": recourse_action_rate,
            "recourse_positive_cost": recourse_positive_cost,
            "recourse_cost_share": recourse_positive_cost
            / max(float(deployed_result.labor_cost), 1.0),
        }
    )
    policy_results["post_recourse"] = False
    policy_results["eligible_for_release"] = policy_results["eligible_for_selection"]
    policy_results = (
        pd.concat(
            [policy_results, pd.DataFrame([deployed_row])],
            ignore_index=True,
            sort=False,
        )
        .sort_values("total_cost")
        .reset_index(drop=True)
    )
    policy_results.to_csv(output_dir / "metrics" / "policy_comparison.csv", index=False)

    selected_policy_row = policy_results[policy_results["policy"] == deployed_policy].iloc[0]
    decision_metrics = {
        "capacity_plan_success": bool(capacity_plan.success),
        "strategic_tactical_alignment": float(strategic_tactical_alignment),
        "schedule_feasibility": float(selected_policy_row["schedule_feasibility"]),
        "abandonment_rate": float(selected_policy_row["abandonment_rate"]),
        "abandonment_rate_ucb95": float(selected_policy_row["abandonment_rate_ucb95"]),
        "service_level_lcb95": float(selected_policy_row["service_level_lcb95"]),
        "p95_wait_seconds_ucb95": float(selected_policy_row["p95_wait_seconds_ucb95"]),
        "hard_violations": int(selected_policy_row["hard_violations"]),
        "flow_conservation": bool(selected_policy_row["flow_conservation"]),
        "recourse_action_rate": recourse_action_rate,
        "recourse_cost_share": recourse_positive_cost / max(float(deployed_result.labor_cost), 1.0),
    }
    release_forecast_metrics = dict(selected_metrics)
    release_forecast_metrics["worst_slice_wape"] = worst_slice_wape
    release_forecast_metrics["peak_q90_coverage"] = peak_q90_coverage
    release_forecast_metrics["incident_q90_coverage"] = incident_q90_coverage
    release_forecast_metrics["peak_sample_count"] = peak_sample_count
    release_forecast_metrics["incident_sample_count"] = incident_sample_count
    release_forecast_metrics["fixed_origin_wape"] = fixed_origin_metrics["wape"]
    release_forecast_metrics["fixed_origin_coverage_error"] = fixed_origin_metrics["coverage_error"]
    release_forecast_metrics["scenario_coherence_max_abs_error"] = scenario_coherence_error
    release_thresholds = config.release_gate.model_dump()
    release_thresholds["min_service_level_lcb95"] = config.queue.service_level_target
    recourse_required_for_release = not bool(selected_final_row["eligible_for_selection"])
    release_decision = evaluate_release_gate(
        release_forecast_metrics,
        decision_metrics,
        release_thresholds,
        recourse_required=recourse_required_for_release,
        post_recourse_eligible=bool(selected_policy_row["eligible_for_release"]),
    )
    decision_diagnostics.update(
        {
            "recourse_required_for_release": recourse_required_for_release,
            "recourse_release_override_applied": bool(
                release_decision.status == "PASS_WITH_RECOURSE"
                and any(
                    not check.passed and check.name == "strategic_tactical_alignment"
                    for check in release_decision.checks
                )
            ),
            "recourse_release_override_checks": [
                check.name
                for check in release_decision.checks
                if not check.passed and check.name == "strategic_tactical_alignment"
            ],
            "recourse_semantics": (
                "two-stage frozen-scenario recoverability evaluation using realized offered "
                "demand; not a causal real-time production rollout"
            ),
        }
    )
    write_json(output_dir / "metrics" / "decision_diagnostics.json", decision_diagnostics)

    _mark(output_dir, "12_release_gate")
    hardware_summary = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "forecast_hardware": hardware_reports,
        "chronos_availability": asdict(Chronos2Adapter.availability()),
    }
    write_json(output_dir / "hardware_profile.json", hardware_summary)
    validation_leaderboard.to_csv(
        output_dir / "metrics" / "validation_leaderboard.csv", index=False
    )
    test_metrics.to_csv(output_dir / "metrics" / "frozen_test_metrics.csv", index=False)

    summary = {
        "project_name": config.project.name,
        "seed": config.project.seed,
        "selected_variant": selected_variant,
        "model_bundle": model_bundle_manifest,
        "monitoring": monitoring_snapshot,
        "selected_policy_from_replay": selected_policy,
        "deployed_policy": deployed_policy,
        "selected_policy": deployed_policy,  # backward-compatible alias for downstream consumers
        "release_status": release_decision.status,
        "rows": {
            "intervals": len(bundle.intervals),
            "contacts": len(bundle.contacts),
            "agents": len(bundle.agents),
            "supervised": len(supervised),
        },
        "selected_forecast_metrics": selected_metrics,
        "forecast_tail_metrics": {
            "minimum_slice_count_for_wape": config.release_gate.min_slice_count_for_wape,
            "worst_supported_slice_wape": worst_slice_wape,
            "peak_q90_coverage": peak_q90_coverage,
            "incident_q90_coverage": incident_q90_coverage,
            "peak_sample_count": peak_sample_count,
            "incident_sample_count": incident_sample_count,
        },
        "fixed_origin_forecast_metrics": fixed_origin_metrics,
        "forecast_protocol": {
            "validation": "temporal_holdout_one_step_with_observed_history",
            "decision": "fixed_origin_recursive",
            "decision_origin": str(decision_origin),
            "decision_horizon_intervals": decision_steps,
        },
        "selected_decision_metrics": selected_policy_row.to_dict(),
        "capacity_plan_success": capacity_plan.success,
        "capacity_plan_status": capacity_plan.status,
        "strategic_tactical_alignment": strategic_tactical_alignment,
        "planning_horizon_hours": planning_horizon_hours,
        "selected_rcwe_applied": bool("+rcwe" in selected_variant),
        "mean_rcwe_support": float(np.mean(rcwe_support)) if "+rcwe" in selected_variant else None,
        "low_rcwe_support_rate": float(np.mean(rcwe_low_support))
        if "+rcwe" in selected_variant
        else None,
        "decision_diagnostics": decision_diagnostics,
        "intraday_recourse_actions": applied_recourse_actions,
        "intraday_recourse_decision_rows": recourse_decision_rows,
        "recourse_remaining_hard_violations": repair.remaining_hard_violations,
        "claim_boundary": "offline synthetic operational replay; no live customer or AWS data",
    }
    _mark(output_dir, "13_before_reports")
    write_reports(
        output_dir,
        summary,
        validation_leaderboard,
        test_metrics,
        slice_report,
        policy_results,
        capacity_frame,
        release_decision.to_dict(),
    )
    recourse_markdown = (
        "# Intraday Recourse Report\n\n"
        + recourse_frame.to_markdown(index=False)
        + "\n\nThe actions were applied in a two-stage frozen-scenario recoverability experiment and "
        "the repaired schedule was replayed against the frozen realized horizon. This is "
        "not claimed as a causal real-time rollout.\n"
    )
    (output_dir / "reports" / "intraday_recourse_report.md").write_text(
        recourse_markdown, encoding="utf-8"
    )
    _mark(output_dir, "14_complete")
    return summary


def _source_tree_hash() -> str:
    package_root = Path(__file__).resolve().parent
    payload: dict[str, str] = {}
    for path in sorted(package_root.rglob("*.py")):
        payload[str(path.relative_to(package_root))] = stable_hash(path.read_text(encoding="utf-8"))
    project_root = package_root.parents[1]
    for name in ["pyproject.toml", "Makefile"]:
        path = project_root / name
        if path.is_file():
            payload[name] = stable_hash(path.read_text(encoding="utf-8"))
    return stable_hash(payload)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _acquire_run_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        stale = False
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            stale = not _pid_is_alive(int(payload.get("pid", -1)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            stale = False
        if stale:
            lock_path.unlink(missing_ok=True)
            return _acquire_run_lock(lock_path)
        raise PipelineBusyError(f"Another pipeline run owns lock: {lock_path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "pid": os.getpid(),
                "created_at_utc": datetime.now(UTC).isoformat(),
            },
            handle,
        )


def _promote_staging_output(staging: Path, final: Path) -> None:
    backup = final.parent / f".{final.name}.backup-{uuid.uuid4().hex}"
    had_previous = final.exists()
    if had_previous:
        final.rename(backup)
    try:
        staging.rename(final)
    except Exception:
        if had_previous and backup.exists() and not final.exists():
            backup.rename(final)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    """Run into a staging directory and atomically publish only successful outputs."""
    config = load_config(config_path)
    final_output = Path(config.project.output_dir).expanduser().resolve()
    final_output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = final_output.parent / f".{final_output.name}.lock"
    staging = final_output.parent / (
        f".{final_output.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    _acquire_run_lock(lock_path)
    try:
        for stale_staging in final_output.parent.glob(f".{final_output.name}.staging-*"):
            shutil.rmtree(stale_staging, ignore_errors=True)
        staged_project = config.project.model_copy(update={"output_dir": str(staging)})
        staged_config = config.model_copy(update={"project": staged_project})
        run_id = uuid.uuid4().hex
        config_hash = stable_hash(config.model_dump(mode="json"))
        source_tree_hash = _source_tree_hash()
        summary = _run_pipeline_impl(staged_config)
        summary["reproducibility"] = {
            "run_id": run_id,
            "config_hash": config_hash,
            "source_tree_hash": source_tree_hash,
            "git_sha": os.environ.get("GITHUB_SHA"),
            "python_version": platform.python_version(),
        }
        write_json(staging / "run_summary.json", summary)
        manifest_payload = {
            "status": "SUCCESS",
            "run_id": run_id,
            "started_from_config": str(Path(config_path)),
            "published_output": str(final_output),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "config_hash": config_hash,
            "source_tree_hash": source_tree_hash,
            "git_sha": os.environ.get("GITHUB_SHA"),
            "python_version": platform.python_version(),
            "summary_hash": stable_hash(summary),
        }
        write_json(staging / "run_manifest.json", manifest_payload)
        artifact_contract = validate_run_artifacts(staging)
        manifest_payload["artifact_contract"] = artifact_contract
        write_json(staging / "run_manifest.json", manifest_payload)
        _promote_staging_output(staging, final_output)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        lock_path.unlink(missing_ok=True)
