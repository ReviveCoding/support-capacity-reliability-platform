from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from support_capacity_reliability.config import AppConfig
from support_capacity_reliability.evaluation.decision import (
    evaluate_existing_schedule,
    evaluate_staffing_policy,
)
from support_capacity_reliability.forecasting.base import ForecastOutput
from support_capacity_reliability.optimization.recourse import apply_intraday_recourse


def historical_prediction(
    train: pd.DataFrame,
    horizon: pd.DataFrame,
    target: str,
    interval_minutes: int,
) -> np.ndarray:
    train_work = train.copy()
    slots_per_hour = max(1, int(round(60 / interval_minutes)))
    train_time = pd.to_datetime(train_work["timestamp"], utc=True)
    train_work["hour_slot"] = train_time.dt.hour * slots_per_hour + (
        train_time.dt.minute // interval_minutes
    )
    medians = train_work.groupby(["region", "skill", "hour_slot"])[target].median()
    global_median = float(train_work[target].median())
    horizon_time = pd.to_datetime(horizon["timestamp"], utc=True)
    slots = horizon_time.dt.hour * slots_per_hour + horizon_time.dt.minute // interval_minutes
    values = []
    for region, skill, slot in zip(horizon["region"], horizon["skill"], slots, strict=True):
        values.append(float(medians.get((region, skill, slot), global_median)))
    return np.asarray(values, dtype=float)


def evaluate_policy_candidates(
    *,
    horizon: pd.DataFrame,
    horizon_forecast: ForecastOutput,
    historical_reference: pd.DataFrame,
    agents: pd.DataFrame,
    config: AppConfig,
    safety_policy_name: str,
    seed: int,
    replications: int,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, dict[tuple[str, str], int]],
]:
    working = horizon.copy()
    working["fixed_prediction"] = historical_prediction(
        historical_reference,
        working,
        config.forecast.target,
        config.data.interval_minutes,
    )
    working["point_prediction"] = horizon_forecast.q50
    working["probabilistic_prediction"] = horizon_forecast.q90
    working["realized_prediction"] = working["offered_contacts"].to_numpy(float)
    policy_specs = [
        ("fixed_ratio", "fixed_prediction"),
        ("point_forecast", "point_prediction"),
        (safety_policy_name, "probabilistic_prediction"),
        ("realized_offered_staffing_reference", "realized_prediction"),
    ]
    rows: list[dict[str, Any]] = []
    schedules: dict[str, pd.DataFrame] = {}
    requirements: dict[str, dict[tuple[str, str], int]] = {}
    for policy_name, prediction_column in policy_specs:
        result, schedule, required = evaluate_staffing_policy(
            policy_name=policy_name,
            horizon=working,
            prediction_column=prediction_column,
            agents=agents,
            skills=config.data.skills,
            interval_minutes=config.data.interval_minutes,
            service_level_seconds=config.queue.service_level_seconds,
            service_level_target=config.queue.service_level_target,
            abandonment_target=config.queue.abandonment_target,
            max_agents=config.queue.max_agents_per_pool,
            shrinkage_buffer=config.optimization.shrinkage_buffer,
            scheduler_time_limit=config.optimization.solver_time_limit_seconds,
            preference_penalty=int(config.optimization.schedule_preference_penalty),
            regular_hourly_cost=config.optimization.regular_hourly_cost,
            overtime_hourly_cost=config.optimization.overtime_hourly_cost,
            shortage_penalty=config.optimization.shortage_penalty,
            seed=seed,
            replications=replications,
            staffing_load_quantile=config.queue.staffing_load_quantile,
        )
        rows.append(result.to_dict())
        schedules[policy_name] = schedule
        requirements[policy_name] = required
    frame = pd.DataFrame(rows)
    frame["eligible_for_selection"] = (
        (frame["policy"] != "realized_offered_staffing_reference")
        & (frame["schedule_feasibility"] >= config.release_gate.min_schedule_feasibility)
        & (frame["service_level_lcb95"] >= config.queue.service_level_target)
        & (frame["abandonment_rate_ucb95"] <= config.release_gate.max_abandonment_rate)
        & (frame["p95_wait_seconds_ucb95"] <= config.release_gate.max_p95_wait_seconds)
        & frame["flow_conservation"].astype(bool)
    )
    return frame, schedules, requirements


def evaluate_recourse_aware_policy_candidates(
    *,
    base_results: pd.DataFrame,
    schedules: dict[str, pd.DataFrame],
    requirements: dict[str, dict[tuple[str, str], int]],
    horizon: pd.DataFrame,
    agents: pd.DataFrame,
    config: AppConfig,
    seed: int,
    replications: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Evaluate each deployable policy as a base schedule plus bounded recourse."""
    realized_required = requirements["realized_offered_staffing_reference"]
    base_lookup = base_results.set_index("policy")
    rows: list[dict[str, Any]] = []
    repaired_schedules: dict[str, pd.DataFrame] = {}
    action_frames: dict[str, pd.DataFrame] = {}
    shift_duration_hours = (
        max(len(pd.to_datetime(horizon["timestamp"], utc=True).unique()), 1)
        * config.data.interval_minutes
        / 60.0
        / 2.0
    )

    for base_policy, schedule in schedules.items():
        if base_policy == "realized_offered_staffing_reference":
            continue
        repair = apply_intraday_recourse(
            schedule=schedule,
            agents=agents,
            required_coverage=realized_required,
            regular_hourly_cost=config.optimization.regular_hourly_cost,
            overtime_hourly_cost=config.optimization.overtime_hourly_cost,
            shift_duration_hours=shift_duration_hours,
        )
        deployed_name = f"{base_policy}+intraday_recourse"
        result = evaluate_existing_schedule(
            policy_name=deployed_name,
            horizon=horizon,
            schedule=repair.schedule,
            agents=agents,
            required=realized_required,
            interval_minutes=config.data.interval_minutes,
            service_level_seconds=config.queue.service_level_seconds,
            service_level_target=config.queue.service_level_target,
            regular_hourly_cost=config.optimization.regular_hourly_cost,
            overtime_hourly_cost=config.optimization.overtime_hourly_cost,
            shortage_penalty=config.optimization.shortage_penalty,
            seed=seed,
            replications=replications,
            shift_duration_hours=shift_duration_hours,
        )
        actions = repair.actions.copy()
        applied_actions = int(actions.loc[actions["amount"] > 0, "amount"].sum())
        positive_recourse_cost = float(
            actions.loc[actions["estimated_cost"] > 0, "estimated_cost"].sum()
        )
        base_assigned = max(int(base_lookup.loc[base_policy, "assigned_agent_shifts"]), 1)
        action_rate = applied_actions / base_assigned
        cost_share = positive_recourse_cost / max(float(result.labor_cost), 1.0)
        row = result.to_dict()
        row.update(
            {
                "base_policy": base_policy,
                "deployed_policy": deployed_name,
                "recourse_applied_action_count": applied_actions,
                "recourse_action_rate": action_rate,
                "recourse_positive_cost": positive_recourse_cost,
                "recourse_cost_share": cost_share,
                "recourse_remaining_hard_violations": repair.remaining_hard_violations,
            }
        )
        row["eligible_for_selection"] = bool(
            result.schedule_feasibility >= config.release_gate.min_schedule_feasibility
            and result.service_level_lcb95 >= config.queue.service_level_target
            and result.abandonment_rate_ucb95 <= config.release_gate.max_abandonment_rate
            and result.p95_wait_seconds_ucb95 <= config.release_gate.max_p95_wait_seconds
            and result.flow_conservation
            and repair.remaining_hard_violations == 0
            and action_rate <= config.release_gate.max_recourse_action_rate
            and cost_share <= config.release_gate.max_recourse_cost_share
        )
        rows.append(row)
        repaired_schedules[base_policy] = repair.schedule
        action_frames[base_policy] = actions

    return pd.DataFrame(rows), repaired_schedules, action_frames


def select_policy_from_replay(policy_results: pd.DataFrame) -> tuple[str, bool]:
    eligible = policy_results[policy_results["eligible_for_selection"]].copy()
    if not eligible.empty:
        winner = eligible.sort_values(["total_cost_ucb95", "total_cost"]).iloc[0]
        return str(winner.get("base_policy", winner["policy"])), False
    candidates = policy_results[
        policy_results["policy"] != "realized_offered_staffing_reference"
    ].copy()
    candidates["fallback_score"] = (
        candidates["total_cost"]
        + 1_000_000.0 * candidates["hard_violations"]
        + 1_000_000.0 * np.maximum(1.0 - candidates["schedule_feasibility"], 0.0)
    )
    winner = candidates.sort_values("fallback_score").iloc[0]
    return str(winner.get("base_policy", winner["policy"])), True
