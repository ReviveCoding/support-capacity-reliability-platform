from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from support_capacity_reliability.optimization.scheduler import TacticalShiftScheduler
from support_capacity_reliability.queueing.erlang import required_agents_erlang_a
from support_capacity_reliability.queueing.simulator import (
    MultiSkillVoiceSimulator,
    agents_from_schedule,
)


@dataclass
class PolicyResult:
    policy: str
    assigned_agent_shifts: int
    schedule_feasibility: float
    offered: int
    served: int
    abandoned: int
    abandonment_rate: float
    service_level: float
    service_level_answered: float
    service_level_lcb95: float
    abandonment_rate_ucb95: float
    average_wait_seconds: float
    p95_wait_seconds: float
    p95_wait_seconds_ucb95: float
    utilization: float
    flow_conservation: bool
    labor_cost: float
    service_penalty: float
    abandonment_penalty: float
    total_cost: float
    hard_violations: int
    replications: int = 1
    total_cost_std: float = 0.0
    total_cost_ucb95: float = float("inf")
    service_level_std: float = 0.0
    abandonment_rate_std: float = 0.0
    p95_wait_seconds_std: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_shift_mapping(horizon: pd.DataFrame) -> dict[pd.Timestamp, str]:
    timestamps = sorted(pd.to_datetime(horizon["timestamp"], utc=True).unique())
    midpoint = max(1, len(timestamps) // 2)
    return {
        pd.Timestamp(timestamp): ("early" if idx < midpoint else "late")
        for idx, timestamp in enumerate(timestamps)
    }


def build_required_coverage(
    horizon: pd.DataFrame,
    prediction_column: str,
    skills: list[str],
    interval_minutes: int,
    service_level_seconds: float,
    service_level_target: float,
    abandonment_target: float,
    max_agents: int,
    shrinkage_buffer: float,
    staffing_load_quantile: float = 0.85,
) -> tuple[dict[tuple[str, str], int], dict[pd.Timestamp, str]]:
    mapping = build_shift_mapping(horizon)
    working = horizon.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    working["shift"] = working["timestamp"].map(mapping)
    required: dict[tuple[str, str], int] = {}
    for shift in ["early", "late"]:
        shift_frame = working[working["shift"] == shift]
        for skill in skills:
            group = shift_frame[shift_frame["skill"] == skill]
            if group.empty:
                required[(shift, skill)] = 0
                continue
            contacts_by_interval = (
                group.groupby("timestamp")[prediction_column].sum().to_numpy(float)
            )
            contacts_per_interval = float(np.quantile(contacts_by_interval, staffing_load_quantile))
            arrival_rate = contacts_per_interval / (interval_minutes * 60.0)
            aht = float(
                np.average(
                    group["planning_average_handle_time_seconds"],
                    weights=np.maximum(group[prediction_column], 0.1),
                )
            )
            patience = float(group["planning_patience_mean_seconds"].median())
            approximation = required_agents_erlang_a(
                arrival_rate_per_second=arrival_rate,
                average_handle_time_seconds=aht,
                patience_mean_seconds=patience,
                service_level_target=service_level_target,
                abandonment_target=abandonment_target,
                service_level_seconds=service_level_seconds,
                max_agents=max_agents,
            )
            planned_shrinkage = float(
                np.average(
                    group["planning_shrinkage_rate"],
                    weights=np.maximum(group[prediction_column], 0.1),
                )
            )
            effective_availability = max(
                (1.0 - planned_shrinkage) * (1.0 - shrinkage_buffer),
                0.1,
            )
            buffered = int(np.ceil(approximation.agents / effective_availability))
            required[(shift, skill)] = buffered
    return required, mapping


def schedule_coverage(
    schedule: pd.DataFrame,
    required: dict[tuple[str, str], int],
) -> tuple[dict[tuple[str, str], int], int, float]:
    selected = schedule[schedule["assigned"] == 1]
    achieved = selected.groupby(["shift", "assigned_skill"]).size().to_dict()
    hard_violations = int(
        sum(
            max(requirement - int(achieved.get(key, 0)), 0) for key, requirement in required.items()
        )
    )
    total = int(sum(required.values()))
    met = int(
        sum(min(int(achieved.get(key, 0)), requirement) for key, requirement in required.items())
    )
    feasibility = 1.0 if total == 0 else met / total
    return achieved, hard_violations, feasibility


def _simulate_once(
    horizon: pd.DataFrame,
    schedule: pd.DataFrame,
    agents: pd.DataFrame,
    shift_mapping: dict[pd.Timestamp, str],
    interval_minutes: int,
    service_level_seconds: float,
    seed: int,
) -> dict[str, float]:
    working = horizon.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    working["shift"] = working["timestamp"].map(shift_mapping)
    offered_total = served_total = abandoned_total = 0
    wait_values: list[float] = []
    weighted_service = 0.0
    weighted_answered_service = 0.0
    weighted_utilization = 0.0
    simulation_weight = 0
    answered_weight = 0
    all_flow_conserved = True
    for shift_idx, shift in enumerate(["early", "late"]):
        shift_frame = working[working["shift"] == shift]
        if shift_frame.empty:
            continue
        aggregated = shift_frame.groupby(["timestamp", "skill"], as_index=False).agg(
            offered_contacts=("offered_contacts", "sum"),
            average_handle_time_seconds=("average_handle_time_seconds", "mean"),
            patience_mean_seconds=("patience_mean_seconds", "mean"),
        )
        scheduled_agents = agents_from_schedule(schedule, agents, shift)
        simulator = MultiSkillVoiceSimulator(
            service_level_seconds=service_level_seconds,
            seed=seed + shift_idx,
        )
        result = simulator.run(aggregated, scheduled_agents, interval_minutes=interval_minutes)
        offered_total += result.offered
        served_total += result.served
        abandoned_total += result.abandoned
        wait_values.extend(result.waits)
        weighted_service += result.service_level * result.offered
        weighted_answered_service += result.service_level_answered * result.served
        weighted_utilization += result.utilization * result.offered
        simulation_weight += result.offered
        answered_weight += result.served
        all_flow_conserved = all_flow_conserved and result.flow_conservation
    wait_array = np.asarray(wait_values, dtype=float)
    return {
        "offered": float(offered_total),
        "served": float(served_total),
        "abandoned": float(abandoned_total),
        "abandonment_rate": abandoned_total / max(offered_total, 1),
        "service_level": weighted_service / max(simulation_weight, 1),
        "service_level_answered": weighted_answered_service / max(answered_weight, 1),
        "average_wait_seconds": float(wait_array.mean()) if len(wait_array) else 0.0,
        "p95_wait_seconds": float(np.quantile(wait_array, 0.95)) if len(wait_array) else 0.0,
        "utilization": weighted_utilization / max(simulation_weight, 1),
        "flow_conservation": float(all_flow_conserved),
    }


def _labor_cost(
    schedule: pd.DataFrame,
    agents: pd.DataFrame,
    regular_hourly_cost: float,
    overtime_hourly_cost: float,
    shift_duration_hours: float,
) -> float:
    selected = schedule[schedule["assigned"] == 1].copy()
    if selected.empty:
        return 0.0
    if "assignment_type" not in selected.columns:
        selected["assignment_type"] = "regular"
    cost_columns = ["agent_id"]
    if "regular_hourly_cost" in agents.columns:
        cost_columns.append("regular_hourly_cost")
    if "overtime_hourly_cost" in agents.columns:
        cost_columns.append("overtime_hourly_cost")
    selected = selected.merge(
        agents[cost_columns].assign(agent_id=lambda frame: frame["agent_id"].astype(str)),
        on="agent_id",
        how="left",
    )
    selected["regular_hourly_cost"] = selected.get(
        "regular_hourly_cost", pd.Series(index=selected.index, dtype=float)
    ).fillna(regular_hourly_cost)
    selected["overtime_hourly_cost"] = selected.get(
        "overtime_hourly_cost", pd.Series(index=selected.index, dtype=float)
    ).fillna(overtime_hourly_cost)
    # Recourse rows are simulated as available for the full shift, so labor cost uses the
    # same horizon-derived shift duration as the queue simulation and tactical optimizer.
    hourly = np.where(
        selected["assignment_type"].eq("overtime"),
        selected["overtime_hourly_cost"],
        selected["regular_hourly_cost"],
    )
    return float(np.asarray(hourly, dtype=float).sum() * shift_duration_hours)


def evaluate_existing_schedule(
    *,
    policy_name: str,
    horizon: pd.DataFrame,
    schedule: pd.DataFrame,
    agents: pd.DataFrame,
    required: dict[tuple[str, str], int],
    interval_minutes: int,
    service_level_seconds: float,
    service_level_target: float,
    regular_hourly_cost: float,
    overtime_hourly_cost: float,
    shortage_penalty: float,
    seed: int,
    replications: int,
    shift_duration_hours: float | None = None,
) -> PolicyResult:
    shift_mapping = build_shift_mapping(horizon)
    _, hard_violations, feasibility = schedule_coverage(schedule, required)
    assigned_shifts = int(schedule["assigned"].sum())
    if shift_duration_hours is None:
        interval_count = max(len(pd.to_datetime(horizon["timestamp"], utc=True).unique()), 1)
        shift_duration_hours = interval_count * interval_minutes / 60.0 / 2.0
    labor_cost = _labor_cost(
        schedule,
        agents,
        regular_hourly_cost,
        overtime_hourly_cost,
        shift_duration_hours,
    )
    replication_rows: list[dict[str, float]] = []
    for replication_id in range(max(1, int(replications))):
        row = _simulate_once(
            horizon=horizon,
            schedule=schedule,
            agents=agents,
            shift_mapping=shift_mapping,
            interval_minutes=interval_minutes,
            service_level_seconds=service_level_seconds,
            seed=seed + 1009 * replication_id,
        )
        required_simulation_metrics = {
            "offered",
            "served",
            "abandoned",
            "abandonment_rate",
            "service_level",
            "service_level_answered",
            "average_wait_seconds",
            "p95_wait_seconds",
            "utilization",
            "flow_conservation",
        }
        missing_simulation_metrics = sorted(required_simulation_metrics - row.keys())
        if missing_simulation_metrics:
            raise ValueError(
                f"Simulation replication output is missing metrics: {missing_simulation_metrics}"
            )
        if not np.isfinite(
            np.asarray([row[name] for name in sorted(required_simulation_metrics)], dtype=float)
        ).all():
            raise ValueError("Simulation replication output contains non-finite metrics")
        service_penalty = (
            max(service_level_target - row["service_level"], 0.0) * row["offered"] * 60.0
        )
        abandonment_penalty = row["abandoned"] * 35.0
        violation_penalty = hard_violations * shortage_penalty
        row["total_cost"] = labor_cost + service_penalty + abandonment_penalty + violation_penalty
        row["service_penalty"] = service_penalty
        row["abandonment_penalty"] = abandonment_penalty
        replication_rows.append(row)

    rep_frame = pd.DataFrame(replication_rows)
    required_replication_metrics = {
        "offered",
        "served",
        "abandoned",
        "abandonment_rate",
        "service_level",
        "service_level_answered",
        "average_wait_seconds",
        "p95_wait_seconds",
        "utilization",
        "flow_conservation",
        "total_cost",
        "service_penalty",
        "abandonment_penalty",
    }
    missing_metrics = sorted(required_replication_metrics - set(rep_frame.columns))
    if missing_metrics:
        raise ValueError(f"Simulation replication output is missing metrics: {missing_metrics}")
    numeric_metrics = rep_frame[sorted(required_replication_metrics)].to_numpy(dtype=float)
    if not np.isfinite(numeric_metrics).all():
        raise ValueError("Simulation replication output contains non-finite metrics")
    mean_row = rep_frame.mean(numeric_only=True).to_dict()
    std_row = rep_frame.std(ddof=1, numeric_only=True).fillna(0.0).to_dict()
    all_flows_conserved = bool(rep_frame["flow_conservation"].eq(1.0).all())
    replication_count = max(1, int(replications))
    if replication_count < 2:
        # A single stochastic replay cannot support a variance-based confidence bound.
        # Use conservative bounds so one-replication experiments cannot appear precise.
        service_level_lcb95 = 0.0
        abandonment_rate_ucb95 = 1.0
        total_cost_ucb95 = float("inf")
        p95_wait_seconds_ucb95 = float("inf")
    else:
        # One-sided Student-t bounds are appropriate for the small Monte Carlo samples used
        # in smoke and policy-tuning runs. A normal 1.96 multiplier is overconfident here.
        multiplier = float(student_t.ppf(0.95, df=replication_count - 1)) / np.sqrt(
            replication_count
        )
        service_level_lcb95 = max(
            0.0,
            float(mean_row.get("service_level", 0.0))
            - multiplier * float(std_row.get("service_level", 0.0)),
        )
        abandonment_rate_ucb95 = min(
            1.0,
            float(mean_row.get("abandonment_rate", 0.0))
            + multiplier * float(std_row.get("abandonment_rate", 0.0)),
        )
        total_cost_ucb95 = float(mean_row.get("total_cost", labor_cost)) + multiplier * float(
            std_row.get("total_cost", 0.0)
        )
        p95_wait_seconds_ucb95 = max(
            0.0,
            float(mean_row.get("p95_wait_seconds", 0.0))
            + multiplier * float(std_row.get("p95_wait_seconds", 0.0)),
        )
    return PolicyResult(
        policy=policy_name,
        assigned_agent_shifts=assigned_shifts,
        schedule_feasibility=feasibility,
        offered=int(round(mean_row.get("offered", 0.0))),
        served=int(round(mean_row.get("served", 0.0))),
        abandoned=int(round(mean_row.get("abandoned", 0.0))),
        abandonment_rate=float(mean_row.get("abandonment_rate", 0.0)),
        service_level=float(mean_row.get("service_level", 0.0)),
        service_level_answered=float(mean_row.get("service_level_answered", 0.0)),
        service_level_lcb95=service_level_lcb95,
        abandonment_rate_ucb95=abandonment_rate_ucb95,
        average_wait_seconds=float(mean_row.get("average_wait_seconds", 0.0)),
        p95_wait_seconds=float(mean_row.get("p95_wait_seconds", 0.0)),
        p95_wait_seconds_ucb95=p95_wait_seconds_ucb95,
        utilization=float(mean_row.get("utilization", 0.0)),
        flow_conservation=all_flows_conserved,
        labor_cost=labor_cost,
        service_penalty=float(mean_row.get("service_penalty", 0.0)),
        abandonment_penalty=float(mean_row.get("abandonment_penalty", 0.0)),
        total_cost=float(mean_row.get("total_cost", labor_cost)),
        hard_violations=hard_violations,
        replications=replication_count,
        total_cost_std=float(std_row.get("total_cost", 0.0)),
        total_cost_ucb95=total_cost_ucb95,
        service_level_std=float(std_row.get("service_level", 0.0)),
        abandonment_rate_std=float(std_row.get("abandonment_rate", 0.0)),
        p95_wait_seconds_std=float(std_row.get("p95_wait_seconds", 0.0)),
    )


def evaluate_staffing_policy(
    policy_name: str,
    horizon: pd.DataFrame,
    prediction_column: str,
    agents: pd.DataFrame,
    skills: list[str],
    interval_minutes: int,
    service_level_seconds: float,
    service_level_target: float,
    abandonment_target: float,
    max_agents: int,
    shrinkage_buffer: float,
    scheduler_time_limit: int,
    preference_penalty: int,
    regular_hourly_cost: float,
    overtime_hourly_cost: float,
    shortage_penalty: float,
    seed: int,
    replications: int = 1,
    staffing_load_quantile: float = 0.85,
) -> tuple[PolicyResult, pd.DataFrame, dict[tuple[str, str], int]]:
    required, _ = build_required_coverage(
        horizon,
        prediction_column,
        skills,
        interval_minutes,
        service_level_seconds,
        service_level_target,
        abandonment_target,
        max_agents,
        shrinkage_buffer,
        staffing_load_quantile,
    )
    shift_duration_hours = (
        max(len(pd.to_datetime(horizon["timestamp"], utc=True).unique()), 1)
        * interval_minutes
        / 60.0
        / 2.0
    )
    scheduler = TacticalShiftScheduler(
        time_limit_seconds=scheduler_time_limit,
        preference_penalty=preference_penalty,
        shortage_penalty=max(1000, int(shortage_penalty * 10)),
    )
    schedule_result = scheduler.solve(
        agents,
        required,
        ["early", "late"],
        skills,
        shift_duration_hours=shift_duration_hours,
    )
    schedule = schedule_result.schedule.copy()
    schedule["assignment_type"] = np.where(schedule["assigned"] == 1, "regular", "unassigned")
    policy = evaluate_existing_schedule(
        policy_name=policy_name,
        horizon=horizon,
        schedule=schedule,
        agents=agents,
        required=required,
        interval_minutes=interval_minutes,
        service_level_seconds=service_level_seconds,
        service_level_target=service_level_target,
        regular_hourly_cost=regular_hourly_cost,
        overtime_hourly_cost=overtime_hourly_cost,
        shortage_penalty=shortage_penalty,
        seed=seed,
        replications=replications,
        shift_duration_hours=shift_duration_hours,
    )
    return policy, schedule, required
