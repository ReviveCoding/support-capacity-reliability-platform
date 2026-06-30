import math

import pandas as pd
import pytest

from support_capacity_reliability.evaluation.decision import evaluate_existing_schedule


def _inputs():
    horizon = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")],
            "skill": ["billing"],
            "offered_contacts": [0],
            "average_handle_time_seconds": [420.0],
            "patience_mean_seconds": [240.0],
        }
    )
    schedule = pd.DataFrame(
        [
            {
                "agent_id": "a1",
                "shift": "early",
                "assigned": 1,
                "assigned_skill": "billing",
                "assignment_type": "regular",
            }
        ]
    )
    agents = pd.DataFrame(
        [
            {
                "agent_id": "a1",
                "skills": "billing",
                "proficiency": 1.0,
                "regular_hourly_cost": 30.0,
                "overtime_hourly_cost": 50.0,
            }
        ]
    )
    return horizon, schedule, agents


def test_single_replication_is_not_treated_as_precise():
    horizon, schedule, agents = _inputs()
    result = evaluate_existing_schedule(
        policy_name="x",
        horizon=horizon,
        schedule=schedule,
        agents=agents,
        required={("early", "billing"): 1},
        interval_minutes=30,
        service_level_seconds=120,
        service_level_target=0.8,
        regular_hourly_cost=30,
        overtime_hourly_cost=50,
        shortage_penalty=100,
        seed=1,
        replications=1,
    )
    assert result.service_level_lcb95 == 0.0
    assert result.abandonment_rate_ucb95 == 1.0
    assert math.isinf(result.total_cost_ucb95)
    assert math.isinf(result.p95_wait_seconds_ucb95)


def test_multiple_replications_produce_finite_small_sample_bounds():
    horizon, schedule, agents = _inputs()
    result = evaluate_existing_schedule(
        policy_name="x",
        horizon=horizon,
        schedule=schedule,
        agents=agents,
        required={("early", "billing"): 1},
        interval_minutes=30,
        service_level_seconds=120,
        service_level_target=0.8,
        regular_hourly_cost=30,
        overtime_hourly_cost=50,
        shortage_penalty=100,
        seed=1,
        replications=3,
    )
    assert math.isfinite(result.total_cost_ucb95)
    assert 0 <= result.service_level_lcb95 <= 1
    assert 0 <= result.abandonment_rate_ucb95 <= 1
    assert math.isfinite(result.p95_wait_seconds_ucb95)
    assert result.p95_wait_seconds_ucb95 >= result.p95_wait_seconds


def test_replication_statistics_use_sample_std_and_all_flow_conservation(monkeypatch):
    import support_capacity_reliability.evaluation.decision as decision_module

    horizon, schedule, agents = _inputs()
    rows = iter(
        [
            {
                "offered": 10.0,
                "served": 10.0,
                "abandoned": 0.0,
                "abandonment_rate": 0.0,
                "service_level": 1.0,
                "service_level_answered": 1.0,
                "average_wait_seconds": 0.0,
                "p95_wait_seconds": 0.0,
                "utilization": 0.1,
                "flow_conservation": 1.0,
            },
            {
                "offered": 10.0,
                "served": 9.0,
                "abandoned": 0.0,
                "abandonment_rate": 0.0,
                "service_level": 0.9,
                "service_level_answered": 1.0,
                "average_wait_seconds": 5.0,
                "p95_wait_seconds": 10.0,
                "utilization": 0.2,
                "flow_conservation": 0.0,
            },
            {
                "offered": 10.0,
                "served": 10.0,
                "abandoned": 0.0,
                "abandonment_rate": 0.0,
                "service_level": 0.8,
                "service_level_answered": 1.0,
                "average_wait_seconds": 10.0,
                "p95_wait_seconds": 20.0,
                "utilization": 0.3,
                "flow_conservation": 1.0,
            },
        ]
    )
    monkeypatch.setattr(decision_module, "_simulate_once", lambda **_: next(rows))
    result = decision_module.evaluate_existing_schedule(
        policy_name="x",
        horizon=horizon,
        schedule=schedule,
        agents=agents,
        required={("early", "billing"): 1},
        interval_minutes=30,
        service_level_seconds=120,
        service_level_target=0.8,
        regular_hourly_cost=30,
        overtime_hourly_cost=50,
        shortage_penalty=100,
        seed=1,
        replications=3,
    )
    assert result.p95_wait_seconds_std == 10.0
    assert not result.flow_conservation


def test_replication_schema_drift_fails_loudly(monkeypatch):
    import support_capacity_reliability.evaluation.decision as decision_module

    horizon, schedule, agents = _inputs()
    monkeypatch.setattr(
        decision_module,
        "_simulate_once",
        lambda **_: {
            "offered": 1.0,
            "served": 1.0,
            "abandoned": 0.0,
            # service_level deliberately omitted
            "service_level_answered": 1.0,
            "abandonment_rate": 0.0,
            "average_wait_seconds": 0.0,
            "p95_wait_seconds": 0.0,
            "utilization": 0.1,
            "flow_conservation": 1.0,
        },
    )
    with pytest.raises(ValueError, match="missing metrics"):
        decision_module.evaluate_existing_schedule(
            policy_name="x",
            horizon=horizon,
            schedule=schedule,
            agents=agents,
            required={("early", "billing"): 1},
            interval_minutes=30,
            service_level_seconds=120,
            service_level_target=0.8,
            regular_hourly_cost=30,
            overtime_hourly_cost=50,
            shortage_penalty=100,
            seed=1,
            replications=2,
        )
