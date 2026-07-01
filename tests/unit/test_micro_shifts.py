from __future__ import annotations

import pandas as pd

from support_capacity_reliability.evaluation.decision import build_shift_mapping
from support_capacity_reliability.optimization.recourse import apply_intraday_recourse
from support_capacity_reliability.optimization.scheduler import TacticalShiftScheduler


def _horizon(intervals: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=intervals,
                freq="30min",
                tz="UTC",
            )
        }
    )


def test_legacy_shift_mapping_preserves_two_shift_contract() -> None:
    mapping = build_shift_mapping(_horizon(48))

    shifts = list(dict.fromkeys(mapping.values()))

    assert shifts == ["early", "late"]
    assert list(mapping.values()).count("early") == 24
    assert list(mapping.values()).count("late") == 24


def test_six_hour_micro_shift_mapping_uses_band_labels() -> None:
    mapping = build_shift_mapping(
        _horizon(48),
        interval_minutes=30,
        configured_shift_duration_hours=6,
    )

    shifts = list(dict.fromkeys(mapping.values()))

    assert shifts == ["early_1", "early_2", "late_1", "late_2"]
    assert list(mapping.values()).count("early_1") == 12
    assert list(mapping.values()).count("early_2") == 12
    assert list(mapping.values()).count("late_1") == 12
    assert list(mapping.values()).count("late_2") == 12


def test_scheduler_applies_daily_hours_and_band_availability() -> None:
    agents = pd.DataFrame(
        [
            {
                "agent_id": "six_hour",
                "skills": "billing",
                "preferred_shift": "early",
                "available_shifts": "early",
                "overtime_eligible": False,
                "max_daily_hours": 6,
                "regular_hourly_cost": 30,
            },
            {
                "agent_id": "twelve_hour",
                "skills": "billing",
                "preferred_shift": "early",
                "available_shifts": "early",
                "overtime_eligible": True,
                "max_daily_hours": 12,
                "regular_hourly_cost": 30,
            },
        ]
    )
    shifts = ["early_1", "early_2", "early_3"]
    required = {(shift, "billing"): 1 for shift in shifts}

    result = TacticalShiftScheduler(time_limit_seconds=5).solve(
        agents,
        required,
        shifts,
        ["billing"],
        shift_duration_hours=6,
        allow_multiple_shifts_per_agent=True,
    )

    selected = result.schedule[result.schedule["assigned"] == 1]
    selected_counts = selected.groupby("agent_id").size().to_dict()

    assert result.hard_violations == 0
    assert selected_counts == {"six_hour": 1, "twelve_hour": 2}
    assert set(selected["shift"]) == set(shifts)


def test_recourse_uses_band_availability_for_micro_shift_reserve() -> None:
    agents = pd.DataFrame(
        [
            {
                "agent_id": "reserve",
                "skills": "billing",
                "preferred_shift": "early",
                "available_shifts": "early",
                "overtime_eligible": False,
                "max_daily_hours": 6,
                "proficiency": 1.0,
                "regular_hourly_cost": 30.0,
                "overtime_hourly_cost": 50.0,
            }
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "agent_id": "reserve",
                "shift": "early_2",
                "assigned": 0,
                "assigned_skill": None,
            }
        ]
    )

    result = apply_intraday_recourse(
        schedule=schedule,
        agents=agents,
        required_coverage={("early_2", "billing"): 1},
        regular_hourly_cost=30.0,
        overtime_hourly_cost=50.0,
        shift_duration_hours=6.0,
    )

    assert result.remaining_hard_violations == 0
    assert result.schedule.loc[0, "assigned"] == 1
    assert result.actions.loc[result.actions["amount"] > 0, "action"].tolist() == [
        "activate_reserve"
    ]


def test_recourse_records_unrepaired_undercoverage() -> None:
    agents = pd.DataFrame(
        [
            {
                "agent_id": "short_limit",
                "skills": "billing",
                "preferred_shift": "early",
                "available_shifts": "early",
                "overtime_eligible": False,
                "max_daily_hours": 6,
                "proficiency": 1.0,
                "regular_hourly_cost": 30.0,
                "overtime_hourly_cost": 50.0,
            }
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "agent_id": "short_limit",
                "shift": "early",
                "assigned": 0,
                "assigned_skill": None,
            }
        ]
    )

    result = apply_intraday_recourse(
        schedule=schedule,
        agents=agents,
        required_coverage={("early", "billing"): 1},
        regular_hourly_cost=30.0,
        overtime_hourly_cost=50.0,
        shift_duration_hours=12.0,
    )

    unrepaired = result.actions[result.actions["action"].eq("unrepaired_undercoverage")]

    assert result.remaining_hard_violations == 1
    assert unrepaired["amount"].tolist() == [1]
    assert unrepaired["estimated_cost"].tolist() == [0.0]
