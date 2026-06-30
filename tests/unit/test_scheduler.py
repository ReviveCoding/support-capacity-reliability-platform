import pandas as pd

from support_capacity_reliability.optimization.scheduler import TacticalShiftScheduler


def test_scheduler_assigns_feasible_skills():
    agents = pd.DataFrame(
        [
            {
                "agent_id": "a1",
                "skills": "billing|technical",
                "preferred_shift": "early",
                "regular_hourly_cost": 30,
            },
            {
                "agent_id": "a2",
                "skills": "technical",
                "preferred_shift": "early",
                "regular_hourly_cost": 30,
            },
            {
                "agent_id": "a3",
                "skills": "billing",
                "preferred_shift": "late",
                "regular_hourly_cost": 30,
            },
            {
                "agent_id": "a4",
                "skills": "technical",
                "preferred_shift": "late",
                "regular_hourly_cost": 30,
            },
        ]
    )
    required = {
        ("early", "billing"): 1,
        ("early", "technical"): 1,
        ("late", "billing"): 1,
        ("late", "technical"): 1,
    }
    result = TacticalShiftScheduler(time_limit_seconds=5).solve(
        agents, required, ["early", "late"], ["billing", "technical"]
    )
    assert result.status in {"OPTIMAL", "FEASIBLE"}
    assert result.feasibility == 1.0
    assert result.hard_violations == 0


def test_scheduler_respects_agent_shift_availability():
    agents = pd.DataFrame(
        [
            {
                "agent_id": "early_only",
                "skills": "billing",
                "preferred_shift": "early",
                "available_shifts": "early",
                "regular_hourly_cost": 30,
            },
            {
                "agent_id": "late_only",
                "skills": "billing",
                "preferred_shift": "late",
                "available_shifts": "late",
                "regular_hourly_cost": 30,
            },
        ]
    )
    result = TacticalShiftScheduler(time_limit_seconds=5).solve(
        agents,
        {("early", "billing"): 1, ("late", "billing"): 1},
        ["early", "late"],
        ["billing"],
        shift_duration_hours=6.0,
    )
    selected = result.schedule[result.schedule["assigned"] == 1]
    assignments = set(zip(selected["agent_id"], selected["shift"], strict=True))
    assert assignments == {("early_only", "early"), ("late_only", "late")}


def test_scheduler_rejects_shift_longer_than_agent_daily_limit():
    agents = pd.DataFrame(
        [
            {
                "agent_id": "short_limit",
                "skills": "billing",
                "preferred_shift": "early",
                "available_shifts": "early",
                "max_daily_hours": 6,
                "regular_hourly_cost": 30,
            }
        ]
    )
    result = TacticalShiftScheduler(time_limit_seconds=5).solve(
        agents,
        {("early", "billing"): 1},
        ["early"],
        ["billing"],
        shift_duration_hours=12.0,
    )
    assert result.hard_violations == 1
    assert result.feasibility == 0.0
    assert result.schedule["assigned"].sum() == 0
