from support_capacity_reliability.optimization.recourse import intraday_recourse


def test_recourse_activates_reserve_then_overtime():
    actions = intraday_recourse(
        scheduled_agents=2,
        required_agents=6,
        reserve_available=3,
        overtime_hourly_cost=50,
    )
    assert actions[0].action == "activate_reserve"
    assert actions[0].amount == 3
    assert actions[1].action == "offer_overtime"
    assert actions[1].amount == 1


def test_recourse_holds_when_within_tolerance():
    actions = intraday_recourse(
        scheduled_agents=5,
        required_agents=4,
        reserve_available=0,
        overtime_hourly_cost=50,
    )
    assert len(actions) == 1
    assert actions[0].action == "hold_schedule"


def test_schedule_repair_closes_gap_when_eligible_agent_exists():
    import pandas as pd

    from support_capacity_reliability.optimization.recourse import apply_intraday_recourse

    agents = pd.DataFrame(
        [
            {
                "agent_id": "a1",
                "skills": "billing",
                "preferred_shift": "early",
                "proficiency": 1.0,
            },
            {
                "agent_id": "a2",
                "skills": "billing",
                "preferred_shift": "late",
                "proficiency": 0.9,
            },
        ]
    )
    schedule = pd.DataFrame(
        [
            {"agent_id": "a1", "shift": "early", "assigned": 1, "assigned_skill": "billing"},
            {"agent_id": "a1", "shift": "late", "assigned": 0, "assigned_skill": None},
            {"agent_id": "a2", "shift": "early", "assigned": 0, "assigned_skill": None},
            {"agent_id": "a2", "shift": "late", "assigned": 0, "assigned_skill": None},
        ]
    )
    result = apply_intraday_recourse(
        schedule=schedule,
        agents=agents,
        required_coverage={("early", "billing"): 2},
        regular_hourly_cost=30,
        overtime_hourly_cost=50,
    )
    assert result.remaining_hard_violations == 0
    assert int(result.schedule["assigned"].sum()) == 2
    assert "activate_reserve" in set(result.actions["action"])


def test_schedule_repair_applies_vto_to_material_excess():
    import pandas as pd

    from support_capacity_reliability.optimization.recourse import apply_intraday_recourse

    agents = pd.DataFrame(
        [
            {
                "agent_id": f"a{idx}",
                "skills": "billing",
                "preferred_shift": "early",
                "proficiency": 1.0,
                "regular_hourly_cost": 30.0,
                "overtime_hourly_cost": 50.0,
            }
            for idx in range(5)
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "agent_id": f"a{idx}",
                "shift": "early",
                "assigned": 1,
                "assigned_skill": "billing",
            }
            for idx in range(5)
        ]
    )
    result = apply_intraday_recourse(
        schedule=schedule,
        agents=agents,
        required_coverage={("early", "billing"): 2},
        regular_hourly_cost=30,
        overtime_hourly_cost=50,
    )
    assert result.remaining_hard_violations == 0
    assert int(result.schedule["assigned"].sum()) == 2
    assert set(result.actions["action"]) == {"apply_voluntary_time_off"}


def test_schedule_repair_respects_overtime_eligibility_and_daily_hours():
    import pandas as pd

    from support_capacity_reliability.optimization.recourse import apply_intraday_recourse

    agents = pd.DataFrame(
        [
            {
                "agent_id": "ineligible",
                "skills": "billing",
                "preferred_shift": "early",
                "available_shifts": "early|late",
                "overtime_eligible": False,
                "max_daily_hours": 6,
                "proficiency": 1.1,
                "regular_hourly_cost": 30.0,
                "overtime_hourly_cost": 50.0,
            },
            {
                "agent_id": "eligible",
                "skills": "billing",
                "preferred_shift": "late",
                "available_shifts": "early|late",
                "overtime_eligible": True,
                "max_daily_hours": 12,
                "proficiency": 1.0,
                "regular_hourly_cost": 30.0,
                "overtime_hourly_cost": 50.0,
            },
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "agent_id": agent,
                "shift": shift,
                "assigned": int(shift == "early"),
                "assigned_skill": "billing" if shift == "early" else None,
            }
            for agent in ["ineligible", "eligible"]
            for shift in ["early", "late"]
        ]
    )
    result = apply_intraday_recourse(
        schedule=schedule,
        agents=agents,
        required_coverage={("late", "billing"): 1},
        regular_hourly_cost=30,
        overtime_hourly_cost=50,
        shift_duration_hours=6.0,
    )
    late = result.schedule[
        (result.schedule["shift"] == "late") & (result.schedule["assigned"] == 1)
    ]
    assert late["agent_id"].tolist() == ["eligible"]
    assert result.actions.loc[result.actions["amount"] > 0, "estimated_cost"].iloc[0] == 300.0


def test_schedule_repair_reassigns_cross_trained_surplus_before_overtime():
    import pandas as pd

    from support_capacity_reliability.optimization.recourse import apply_intraday_recourse

    agents = pd.DataFrame(
        [
            {
                "agent_id": "cross_trained",
                "skills": "billing|technical",
                "preferred_shift": "early",
                "available_shifts": "early",
                "overtime_eligible": False,
                "max_daily_hours": 6,
                "proficiency": 1.0,
                "regular_hourly_cost": 30.0,
                "overtime_hourly_cost": 50.0,
            },
            {
                "agent_id": "billing_only",
                "skills": "billing",
                "preferred_shift": "early",
                "available_shifts": "early",
                "overtime_eligible": False,
                "max_daily_hours": 6,
                "proficiency": 1.0,
                "regular_hourly_cost": 30.0,
                "overtime_hourly_cost": 50.0,
            },
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "agent_id": "cross_trained",
                "shift": "early",
                "assigned": 1,
                "assigned_skill": "billing",
            },
            {
                "agent_id": "billing_only",
                "shift": "early",
                "assigned": 1,
                "assigned_skill": "billing",
            },
        ]
    )
    result = apply_intraday_recourse(
        schedule=schedule,
        agents=agents,
        required_coverage={
            ("early", "billing"): 1,
            ("early", "technical"): 1,
        },
        regular_hourly_cost=30,
        overtime_hourly_cost=50,
        shift_duration_hours=6.0,
    )
    assert result.remaining_hard_violations == 0
    assert "cross_skill_reassignment" in set(result.actions["action"])
    assert "offer_overtime" not in set(result.actions["action"])
    selected = result.schedule[result.schedule["assigned"] == 1]
    assert set(selected["assigned_skill"]) == {"billing", "technical"}


def test_schedule_repair_does_not_activate_reserve_beyond_daily_hour_limit():
    import pandas as pd

    from support_capacity_reliability.optimization.recourse import apply_intraday_recourse

    agents = pd.DataFrame(
        [
            {
                "agent_id": "reserve",
                "skills": "billing",
                "preferred_shift": "early",
                "available_shifts": "early",
                "overtime_eligible": True,
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
        regular_hourly_cost=30,
        overtime_hourly_cost=50,
        shift_duration_hours=12.0,
    )
    assert result.remaining_hard_violations == 1
    assert result.schedule["assigned"].sum() == 0
