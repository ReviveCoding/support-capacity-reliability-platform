import pandas as pd

from support_capacity_reliability.data.contracts import validate_agents, validate_intervals


def _intervals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")],
            "region": ["north"],
            "skill": ["billing"],
            "regime": ["normal"],
            "latent_demand": [2],
            "offered_contacts": [2],
            "served_contacts": [1],
            "abandoned_contacts": [1],
            "average_handle_time_seconds": [420.0],
            "patience_mean_seconds": [240.0],
            "shrinkage_rate": [0.1],
            "offered_load_estimate": [2.0],
            "source_type": ["synthetic_operational"],
        }
    )


def _agents() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "agent_id": ["a1"],
            "home_region": ["north"],
            "skills": ["billing|technical"],
            "primary_skill": ["billing"],
            "proficiency": [1.0],
            "regular_hourly_cost": [30.0],
            "overtime_hourly_cost": [45.0],
            "preferred_shift": ["early"],
            "available_shifts": ["early|late"],
            "overtime_eligible": [True],
            "max_daily_hours": [12.0],
            "absence_probability": [0.05],
            "source_type": ["synthetic_operational"],
        }
    )


def test_interval_contract_requires_pipeline_target_and_operational_bounds():
    frame = _intervals()
    assert validate_intervals(frame).passed
    frame.loc[0, "shrinkage_rate"] = 1.0
    result = validate_intervals(frame)
    assert not result.passed
    assert any("shrinkage_rate" in error for error in result.errors)


def test_agent_contract_rejects_unknown_skill_and_invalid_overtime_cost():
    frame = _agents()
    assert validate_agents(frame, allowed_skills=["billing", "technical"]).passed
    frame.loc[0, "skills"] = "billing|unknown"
    frame.loc[0, "overtime_hourly_cost"] = 20.0
    result = validate_agents(frame, allowed_skills=["billing", "technical"])
    assert not result.passed
    assert any("unsupported skills" in error for error in result.errors)
    assert any("Overtime cost" in error for error in result.errors)
