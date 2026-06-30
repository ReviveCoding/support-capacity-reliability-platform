import pandas as pd

from support_capacity_reliability.queueing.simulator import MultiSkillVoiceSimulator, SimAgent


def test_simulator_conserves_flow_and_is_deterministic():
    plan = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")],
            "skill": ["billing"],
            "offered_contacts": [8],
            "average_handle_time_seconds": [120.0],
            "patience_mean_seconds": [300.0],
        }
    )
    agents = [SimAgent("a1", {"billing"}, 1.0), SimAgent("a2", {"billing"}, 1.0)]
    first = MultiSkillVoiceSimulator(seed=3).run(plan, agents)
    second = MultiSkillVoiceSimulator(seed=3).run(plan, agents)
    assert first.flow_conservation
    assert first.offered == first.served + first.abandoned
    assert first.__dict__ == second.__dict__


def test_schedule_assignment_restricts_cross_trained_agent_to_assigned_skill():
    from support_capacity_reliability.queueing.simulator import agents_from_schedule

    schedule = pd.DataFrame(
        [
            {
                "agent_id": "a1",
                "shift": "early",
                "assigned": 1,
                "assigned_skill": "billing",
            }
        ]
    )
    registry = pd.DataFrame(
        [
            {
                "agent_id": "a1",
                "skills": "billing|technical",
                "proficiency": 1.0,
            }
        ]
    )
    agents = agents_from_schedule(schedule, registry, "early")
    assert len(agents) == 1
    assert agents[0].skills == {"billing"}


def test_shift_closure_prevents_new_answers_after_staffed_horizon():
    plan = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")],
            "skill": ["billing"],
            "offered_contacts": [20],
            "average_handle_time_seconds": [1800.0],
            "patience_mean_seconds": [14400.0],
        }
    )
    result = MultiSkillVoiceSimulator(seed=9).run(
        plan,
        [SimAgent("a1", {"billing"}, 1.0)],
        interval_minutes=30,
    )
    assert result.flow_conservation
    assert result.served + result.abandoned == 20
    assert result.served == 1
    assert all(wait <= 30 * 60 for wait in result.waits)
