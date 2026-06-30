import numpy as np

from support_capacity_reliability.optimization.capacity import StrategicCapacityPlanner


def test_capacity_plan_covers_typical_demand():
    scenarios = np.array([[3, 4], [4, 5], [5, 6], [4, 7]], dtype=float)
    plan = StrategicCapacityPlanner(regular_cost=100, shortage_penalty=1000).solve(
        scenarios, ["billing", "technical"]
    )
    assert len(plan.regular_fte) == 2
    assert np.all(plan.regular_fte >= 0)
    assert plan.objective_value > 0


def test_capacity_plan_accounts_for_excess_capacity():
    scenarios = np.array([[2], [8]], dtype=float)
    low_excess_penalty = StrategicCapacityPlanner(
        regular_cost=100,
        shortage_penalty=1000,
        excess_penalty=0,
    ).solve(scenarios, ["billing"])
    high_excess_penalty = StrategicCapacityPlanner(
        regular_cost=100,
        shortage_penalty=1000,
        excess_penalty=500,
    ).solve(scenarios, ["billing"])
    assert low_excess_penalty.expected_excess.shape == (1,)
    assert high_excess_penalty.expected_excess.shape == (1,)
    assert high_excess_penalty.regular_fte[0] <= low_excess_penalty.regular_fte[0]
