import numpy as np
import pandas as pd

from support_capacity_reliability.config import load_config
from support_capacity_reliability.pipeline import _capacity_requirements_from_scenarios
from support_capacity_reliability.queueing.erlang import required_agents_erlang_a


def test_capacity_scenarios_sum_regions_within_each_interval_not_across_time():
    config = load_config("configs/smoke.yaml")
    # One scenario, two intervals, two regional leaves for the same skill.
    samples = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    horizon = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                pd.Timestamp("2026-01-01 00:30", tz="UTC"),
            ],
            "skill": ["billing", "billing"],
            "planning_average_handle_time_seconds": [420.0, 420.0],
            "planning_patience_mean_seconds": [240.0, 240.0],
            "planning_shrinkage_rate": [0.10, 0.10],
        }
    )
    quantile = 0.5
    result = _capacity_requirements_from_scenarios(
        samples,
        ["north::billing", "south::billing"],
        horizon,
        ["billing"],
        30,
        config.queue,
        0.0,
        quantile,
        operational_skill_keys=["billing"],
        aht_multipliers=np.ones((1, 2, 1)),
        patience_multipliers=np.ones((1, 2, 1)),
        shrinkage_rates=np.full((1, 2, 1), 0.10),
    )
    # Correct interval totals are [3, 7], whose median is 5. The prior indexing bug
    # produced leaf totals [4, 6], which happens to share the median; use q=0.85 below
    # to ensure axis mistakes are detectable.
    quantile = 0.85
    result = _capacity_requirements_from_scenarios(
        samples,
        ["north::billing", "south::billing"],
        horizon,
        ["billing"],
        30,
        config.queue,
        0.0,
        quantile,
        operational_skill_keys=["billing"],
        aht_multipliers=np.ones((1, 2, 1)),
        patience_multipliers=np.ones((1, 2, 1)),
        shrinkage_rates=np.full((1, 2, 1), 0.10),
    )
    contacts = float(np.quantile(np.array([3.0, 7.0]), quantile))
    approximation = required_agents_erlang_a(
        arrival_rate_per_second=contacts / (30 * 60),
        average_handle_time_seconds=420.0,
        patience_mean_seconds=240.0,
        service_level_target=config.queue.service_level_target,
        abandonment_target=config.queue.abandonment_target,
        service_level_seconds=config.queue.service_level_seconds,
        max_agents=config.queue.max_agents_per_pool,
    )
    expected = np.ceil(approximation.agents / 0.90)
    assert result[0, 0] == expected
