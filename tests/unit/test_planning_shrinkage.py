import pandas as pd

from support_capacity_reliability.evaluation.decision import build_required_coverage


def _frame(shrinkage: float) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=4, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "skill": ["billing"] * 4,
            "prediction": [6.0] * 4,
            "planning_average_handle_time_seconds": [420.0] * 4,
            "planning_patience_mean_seconds": [240.0] * 4,
            "planning_shrinkage_rate": [shrinkage] * 4,
        }
    )


def test_higher_planned_shrinkage_never_reduces_required_coverage():
    common = dict(
        prediction_column="prediction",
        skills=["billing"],
        interval_minutes=30,
        service_level_seconds=120,
        service_level_target=0.8,
        abandonment_target=0.12,
        max_agents=30,
        shrinkage_buffer=0.05,
    )
    low, _ = build_required_coverage(_frame(0.05), **common)
    high, _ = build_required_coverage(_frame(0.35), **common)
    assert high[("early", "billing")] >= low[("early", "billing")]
    assert high[("late", "billing")] >= low[("late", "billing")]
