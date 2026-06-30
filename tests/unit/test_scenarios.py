import numpy as np
import pandas as pd

from support_capacity_reliability.forecasting.base import ForecastOutput
from support_capacity_reliability.reliability.scenarios import generate_coherent_scenarios


def test_scenarios_are_nonnegative_and_coherent():
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")] * 2,
            "region": ["north", "south"],
            "skill": ["billing", "billing"],
        }
    )
    forecast = ForecastOutput("x", np.array([1, 2]), np.array([3, 4]), np.array([5, 6]))
    bundle = generate_coherent_scenarios(frame, forecast, 8, seed=1)
    assert bundle.leaf_samples.shape == (8, 1, 2)
    assert np.all(bundle.leaf_samples >= 0)
    totals = bundle.leaf_samples.sum(axis=2).reshape(-1)
    assert np.allclose(totals, bundle.aggregate_samples["global_total"].to_numpy())
    assert np.isfinite(bundle.temporal_autocorrelation)
    assert bundle.cross_sectional_correlation.shape == (2, 2)
    assert bundle.aht_multipliers.shape == (8, 1, 1)
    assert bundle.patience_multipliers.shape == (8, 1, 1)
    assert bundle.shrinkage_rates.shape == (8, 1, 1)
    assert np.all(bundle.aht_multipliers > 0)
    assert np.all((bundle.shrinkage_rates > 0) & (bundle.shrinkage_rates < 1))


def test_scenario_aggregate_order_matches_sample_tensor_for_multiple_times():
    timestamps = [
        pd.Timestamp("2026-01-01 00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 00:30", tz="UTC"),
    ]
    frame = pd.DataFrame(
        {
            "timestamp": [timestamps[0], timestamps[0], timestamps[1], timestamps[1]],
            "region": ["north", "south", "north", "south"],
            "skill": ["billing", "billing", "billing", "billing"],
        }
    )
    forecast = ForecastOutput(
        "x",
        np.array([1, 2, 2, 3], dtype=float),
        np.array([3, 4, 4, 5], dtype=float),
        np.array([5, 6, 6, 7], dtype=float),
    )
    bundle = generate_coherent_scenarios(frame, forecast, 5, seed=3)
    tensor_totals = bundle.leaf_samples.sum(axis=2).reshape(-1)
    aggregate_totals = bundle.aggregate_samples["global_total"].to_numpy(float)
    assert np.allclose(tensor_totals, aggregate_totals)


def test_operational_scenarios_use_history_and_remain_bounded():
    timestamps = pd.date_range("2026-01-01", periods=6, freq="30min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps[-2:], 2),
            "region": ["north", "south"] * 2,
            "skill": ["billing"] * 4,
            "planning_average_handle_time_seconds": [420.0] * 4,
            "planning_patience_mean_seconds": [240.0] * 4,
            "planning_shrinkage_rate": [0.12] * 4,
        }
    )
    forecast = ForecastOutput(
        "x",
        np.array([2, 2, 3, 3], dtype=float),
        np.array([4, 4, 5, 5], dtype=float),
        np.array([6, 6, 7, 7], dtype=float),
    )
    history = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, 2),
            "region": ["north", "south"] * len(timestamps),
            "skill": ["billing"] * (2 * len(timestamps)),
            "leaf_key": ["north::billing", "south::billing"] * len(timestamps),
            "residual": np.tile(np.linspace(-2, 2, len(timestamps)), 2),
            "average_handle_time_seconds": np.tile(np.linspace(380, 500, len(timestamps)), 2),
            "patience_mean_seconds": np.tile(np.linspace(270, 210, len(timestamps)), 2),
            "shrinkage_rate": np.tile(np.linspace(0.08, 0.22, len(timestamps)), 2),
        }
    )
    bundle = generate_coherent_scenarios(
        frame, forecast, 12, seed=7, residual_history=history, operational_history=history
    )
    assert bundle.operational_diagnostics["billing"]["history_rows"] == 6
    assert float(bundle.aht_multipliers.std()) > 0
    assert float(bundle.patience_multipliers.std()) > 0
    assert np.all((bundle.shrinkage_rates >= 0.01) & (bundle.shrinkage_rates <= 0.55))
