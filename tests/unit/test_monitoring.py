from __future__ import annotations

import numpy as np
import pandas as pd

from support_capacity_reliability.forecasting.base import ForecastOutput
from support_capacity_reliability.monitoring import (
    build_monitoring_snapshot,
    population_stability_index,
)


def test_population_stability_index_detects_shift():
    reference = np.linspace(0, 1, 1000)
    stable = reference.copy()
    shifted = reference + 3.0
    assert population_stability_index(reference, stable) < 0.01
    assert population_stability_index(reference, shifted) > 0.25


def test_monitoring_snapshot_contains_drift_and_reliability_signals():
    train = pd.DataFrame(
        {
            "target": np.linspace(1, 10, 100),
            "average_handle_time_seconds": np.linspace(300, 500, 100),
        }
    )
    calibration = train.copy()
    test = pd.DataFrame(
        {
            "target": np.linspace(3, 12, 100),
            "average_handle_time_seconds": np.linspace(450, 700, 100),
        }
    )
    forecast = ForecastOutput(
        "x",
        q10=test["target"].to_numpy(float) - 1,
        q50=test["target"].to_numpy(float),
        q90=test["target"].to_numpy(float) + 1,
    )
    snapshot, frame = build_monitoring_snapshot(
        train=train,
        calibration=calibration,
        test=test,
        target="target",
        operational_columns=["average_handle_time_seconds"],
        forecast=forecast,
        rcwe_support=np.full(100, 0.8),
        rcwe_low_support=np.zeros(100, dtype=bool),
    )
    assert snapshot["schema_version"] == "1.0"
    assert snapshot["maximum_test_psi"] > 0
    assert snapshot["low_rcwe_support_rate"] == 0.0
    assert {"psi", "forecast_signed_bias", "interval_coverage_80"}.issubset(set(frame["metric"]))
